"""
app/main.py
───────────
FastAPI application entry point.

Endpoints:
  POST /chat                   — Main chat endpoint (RAG pipeline, 100% offline)
  POST /train                  — Trigger model retraining (API-key protected)
  GET  /health                 — Health check + mode (offline/gemini)
  GET  /history/{user_id}      — Paginated conversation history
  GET  /knowledge              — List knowledge base entries
  POST /knowledge              — Add a new knowledge entry
  PUT  /knowledge/{id}         — Update a knowledge entry
  DELETE /knowledge/{id}       — Delete a knowledge entry
  POST /knowledge/sync         — Re-sync all entries to ChromaDB
  GET  /mode                   — Show current LLM mode (offline/gemini)

Middleware:
  - CORS for Flutter / web frontends
  - Request timing + structured logging
  - Static file serving (chat UI + admin panel)
  - Global exception handler (never leaks raw stack traces)
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import Conversation, Message, User, create_db_tables, get_db

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    On startup:
      - Create database tables (idempotent)
      - Warm up the ChromaDB collection
      - Attempt to load the pre-trained classifier

    On shutdown:
      - Log graceful shutdown message
    """
    logger.info("Starting %s v%s...", settings.app_name, settings.app_version)

    # Create DB tables
    try:
        await create_db_tables()
        logger.info("Database tables ready.")
    except Exception as exc:
        logger.error("Failed to create DB tables: %s", exc)

    try:
        from app.embeddings import get_chroma_service
        from app.classifier import get_classifier
        from app.llm_client import get_llm_client
        from app.knowledge import get_knowledge_service

        # Warm up ChromaDB
        try:
            chroma = get_chroma_service()
            doc_count = chroma.count()
            logger.info("ChromaDB warm-up OK — %d documents indexed.", doc_count)
            if doc_count == 0:
                logger.warning(
                    "ChromaDB is empty. Run 'python models/train.py' to index data."
                )
        except Exception as exc:
            logger.error("ChromaDB warm-up failed: %s", exc)

        # Warm up classifier
        try:
            clf = get_classifier()
            if clf.is_trained:
                logger.info(
                    "Classifier loaded. Known intents: %s", clf.classes
                )
            else:
                logger.warning(
                    "Classifier not trained. Run 'python models/train.py' first."
                )
        except Exception as exc:
            logger.error("Classifier warm-up failed: %s", exc)

        # Warm up LLM / show mode
        try:
            llm = get_llm_client()
            logger.info("LLM mode: %s", llm.mode)
        except Exception as exc:
            logger.error("LLM warm-up failed: %s", exc)

        # Sync knowledge base entries to ChromaDB
        try:
            from app.database import AsyncSessionLocal
            from app.knowledge import KnowledgeEntry
            from sqlalchemy import text as sql_text
            async with AsyncSessionLocal() as session:
                # Create knowledge_entries table if needed
                ks = get_knowledge_service()
                synced = await ks.sync_all_to_chroma(session)
                logger.info("Knowledge base: %d entries synced to ChromaDB.", synced)
        except Exception as exc:
            logger.warning("Knowledge base sync skipped: %s", exc)

    except MemoryError:
        logger.error("Insufficient memory for ML model loading. Running in API-only mode.")
    except Exception as exc:
        logger.error("Startup error: %s", exc)

    logger.info("%s is ready to serve requests.", settings.app_name)
    yield

    logger.info("Shutting down %s...", settings.app_name)


# ─── App Instance ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-ready AI chatbot with ML intent classification and RAG "
        "for Curify AI Advisor / Customer Support."
    ),
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# ─── CORS Middleware ───────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files (Chat UI + Admin Panel) ────────────────────────────────────
import os
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ─── Request Logging Middleware ───────────────────────────────────────────────

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Log every incoming request with method, path, status, and duration.
    Assigns a unique request ID for traceability.
    """
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    logger.info(
        "[%s] --> %s %s", request_id, request.method, request.url.path
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] Unhandled error in middleware", request_id)
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "[%s] <-- %s %s | status=%d | %dms",
        request_id, request.method, request.url.path,
        response.status_code, elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    return response


# ─── Global Exception Handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch all unhandled exceptions and return clean JSON errors.
    Never leaks raw stack traces in production.
    """
    logger.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method, request.url.path, exc,
        traceback.format_exc() if settings.debug else "",
    )
    error_detail = str(exc) if settings.debug else "An internal error occurred."
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": error_detail,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return HTTPExceptions as clean JSON (not HTML)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code,
        },
    )


# ─── Pydantic Request/Response Models ────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Stable client-supplied user identifier.",
        examples=["user_abc123"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's chat message (supports Tanglish).",
        examples=["Hello! What does Curify offer?"],
    )

    @field_validator("user_id", "message", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace from string fields."""
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("user_id")
    @classmethod
    def no_sql_injection(cls, v: str) -> str:
        """Reject obvious SQL injection patterns in user_id."""
        dangerous = ["'", '"', ";", "--", "/*", "*/", "xp_", "drop ", "select "]
        lower = v.lower()
        for pattern in dangerous:
            if pattern in lower:
                raise ValueError("Invalid characters in user_id.")
        return v


class SourceItem(BaseModel):
    """A single retrieved knowledge source in the chat response."""
    text: str
    intent: str
    similarity: float
    doc_id: str


class ChatResponseModel(BaseModel):
    """Response body for POST /chat."""
    response: str
    intent: str
    confidence: float
    fallback_used: bool
    retrieval_used: bool
    sources: list[SourceItem]
    latency_ms: int
    conversation_id: str


class TrainRequest(BaseModel):
    """Request body for POST /train."""
    reset_chroma: bool = Field(
        default=False,
        description="If true, clears ChromaDB before reindexing.",
    )
    csv_path: str | None = Field(
        default=None,
        description="Optional override CSV path. Defaults to config value.",
    )


class HistoryMessage(BaseModel):
    """Single message in the conversation history."""
    id: int
    user_text: str
    bot_response: str
    intent: str | None
    confidence: float | None
    fallback_used: bool
    latency_ms: int | None
    created_at: str


# ─── Dependency: API Key Check ────────────────────────────────────────────────

async def verify_train_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
) -> None:
    """
    FastAPI dependency that validates the X-API-Key header for protected routes.

    Raises:
        HTTPException 401: If the key is missing.
        HTTPException 403: If the key does not match.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required.",
        )
    if x_api_key != settings.train_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )


async def verify_chatbot_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
) -> None:
    """
    FastAPI dependency that validates the X-API-Key header for chatbot & knowledge routes.

    Raises:
        HTTPException 401: If the key is missing.
        HTTPException 403: If the key does not match.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required.",
        )
    if x_api_key != settings.chatbot_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns system status, classifier state, and ChromaDB document count.
    """
    try:
        from app.classifier import get_classifier
        from app.embeddings import get_chroma_service
        clf = get_classifier()
        chroma = get_chroma_service()
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "classifier_trained": clf.is_trained,
            "known_intents": clf.classes,
            "chroma_docs": chroma.count(),
            "gemini_configured": bool(settings.gemini_api_key),
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(exc)},
        )


@app.post("/chat", response_model=ChatResponseModel, tags=["Chat"], dependencies=[Depends(verify_chatbot_api_key)])
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponseModel:
    """
    Main chat endpoint — runs the full RAG pipeline.

    - Classifies intent using TF-IDF + Logistic Regression
    - Retrieves top-3 relevant documents from ChromaDB
    - Generates a context-aware response via Gemini 2.5 Flash
    - Logs the conversation turn to SQLite
    - Returns structured JSON with intent, confidence, and source snippets

    Supports Tamil-English (Tanglish) mixed messages.
    """
    from app.chatbot import get_chatbot_pipeline
    pipeline = get_chatbot_pipeline()
    try:
        chat_response = await pipeline.process(
            user_id=request.user_id,
            message=request.message,
            db=db,
        )
    except Exception as exc:
        logger.error("Chat pipeline error for user=%r: %s", request.user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat processing failed. Please try again.",
        )

    return ChatResponseModel(
        response=chat_response.response,
        intent=chat_response.intent,
        confidence=round(chat_response.confidence, 4),
        fallback_used=chat_response.fallback_used,
        retrieval_used=chat_response.retrieval_used,
        sources=[
            SourceItem(
                text=s.text,
                intent=s.intent,
                similarity=s.similarity,
                doc_id=s.doc_id,
            )
            for s in chat_response.sources
        ],
        latency_ms=chat_response.latency_ms,
        conversation_id=chat_response.conversation_id,
    )


@app.post("/train", tags=["Admin"], dependencies=[Depends(verify_train_api_key)])
async def trigger_training(request: TrainRequest) -> dict:
    """
    Trigger model retraining (API-key protected).

    Runs `models/train.py` as a subprocess with the configured CSV path.
    Returns immediately with a job ID; training runs in the background.

    Requires header: X-API-Key: <TRAIN_API_KEY>
    """
    job_id = str(uuid.uuid4())[:8]
    csv_path = request.csv_path or settings.training_data_path

    async def _run_training() -> None:
        """Run the training script in a subprocess."""
        cmd = [
            sys.executable,
            "models/train.py",
            "--csv", csv_path,
        ]
        if request.reset_chroma:
            cmd.append("--reset-chroma")

        try:
            logger.info("[job=%s] Starting training: %s", job_id, " ".join(cmd))
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()
            if result.returncode == 0:
                logger.info("[job=%s] Training completed successfully.", job_id)
                logger.debug("[job=%s] stdout: %s", job_id, stdout.decode())
            else:
                logger.error(
                    "[job=%s] Training failed (rc=%d): %s",
                    job_id, result.returncode, stderr.decode()
                )
        except Exception as exc:
            logger.error("[job=%s] Training subprocess error: %s", job_id, exc)

    asyncio.create_task(_run_training())

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": f"Training job {job_id} started in the background.",
        "csv_path": csv_path,
        "reset_chroma": request.reset_chroma,
    }


@app.get("/history/{user_id}", tags=["Chat"])
async def get_history(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Retrieve paginated conversation history for a user.

    Returns the most recent ``limit`` message turns for the given user_id,
    ordered by timestamp descending.
    """
    # Validate user_id against injection
    dangerous = ["'", '"', ";", "--", "drop ", "select "]
    lower_uid = user_id.lower()
    for pattern in dangerous:
        if pattern in lower_uid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user_id format.",
            )

    try:
        # Fetch conversations for this user
        conv_result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
        )
        conversations = conv_result.scalars().all()

        if not conversations:
            return {"user_id": user_id, "messages": [], "total": 0}

        conv_ids = [c.id for c in conversations]

        # Fetch most recent messages across all conversations
        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id.in_(conv_ids))
            .order_by(desc(Message.created_at))
            .limit(limit)
        )
        messages = msg_result.scalars().all()

        return {
            "user_id": user_id,
            "total": len(messages),
            "messages": [
                HistoryMessage(
                    id=m.id,
                    user_text=m.user_text,
                    bot_response=m.bot_response,
                    intent=m.intent,
                    confidence=m.confidence,
                    fallback_used=m.fallback_used,
                    latency_ms=m.latency_ms,
                    created_at=m.created_at.isoformat(),
                ).model_dump()
                for m in messages
            ],
        }
    except Exception as exc:
        logger.error("History fetch failed for user=%r: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history.",
        )


# ─── Root / UI Redirects ─────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """Serve the customer chat UI, or return API info if no UI is present."""
    import os
    ui_path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {
        "message": f"Welcome to {settings.app_name} API",
        "version": settings.app_version,
        "chat_ui": "/",
        "admin_ui": "/static/admin.html",
        "docs": "/docs",
        "health": "/health",
        "mode": "/mode",
    }


@app.get("/admin", tags=["System"])
async def admin_ui():
    """Serve the admin knowledge management panel."""
    import os
    admin_path = os.path.join(os.path.dirname(__file__), "..", "static", "admin.html")
    if os.path.exists(admin_path):
        return FileResponse(admin_path)
    raise HTTPException(status_code=404, detail="Admin UI not found.")


@app.get("/mode", tags=["System"])
async def get_mode() -> dict:
    """
    Show the current response generation mode.

    Returns 'offline' if no Gemini API key is configured,
    or 'gemini' if using the Gemini 2.5 Flash API.
    """
    from app.llm_client import get_llm_client
    llm = get_llm_client()
    return {
        "mode": llm.mode,
        "gemini_configured": llm.is_configured,
        "description": (
            "Using Gemini 2.5 Flash API for responses."
            if llm.is_configured
            else "Running 100% offline — responses use ChromaDB retrieval + smart templates."
        ),
    }


# ─── Knowledge Base Routes ─────────────────────────────────────────────────────

class KnowledgeCreateRequest(BaseModel):
    """Request body for creating a knowledge entry."""
    question: str = Field(..., min_length=3, max_length=1000,
                          description="The question or topic text.")
    answer: str = Field(..., min_length=3, max_length=5000,
                        description="The answer/response for this topic.")
    intent: str = Field(default="product_inquiry",
                        description="Intent label for this entry.")
    category: str | None = Field(default=None, max_length=128,
                                 description="Optional category (e.g. 'pricing').")


class KnowledgeUpdateRequest(BaseModel):
    """Request body for updating a knowledge entry (all fields optional)."""
    question: str | None = Field(default=None, max_length=1000)
    answer: str | None = Field(default=None, max_length=5000)
    intent: str | None = Field(default=None)
    category: str | None = Field(default=None)
    is_active: bool | None = Field(default=None)


def _entry_to_dict(e) -> dict:
    """Serialise a KnowledgeEntry to a plain dict."""
    return {
        "id": e.id,
        "question": e.question,
        "answer": e.answer,
        "intent": e.intent,
        "category": e.category,
        "is_active": e.is_active,
        "created_at": e.created_at.isoformat(),
        "updated_at": e.updated_at.isoformat(),
    }


@app.get("/knowledge", tags=["Knowledge Base"], dependencies=[Depends(verify_chatbot_api_key)])
async def list_knowledge(
    category: str | None = Query(default=None),
    intent: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List all knowledge base entries.

    Supports filtering by category and intent. Used by the admin panel
    to display all custom FAQs and product details.
    """
    from app.knowledge import get_knowledge_service
    ks = get_knowledge_service()
    entries = await ks.list_entries(
        db, category=category, intent=intent,
        limit=limit, offset=offset
    )
    categories = await ks.get_categories(db)
    return {
        "total": len(entries),
        "categories": categories,
        "entries": [_entry_to_dict(e) for e in entries],
    }


@app.post("/knowledge", tags=["Knowledge Base"], status_code=201, dependencies=[Depends(verify_chatbot_api_key)])
async def create_knowledge(
    request: KnowledgeCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Add a new knowledge base entry.

    The entry is immediately indexed in ChromaDB so the chatbot
    can use it for responses — no retraining required.
    """
    from app.knowledge import get_knowledge_service
    ks = get_knowledge_service()
    try:
        entry = await ks.add_entry(
            db=db,
            question=request.question,
            answer=request.answer,
            intent=request.intent,
            category=request.category,
        )
        return {"success": True, "entry": _entry_to_dict(entry)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to create knowledge entry: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create entry.")


@app.put("/knowledge/{entry_id}", tags=["Knowledge Base"], dependencies=[Depends(verify_chatbot_api_key)])
async def update_knowledge(
    entry_id: str,
    request: KnowledgeUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Update an existing knowledge entry.

    Changes are reflected immediately in ChromaDB.
    """
    from app.knowledge import get_knowledge_service
    ks = get_knowledge_service()
    entry = await ks.update_entry(
        db=db,
        entry_id=entry_id,
        question=request.question,
        answer=request.answer,
        intent=request.intent,
        category=request.category,
        is_active=request.is_active,
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found.")
    return {"success": True, "entry": _entry_to_dict(entry)}


@app.delete("/knowledge/{entry_id}", tags=["Knowledge Base"], dependencies=[Depends(verify_chatbot_api_key)])
async def delete_knowledge(
    entry_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Delete a knowledge entry from both SQLite and ChromaDB.
    """
    from app.knowledge import get_knowledge_service
    ks = get_knowledge_service()
    deleted = await ks.delete_entry(db, entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Knowledge entry not found.")
    return {"success": True, "message": f"Entry {entry_id} deleted."}


@app.post("/knowledge/sync", tags=["Knowledge Base"], dependencies=[Depends(verify_chatbot_api_key)])
async def sync_knowledge(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Re-sync all knowledge entries to ChromaDB.

    Use this if ChromaDB was reset or if you suspect index is out of sync.
    """
    from app.knowledge import get_knowledge_service
    ks = get_knowledge_service()
    count = await ks.sync_all_to_chroma(db)
    return {"success": True, "synced": count}
