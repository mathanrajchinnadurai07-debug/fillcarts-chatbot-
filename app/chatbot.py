"""
app/chatbot.py
──────────────
Full RAG (Retrieval-Augmented Generation) pipeline orchestrator.

Pipeline for each user message:
  1. Sanitise and validate input
  2. Classify intent (TF-IDF + LR, with embedding fallback)
  3. Retrieve top-K semantically similar documents from ChromaDB
  4. Build a context-aware, intent-aware prompt
  5. Call Gemini 2.5 Flash (with retry + fallback)
  6. Log the full conversation turn to SQLite
  7. Return a structured ChatResponse with sources

Designed for async FastAPI routes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.classifier import IntentResult, get_classifier
from app.config import settings
from app.database import Conversation, Message, User
from app.embeddings import get_chroma_service
from app.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# ─── System Prompt Template ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Curify AI Advisor — a helpful, professional, and \
empathetic customer support assistant for Curify, an AI-powered advisory platform.

Your role:
- Provide accurate, concise answers about Curify products, pricing, orders, \
accounts, and AI insights.
- Be warm but professional. Use first-person ("I") naturally.
- Support Tamil-English mixed (Tanglish) messages gracefully — respond in the \
language the user used.
- Keep responses under 150 words unless the user explicitly asks for detailed info.
- Never fabricate pricing figures, order IDs, or policies not in context.
- If unsure, acknowledge and offer to escalate to human support.

Detected intent: {intent}
Confidence: {confidence:.0%}

Relevant knowledge context:
{context}

User message: {user_message}

Respond naturally and helpfully based on the context above."""


# ─── Response Dataclass ───────────────────────────────────────────────────────

@dataclass
class SourceSnippet:
    """A single retrieved document snippet returned in the API response."""
    text: str
    intent: str
    similarity: float
    doc_id: str


@dataclass
class ChatResponse:
    """Structured response from the RAG pipeline."""
    response: str
    intent: str
    confidence: float
    fallback_used: bool
    retrieval_used: bool
    sources: list[SourceSnippet] = field(default_factory=list)
    latency_ms: int = 0
    conversation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON API responses."""
        return {
            "response": self.response,
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "fallback_used": self.fallback_used,
            "retrieval_used": self.retrieval_used,
            "sources": [
                {
                    "text": s.text,
                    "intent": s.intent,
                    "similarity": s.similarity,
                    "doc_id": s.doc_id,
                }
                for s in self.sources
            ],
            "latency_ms": self.latency_ms,
            "conversation_id": self.conversation_id,
        }


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class ChatbotPipeline:
    """
    Orchestrates the full RAG pipeline for a single chat turn.

    Designed to be instantiated once at app startup and reused across
    all requests (thread-safe, stateless per request).
    """

    MAX_INPUT_LENGTH: int = 1000  # characters
    MAX_CONTEXT_DOCS: int = settings.top_k_retrieval

    def __init__(self) -> None:
        """Lazily initialise all downstream services."""
        self._classifier = None
        self._chroma = None
        self._llm = None

    def _get_classifier(self):
        if self._classifier is None:
            self._classifier = get_classifier()
        return self._classifier

    def _get_chroma(self):
        if self._chroma is None:
            self._chroma = get_chroma_service()
        return self._chroma

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    # ── Input Sanitisation ────────────────────────────────────────────────────

    def _sanitise_input(self, text: str) -> str:
        """
        Sanitise raw user input for safe processing.

        - Strip leading/trailing whitespace
        - Truncate to MAX_INPUT_LENGTH characters
        - Remove null bytes (SQL injection / injection attempt mitigation)
        - Collapse excessive whitespace

        Args:
            text: Raw user message.

        Returns:
            Sanitised string, or empty string if input is blank.
        """
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = cleaned.replace("\x00", "")  # remove null bytes
        cleaned = " ".join(cleaned.split())     # collapse whitespace
        if len(cleaned) > self.MAX_INPUT_LENGTH:
            logger.warning(
                "Input truncated from %d to %d characters.", len(cleaned), self.MAX_INPUT_LENGTH
            )
            cleaned = cleaned[: self.MAX_INPUT_LENGTH]
        return cleaned

    # ── Context Building ──────────────────────────────────────────────────────

    def _build_context(self, hits: list[dict]) -> str:
        """
        Format ChromaDB retrieval hits into a context block for the LLM.

        Args:
            hits: List of dicts from ChromaService.query().

        Returns:
            Multi-line context string.
        """
        if not hits:
            return "No additional context available."

        lines: list[str] = []
        for i, hit in enumerate(hits, start=1):
            meta = hit.get("metadata", {})
            snippet = hit.get("document", "")
            intent = meta.get("intent", "unknown")
            similarity = hit.get("similarity", 0.0)
            lines.append(
                f"[{i}] (intent: {intent}, similarity: {similarity:.2f})\n"
                f"  Q: {snippet}\n"
                f"  A: {meta.get('response', '')}"
            )
        return "\n\n".join(lines)

    def _build_prompt(
        self,
        user_message: str,
        intent_result: IntentResult,
        context: str,
    ) -> str:
        """
        Construct the full prompt for the Gemini LLM.

        Args:
            user_message: Sanitised user input.
            intent_result: Classified intent with confidence.
            context: Formatted context from ChromaDB retrieval.

        Returns:
            Complete prompt string.
        """
        return SYSTEM_PROMPT.format(
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            context=context,
            user_message=user_message,
        )

    # ── Database Logging ──────────────────────────────────────────────────────

    async def _ensure_user(self, db: AsyncSession, user_id: str) -> User:
        """
        Get or create a User record for the given user_id.

        Args:
            db: Active async database session.
            user_id: Client-supplied user identifier.

        Returns:
            User ORM instance.
        """
        result = await db.execute(
            select(User).where(User.user_id == user_id)
        )
        user = result.scalar_one_or_none()
        if user is None:
            user = User(user_id=user_id)
            db.add(user)
            await db.flush()  # assign PK without committing
        return user

    async def _get_or_create_conversation(
        self, db: AsyncSession, user_id: str
    ) -> Conversation:
        """
        Get the most recent conversation or create a new one.

        Args:
            db: Active async database session.
            user_id: Client-supplied user identifier.

        Returns:
            Conversation ORM instance.
        """
        result = await db.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(1)
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            conversation = Conversation(user_id=user_id)
            db.add(conversation)
            await db.flush()
        return conversation

    async def _log_message(
        self,
        db: AsyncSession,
        conversation_id: str,
        user_text: str,
        chat_response: ChatResponse,
    ) -> None:
        """
        Persist a chat turn (user message + bot response) to SQLite.

        Args:
            db: Active async database session.
            conversation_id: UUID of the parent conversation.
            user_text: The sanitised user message.
            chat_response: The fully-built ChatResponse object.
        """
        message = Message(
            conversation_id=conversation_id,
            user_text=user_text,
            bot_response=chat_response.response,
            intent=chat_response.intent,
            confidence=chat_response.confidence,
            fallback_used=chat_response.fallback_used,
            retrieval_used=chat_response.retrieval_used,
            latency_ms=chat_response.latency_ms,
        )
        db.add(message)
        # Session commits are handled by the get_db() dependency

    # ── Main Pipeline ─────────────────────────────────────────────────────────

    async def process(
        self,
        user_id: str,
        message: str,
        db: AsyncSession,
    ) -> ChatResponse:
        """
        Execute the full RAG pipeline for a single user message.

        Args:
            user_id: Stable client-supplied user identifier.
            message: Raw user message text (may be Tanglish / mixed).
            db: Active async database session (injected by FastAPI dependency).

        Returns:
            ChatResponse with response text, intent, confidence, and sources.
        """
        pipeline_start = time.perf_counter()

        # ── Step 1: Sanitise ──────────────────────────────────────────────────
        clean_text = self._sanitise_input(message)
        if not clean_text:
            return ChatResponse(
                response="I didn't receive any message. Please type your question!",
                intent="general_greeting",
                confidence=0.0,
                fallback_used=True,
                retrieval_used=False,
                latency_ms=0,
            )

        # ── Step 2: Classify intent ───────────────────────────────────────────
        try:
            intent_result: IntentResult = self._get_classifier().predict(clean_text)
            logger.info(
                "Intent classified: %s (conf=%.3f, method=%s)",
                intent_result.intent, intent_result.confidence, intent_result.method
            )
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            intent_result = IntentResult(
                intent="general_greeting",
                confidence=0.0,
                fallback_used=True,
                method="default",
            )

        # ── Step 3: ChromaDB retrieval ────────────────────────────────────────
        hits: list[dict] = []
        retrieval_used = False
        try:
            hits = self._get_chroma().query(
                query_text=clean_text,
                top_k=self.MAX_CONTEXT_DOCS,
            )
            retrieval_used = bool(hits)
            logger.info("ChromaDB retrieved %d documents.", len(hits))
        except Exception as exc:
            logger.error("ChromaDB retrieval failed: %s", exc)

        sources = [
            SourceSnippet(
                text=h["document"],
                intent=h["metadata"].get("intent", ""),
                similarity=h.get("similarity", 0.0),
                doc_id=h["id"],
            )
            for h in hits
        ]

        # ── Step 4: Build prompt ──────────────────────────────────────────────
        context = self._build_context(hits)
        prompt = self._build_prompt(clean_text, intent_result, context)

        # ── Step 5: Generate response ─────────────────────────────────────────
        try:
            response_text, used_fallback = self._get_llm().generate(
                prompt=prompt,
                intent=intent_result.intent,
                use_cache=True,
                user_message=clean_text,
                retrieved_hits=hits,   # passed for offline mode
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            from app.llm_client import get_fallback
            response_text = get_fallback(intent_result.intent)
            used_fallback = True

        latency_ms = int((time.perf_counter() - pipeline_start) * 1000)

        chat_response = ChatResponse(
            response=response_text,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            fallback_used=used_fallback or intent_result.fallback_used,
            retrieval_used=retrieval_used,
            sources=sources,
            latency_ms=latency_ms,
        )

        # ── Step 6: Log to database ───────────────────────────────────────────
        try:
            await self._ensure_user(db, user_id)
            conversation = await self._get_or_create_conversation(db, user_id)
            chat_response.conversation_id = conversation.id
            await self._log_message(db, conversation.id, clean_text, chat_response)
        except Exception as exc:
            logger.error("Failed to log conversation to DB: %s", exc)

        logger.info(
            "Chat pipeline complete: user=%r intent=%r latency=%dms",
            user_id, chat_response.intent, latency_ms,
        )
        return chat_response


# ─── Module-level singleton ───────────────────────────────────────────────────

_pipeline: ChatbotPipeline | None = None


def get_chatbot_pipeline() -> ChatbotPipeline:
    """Return the module-level ChatbotPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ChatbotPipeline()
    return _pipeline
