"""
app/knowledge.py
────────────────
Knowledge base management — Add, edit, delete, and list your own
custom FAQ entries. Each entry is stored in both:
  - SQLite (for persistence and admin management)
  - ChromaDB (for semantic retrieval during chat)

This allows you to input your own product details, FAQs, policies,
and any other information WITHOUT needing an AI API key.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import Boolean, DateTime, Integer, String, Text, select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base, AsyncSessionLocal, _utcnow
from app.embeddings import get_chroma_service

logger = logging.getLogger(__name__)


# ─── Knowledge Entry ORM Model ────────────────────────────────────────────────

class KnowledgeEntry(Base):
    """
    A single knowledge base entry (question + answer + intent tag).

    These are ingested into ChromaDB for semantic retrieval and stored
    in SQLite for admin management (edit, delete, list).
    """

    __tablename__ = "knowledge_entries"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(64), nullable=False, default="product_inquiry")
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<KnowledgeEntry id={self.id[:8]} intent={self.intent!r}>"


# ─── Knowledge Service ────────────────────────────────────────────────────────

class KnowledgeService:
    """
    Service for managing knowledge base entries.

    Keeps ChromaDB and SQLite in sync so the chatbot always has
    up-to-date information without needing to retrain the ML model.
    """

    CHROMA_PREFIX = "kb_"  # Prefix for knowledge base doc IDs in ChromaDB

    def __init__(self) -> None:
        self._chroma = None

    def _get_chroma(self):
        if self._chroma is None:
            self._chroma = get_chroma_service()
        return self._chroma

    # ── Create ────────────────────────────────────────────────────────────────

    async def add_entry(
        self,
        db: AsyncSession,
        question: str,
        answer: str,
        intent: str = "product_inquiry",
        category: str | None = None,
    ) -> KnowledgeEntry:
        """
        Add a new knowledge entry to SQLite and index it in ChromaDB.

        Args:
            db: Active async database session.
            question: The question or topic text.
            answer: The answer/response to return when this topic is matched.
            intent: Intent label for classification context.
            category: Optional category label (e.g. 'pricing', 'products').

        Returns:
            The created KnowledgeEntry ORM instance.

        Raises:
            ValueError: If question or answer is empty.
        """
        if not question.strip() or not answer.strip():
            raise ValueError("Question and answer cannot be empty.")

        entry = KnowledgeEntry(
            question=question.strip(),
            answer=answer.strip(),
            intent=intent.strip(),
            category=category,
        )
        db.add(entry)
        await db.flush()

        # Index in ChromaDB immediately
        try:
            chroma_id = f"{self.CHROMA_PREFIX}{entry.id}"
            self._get_chroma().add_documents(
                ids=[chroma_id],
                documents=[entry.question],
                metadatas=[{
                    "intent": entry.intent,
                    "response": entry.answer,
                    "category": entry.category or "",
                    "source": "knowledge_base",
                    "entry_id": entry.id,
                }],
            )
            logger.info("Knowledge entry %s indexed in ChromaDB.", entry.id[:8])
        except Exception as exc:
            logger.error("Failed to index knowledge entry in ChromaDB: %s", exc)
            # Don't fail the DB write — ChromaDB can be re-indexed later

        return entry

    # ── Read ──────────────────────────────────────────────────────────────────

    async def list_entries(
        self,
        db: AsyncSession,
        category: str | None = None,
        intent: str | None = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KnowledgeEntry]:
        """
        List knowledge entries with optional filters.

        Args:
            db: Active async database session.
            category: Filter by category (optional).
            intent: Filter by intent (optional).
            active_only: If True, only return active entries.
            limit: Max number of results.
            offset: Pagination offset.

        Returns:
            List of KnowledgeEntry instances.
        """
        query = select(KnowledgeEntry).order_by(desc(KnowledgeEntry.created_at))

        if active_only:
            query = query.where(KnowledgeEntry.is_active == True)
        if category:
            query = query.where(KnowledgeEntry.category == category)
        if intent:
            query = query.where(KnowledgeEntry.intent == intent)

        query = query.limit(limit).offset(offset)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_entry(self, db: AsyncSession, entry_id: str) -> KnowledgeEntry | None:
        """
        Fetch a single knowledge entry by ID.

        Args:
            db: Active async database session.
            entry_id: UUID of the entry.

        Returns:
            KnowledgeEntry if found, None otherwise.
        """
        result = await db.execute(
            select(KnowledgeEntry).where(KnowledgeEntry.id == entry_id)
        )
        return result.scalar_one_or_none()

    # ── Update ────────────────────────────────────────────────────────────────

    async def update_entry(
        self,
        db: AsyncSession,
        entry_id: str,
        question: str | None = None,
        answer: str | None = None,
        intent: str | None = None,
        category: str | None = None,
        is_active: bool | None = None,
    ) -> KnowledgeEntry | None:
        """
        Update a knowledge entry and re-index in ChromaDB.

        Args:
            db: Active async database session.
            entry_id: UUID of the entry to update.
            question: New question text (optional).
            answer: New answer text (optional).
            intent: New intent label (optional).
            category: New category (optional).
            is_active: Toggle active status (optional).

        Returns:
            Updated KnowledgeEntry, or None if not found.
        """
        entry = await self.get_entry(db, entry_id)
        if not entry:
            return None

        if question is not None:
            entry.question = question.strip()
        if answer is not None:
            entry.answer = answer.strip()
        if intent is not None:
            entry.intent = intent.strip()
        if category is not None:
            entry.category = category
        if is_active is not None:
            entry.is_active = is_active

        await db.flush()

        # Re-index in ChromaDB
        try:
            chroma_id = f"{self.CHROMA_PREFIX}{entry.id}"
            if entry.is_active:
                self._get_chroma().add_documents(
                    ids=[chroma_id],
                    documents=[entry.question],
                    metadatas=[{
                        "intent": entry.intent,
                        "response": entry.answer,
                        "category": entry.category or "",
                        "source": "knowledge_base",
                        "entry_id": entry.id,
                    }],
                )
            else:
                # Deactivated — remove from ChromaDB
                try:
                    self._get_chroma().collection.delete(ids=[chroma_id])
                except Exception:
                    pass
        except Exception as exc:
            logger.error("Failed to re-index updated entry in ChromaDB: %s", exc)

        return entry

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_entry(self, db: AsyncSession, entry_id: str) -> bool:
        """
        Delete a knowledge entry from SQLite and ChromaDB.

        Args:
            db: Active async database session.
            entry_id: UUID of the entry to delete.

        Returns:
            True if deleted, False if not found.
        """
        entry = await self.get_entry(db, entry_id)
        if not entry:
            return False

        # Remove from ChromaDB first
        try:
            chroma_id = f"{self.CHROMA_PREFIX}{entry.id}"
            self._get_chroma().collection.delete(ids=[chroma_id])
        except Exception as exc:
            logger.warning("ChromaDB delete failed for %s: %s", entry_id, exc)

        await db.delete(entry)
        return True

    # ── Bulk Sync ─────────────────────────────────────────────────────────────

    async def sync_all_to_chroma(self, db: AsyncSession) -> int:
        """
        Re-index all active knowledge entries into ChromaDB.

        Useful after ChromaDB is reset or on first startup.

        Args:
            db: Active async database session.

        Returns:
            Number of entries synced.
        """
        entries = await self.list_entries(db, active_only=True, limit=10000)
        if not entries:
            logger.info("No knowledge entries to sync.")
            return 0

        ids = [f"{self.CHROMA_PREFIX}{e.id}" for e in entries]
        documents = [e.question for e in entries]
        metadatas = [
            {
                "intent": e.intent,
                "response": e.answer,
                "category": e.category or "",
                "source": "knowledge_base",
                "entry_id": e.id,
            }
            for e in entries
        ]

        self._get_chroma().add_documents(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Synced %d knowledge entries to ChromaDB.", len(entries))
        return len(entries)

    async def get_categories(self, db: AsyncSession) -> list[str]:
        """Return a list of all unique categories in the knowledge base."""
        result = await db.execute(
            select(KnowledgeEntry.category)
            .where(KnowledgeEntry.is_active == True)
            .distinct()
        )
        return [r for r in result.scalars().all() if r]


# ─── Module-level singleton ───────────────────────────────────────────────────

_knowledge_service: KnowledgeService | None = None


def get_knowledge_service() -> KnowledgeService:
    """Return the module-level KnowledgeService singleton."""
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service
