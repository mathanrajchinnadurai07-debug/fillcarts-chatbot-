"""
app/database.py
───────────────
Async SQLAlchemy ORM models and session management for:
  - users           : registered chat users / sessions
  - conversations   : logical grouping of messages per user
  - messages        : individual chat turns with metadata
  - feedback        : thumbs-up/down ratings per message (used for retraining)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings


# ─── Engine & Session Factory ─────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False},  # SQLite only
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ─── Base & Helper ────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Return the current UTC timestamp (timezone-aware)."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ─── Models ───────────────────────────────────────────────────────────────────

class User(Base):
    """
    Represents a chat user identified by a client-supplied user_id string.

    A user_id can be any stable identifier from the calling application
    (e.g. Firebase UID, Flutter device ID, UUID).
    """

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("user_id", name="uq_user_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User user_id={self.user_id!r}>"


class Conversation(Base):
    """
    A logical session grouping multiple messages for a user.

    A new conversation is created when the user starts a fresh chat session.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    user_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at"
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id!r} user={self.user_id!r}>"


class Message(Base):
    """
    A single chat turn (user message + bot response) with ML metadata.

    Stores both the raw user text and the bot's generated response, plus
    the classified intent and confidence score for analytics and retraining.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_text: Mapped[str] = mapped_column(Text, nullable=False)
    bot_response: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    retrieval_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
    feedback: Mapped[list["MessageFeedback"]] = relationship(
        "MessageFeedback", back_populates="message", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} intent={self.intent!r} conf={self.confidence}>"


class MessageFeedback(Base):
    """
    User feedback (thumbs-up / thumbs-down) on a bot message.

    This table is used by the retraining script to identify low-quality
    responses that need more training data.
    """

    __tablename__ = "message_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 = good, -1 = bad
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="feedback")

    def __repr__(self) -> str:
        return f"<MessageFeedback msg={self.message_id} rating={self.rating}>"


# ─── Database Lifecycle ───────────────────────────────────────────────────────

async def create_db_tables() -> None:
    """
    Create all tables defined in this module.

    Called on application startup. Safe to call multiple times (CREATE IF NOT EXISTS).
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async database session.

    Usage::

        @app.post("/chat")
        async def chat(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
