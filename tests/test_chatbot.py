"""
tests/test_chatbot.py
─────────────────────
Unit tests for the RAG ChatbotPipeline.

Tests:
  - Input sanitisation edge cases
  - Prompt building structure
  - Context formatting from ChromaDB hits
  - Full pipeline returns a ChatResponse with expected fields
  - Empty message is handled gracefully
  - Tanglish input passes through without errors
  - SQL injection in message field is sanitised
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chatbot import ChatbotPipeline, ChatResponse, SourceSnippet
from app.classifier import IntentClassifier, IntentResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline() -> ChatbotPipeline:
    """Return a fresh ChatbotPipeline instance."""
    return ChatbotPipeline()


@pytest.fixture
def trained_classifier() -> IntentClassifier:
    """Return a minimal trained classifier for testing."""
    data = pd.DataFrame({
        "text": [
            "Hello hi", "Good morning", "Bye goodbye",
            "What is price?", "Cost please", "I want refund",
        ],
        "intent": [
            "general_greeting", "general_greeting", "general_farewell",
            "pricing", "pricing", "refund_request",
        ],
    })
    clf = IntentClassifier(confidence_threshold=0.3)
    clf.train(data)
    return clf


@pytest.fixture
def mock_db():
    """Return a mock async database session."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


# ─── Sanitisation Tests ────────────────────────────────────────────────────────

class TestInputSanitisation:
    """Tests for the _sanitise_input method."""

    def test_normal_text_unchanged(self, pipeline):
        """Normal text should be returned as-is (stripped)."""
        result = pipeline._sanitise_input("  Hello world!  ")
        assert result == "Hello world!"

    def test_empty_string_returns_empty(self, pipeline):
        """Empty string should return empty string."""
        assert pipeline._sanitise_input("") == ""

    def test_none_returns_empty(self, pipeline):
        """None input should return empty string."""
        assert pipeline._sanitise_input(None) == ""

    def test_long_input_truncated(self, pipeline):
        """Input over MAX_INPUT_LENGTH should be truncated."""
        long_text = "A" * 2000
        result = pipeline._sanitise_input(long_text)
        assert len(result) <= pipeline.MAX_INPUT_LENGTH

    def test_null_bytes_removed(self, pipeline):
        """Null bytes should be stripped from input."""
        text_with_nulls = "Hello\x00World\x00"
        result = pipeline._sanitise_input(text_with_nulls)
        assert "\x00" not in result

    def test_excessive_whitespace_collapsed(self, pipeline):
        """Multiple consecutive spaces should be collapsed."""
        result = pipeline._sanitise_input("Hello   World   test")
        assert "  " not in result

    def test_sql_injection_not_executed(self, pipeline):
        """SQL injection attempts should be safely sanitised, not executed."""
        injection = "'; DROP TABLE messages; -- hello"
        result = pipeline._sanitise_input(injection)
        # Should return a string, not raise
        assert isinstance(result, str)

    def test_tanglish_preserved(self, pipeline):
        """Tamil-English mixed text should pass through intact."""
        tanglish = "Vanakkam! Naan Curify pathi kelvi kekkiren."
        result = pipeline._sanitise_input(tanglish)
        assert "Vanakkam" in result


# ─── Context Building Tests ───────────────────────────────────────────────────

class TestContextBuilding:
    """Tests for the _build_context method."""

    def test_empty_hits_returns_placeholder(self, pipeline):
        """Empty hits list should return a placeholder string."""
        result = pipeline._build_context([])
        assert "No additional context" in result

    def test_single_hit_formatted_correctly(self, pipeline):
        """Single hit should produce formatted context with numbering."""
        hits = [
            {
                "id": "doc_0",
                "document": "What is Curify?",
                "metadata": {"intent": "product_inquiry", "response": "Curify is an AI platform."},
                "distance": 0.1,
                "similarity": 0.9,
            }
        ]
        context = pipeline._build_context(hits)
        assert "[1]" in context
        assert "product_inquiry" in context
        assert "What is Curify?" in context

    def test_multiple_hits_all_numbered(self, pipeline):
        """Multiple hits should each be numbered sequentially."""
        hits = [
            {
                "id": f"doc_{i}",
                "document": f"Text {i}",
                "metadata": {"intent": "pricing", "response": f"Response {i}"},
                "distance": 0.1 * i,
                "similarity": 1.0 - 0.1 * i,
            }
            for i in range(3)
        ]
        context = pipeline._build_context(hits)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context


# ─── Prompt Building Tests ────────────────────────────────────────────────────

class TestPromptBuilding:
    """Tests for the _build_prompt method."""

    def test_prompt_contains_user_message(self, pipeline):
        """Built prompt should include the user's message."""
        intent_result = IntentResult(intent="pricing", confidence=0.85)
        prompt = pipeline._build_prompt(
            "What is the price?", intent_result, "Some context here."
        )
        assert "What is the price?" in prompt

    def test_prompt_contains_intent(self, pipeline):
        """Built prompt should include the detected intent."""
        intent_result = IntentResult(intent="complaint", confidence=0.72)
        prompt = pipeline._build_prompt("I have a problem", intent_result, "Context.")
        assert "complaint" in prompt

    def test_prompt_contains_context(self, pipeline):
        """Built prompt should include the retrieval context."""
        intent_result = IntentResult(intent="general_greeting", confidence=0.9)
        context = "Special context about greetings."
        prompt = pipeline._build_prompt("Hello!", intent_result, context)
        assert context in prompt


# ─── Full Pipeline Tests ──────────────────────────────────────────────────────

class TestChatbotPipelineProcess:
    """Integration-style tests for the full pipeline.process() method."""

    @pytest.mark.asyncio
    async def test_empty_message_returns_graceful_response(
        self, pipeline, mock_db
    ):
        """Empty message should return a friendly error response."""
        result = await pipeline.process(
            user_id="test_user",
            message="   ",
            db=mock_db,
        )
        assert isinstance(result, ChatResponse)
        assert result.response  # non-empty response
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_pipeline_returns_chat_response(
        self, pipeline, mock_db, trained_classifier
    ):
        """Full pipeline should return a valid ChatResponse."""
        # Patch classifier to return a known intent
        mock_intent = IntentResult(intent="general_greeting", confidence=0.9)

        # Patch chroma to return no documents (avoid needing real ChromaDB)
        mock_chroma = MagicMock()
        mock_chroma.query.return_value = []

        # Patch LLM to return a known response
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ("Hello! Welcome to Curify!", False)

        with patch.object(pipeline, "_get_classifier", return_value=trained_classifier), \
             patch.object(pipeline, "_get_chroma", return_value=mock_chroma), \
             patch.object(pipeline, "_get_llm", return_value=mock_llm), \
             patch.object(pipeline, "_ensure_user", new_callable=AsyncMock), \
             patch.object(pipeline, "_get_or_create_conversation", new_callable=AsyncMock) as mock_conv, \
             patch.object(pipeline, "_log_message", new_callable=AsyncMock):

            mock_conv.return_value = MagicMock(id="conv-123")

            result = await pipeline.process(
                user_id="test_user_1",
                message="Hello!",
                db=mock_db,
            )

        assert isinstance(result, ChatResponse)
        assert result.response == "Hello! Welcome to Curify!"
        assert result.intent in ["general_greeting", "general_farewell", "pricing", "refund_request"]
        assert isinstance(result.confidence, float)
        assert isinstance(result.latency_ms, int)

    @pytest.mark.asyncio
    async def test_tanglish_input_processed(self, pipeline, mock_db):
        """Tanglish input should be processed without exceptions."""
        mock_chroma = MagicMock()
        mock_chroma.query.return_value = []

        mock_llm = MagicMock()
        mock_llm.generate.return_value = ("Vanakkam! Welcome to Curify!", False)

        mock_clf = MagicMock()
        mock_clf.predict.return_value = IntentResult(
            intent="general_greeting", confidence=0.85
        )

        with patch.object(pipeline, "_get_classifier", return_value=mock_clf), \
             patch.object(pipeline, "_get_chroma", return_value=mock_chroma), \
             patch.object(pipeline, "_get_llm", return_value=mock_llm), \
             patch.object(pipeline, "_ensure_user", new_callable=AsyncMock), \
             patch.object(pipeline, "_get_or_create_conversation", new_callable=AsyncMock) as mock_conv, \
             patch.object(pipeline, "_log_message", new_callable=AsyncMock):

            mock_conv.return_value = MagicMock(id="conv-456")

            result = await pipeline.process(
                user_id="user_tanglish",
                message="Vanakkam! Curify pathi sollunga.",
                db=mock_db,
            )

        assert isinstance(result, ChatResponse)
        assert result.response

    @pytest.mark.asyncio
    async def test_very_long_input_handled(self, pipeline, mock_db):
        """Input over 1000 chars should be truncated and processed."""
        long_msg = "Hello " * 300  # ~1800 chars

        mock_chroma = MagicMock()
        mock_chroma.query.return_value = []
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ("Short response.", False)
        mock_clf = MagicMock()
        mock_clf.predict.return_value = IntentResult(
            intent="general_greeting", confidence=0.8
        )

        with patch.object(pipeline, "_get_classifier", return_value=mock_clf), \
             patch.object(pipeline, "_get_chroma", return_value=mock_chroma), \
             patch.object(pipeline, "_get_llm", return_value=mock_llm), \
             patch.object(pipeline, "_ensure_user", new_callable=AsyncMock), \
             patch.object(pipeline, "_get_or_create_conversation", new_callable=AsyncMock) as mock_conv, \
             patch.object(pipeline, "_log_message", new_callable=AsyncMock):

            mock_conv.return_value = MagicMock(id="conv-789")

            result = await pipeline.process(
                user_id="user_long",
                message=long_msg,
                db=mock_db,
            )

        assert isinstance(result, ChatResponse)
        assert result.response

    @pytest.mark.asyncio
    async def test_retrieval_sources_populated(self, pipeline, mock_db):
        """When ChromaDB returns hits, sources should be populated in response."""
        chroma_hits = [
            {
                "id": "doc_0",
                "document": "What is Curify?",
                "metadata": {"intent": "product_inquiry", "response": "Curify is great."},
                "distance": 0.2,
                "similarity": 0.8,
            }
        ]

        mock_chroma = MagicMock()
        mock_chroma.query.return_value = chroma_hits
        mock_llm = MagicMock()
        mock_llm.generate.return_value = ("Curify is an AI platform.", False)
        mock_clf = MagicMock()
        mock_clf.predict.return_value = IntentResult(
            intent="product_inquiry", confidence=0.9
        )

        with patch.object(pipeline, "_get_classifier", return_value=mock_clf), \
             patch.object(pipeline, "_get_chroma", return_value=mock_chroma), \
             patch.object(pipeline, "_get_llm", return_value=mock_llm), \
             patch.object(pipeline, "_ensure_user", new_callable=AsyncMock), \
             patch.object(pipeline, "_get_or_create_conversation", new_callable=AsyncMock) as mock_conv, \
             patch.object(pipeline, "_log_message", new_callable=AsyncMock):

            mock_conv.return_value = MagicMock(id="conv-src")

            result = await pipeline.process(
                user_id="user_src",
                message="Tell me about Curify products",
                db=mock_db,
            )

        assert result.retrieval_used is True
        assert len(result.sources) == 1
        assert isinstance(result.sources[0], SourceSnippet)
        assert result.sources[0].similarity == 0.8
