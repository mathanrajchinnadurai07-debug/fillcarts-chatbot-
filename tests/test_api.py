"""
tests/test_api.py
─────────────────
FastAPI endpoint integration tests using httpx AsyncClient.

Tests:
  - GET  /health        → 200 with expected fields
  - GET  /             → 200 welcome message
  - POST /chat          → 200 with valid response structure
  - POST /chat          → 422 on missing fields
  - POST /chat          → 400/422 on empty message
  - POST /chat          → handles 500+ char message gracefully
  - POST /chat          → handles Tanglish input
  - POST /chat          → handles SQL injection attempt in message
  - POST /train         → 401 without API key
  - POST /train         → 403 with wrong API key
  - POST /train         → 202 accepted with correct API key
  - GET  /history/{uid} → 200 with correct structure
  - GET  /history/{uid} → 400 with injection in user_id
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_chatbot_pipeline():
    """Return a mock ChatbotPipeline that returns a canned ChatResponse."""
    from app.chatbot import ChatResponse, SourceSnippet

    mock_pipeline = MagicMock()
    mock_pipeline.process = AsyncMock(
        return_value=ChatResponse(
            response="Hello! Welcome to Curify AI Advisor.",
            intent="general_greeting",
            confidence=0.92,
            fallback_used=False,
            retrieval_used=True,
            sources=[
                SourceSnippet(
                    text="Hello, welcome to Curify!",
                    intent="general_greeting",
                    similarity=0.91,
                    doc_id="doc_0",
                )
            ],
            latency_ms=120,
            conversation_id="conv-test-123",
        )
    )
    return mock_pipeline


@pytest_asyncio.fixture(scope="module")
async def async_client(mock_chatbot_pipeline):
    """
    Create an async httpx client pointed at the FastAPI test app.
    Patches the database and pipeline to avoid real I/O.
    """
    from app.main import app
    from app.database import get_db

    # Mock async DB session
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None), scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
    mock_db.add = MagicMock()
    mock_db.flush = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.rollback = AsyncMock()
    mock_db.close = AsyncMock()

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.main.create_db_tables", new_callable=AsyncMock), \
         patch("app.main.get_chroma_service") as mock_chroma, \
         patch("app.main.get_classifier") as mock_clf, \
         patch("app.main.get_chatbot_pipeline", return_value=mock_chatbot_pipeline):

        mock_chroma.return_value.count.return_value = 50
        mock_clf.return_value.is_trained = True
        mock_clf.return_value.classes = ["general_greeting", "pricing", "complaint"]

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"X-API-Key": settings.chatbot_api_key},
        ) as client:
            yield client

    app.dependency_overrides.clear()


# ─── Health Endpoint ──────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client):
        """Health endpoint should return HTTP 200."""
        response = await async_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_has_status_field(self, async_client):
        """Health response should contain a 'status' field."""
        response = await async_client.get("/health")
        data = response.json()
        assert "status" in data

    @pytest.mark.asyncio
    async def test_health_response_has_version(self, async_client):
        """Health response should include app version."""
        response = await async_client.get("/health")
        data = response.json()
        assert "version" in data or "app" in data


# ─── Root Endpoint ────────────────────────────────────────────────────────────

class TestRootEndpoint:
    """Tests for GET /."""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, async_client):
        """Root endpoint should return HTTP 200."""
        response = await async_client.get("/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_root_response_has_message(self, async_client):
        """Root response should include a message field."""
        response = await async_client.get("/")
        data = response.json()
        assert "message" in data


# ─── Chat Endpoint ────────────────────────────────────────────────────────────

class TestChatEndpoint:
    """Tests for POST /chat."""

    @pytest.mark.asyncio
    async def test_chat_returns_401_without_api_key(self, async_client):
        """Missing API key on /chat should return HTTP 401."""
        response = await async_client.post(
            "/chat",
            json={"user_id": "test_user_1", "message": "Hello!"},
            headers={"X-API-Key": ""},  # empty / missing
        )
        assert response.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_chat_returns_403_with_wrong_api_key(self, async_client):
        """Invalid API key on /chat should return HTTP 403."""
        response = await async_client.post(
            "/chat",
            json={"user_id": "test_user_1", "message": "Hello!"},
            headers={"X-API-Key": "invalid_key_999"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_chat_returns_200_with_valid_input(self, async_client):
        """Valid chat request should return HTTP 200."""
        response = await async_client.post(
            "/chat",
            json={"user_id": "test_user_1", "message": "Hello!"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_response_has_required_fields(self, async_client):
        """Chat response should contain all required fields."""
        response = await async_client.post(
            "/chat",
            json={"user_id": "test_user_2", "message": "What does Curify do?"},
        )
        data = response.json()
        required_fields = {
            "response", "intent", "confidence", "fallback_used",
            "retrieval_used", "sources", "latency_ms",
        }
        assert required_fields.issubset(set(data.keys()))

    @pytest.mark.asyncio
    async def test_chat_returns_422_on_missing_fields(self, async_client):
        """Missing required fields should return HTTP 422."""
        response = await async_client.post(
            "/chat",
            json={"user_id": "user_x"},  # missing 'message'
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_returns_422_on_empty_message(self, async_client):
        """Empty message string should return HTTP 422 (min_length validation)."""
        response = await async_client.post(
            "/chat",
            json={"user_id": "user_x", "message": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_handles_long_message(self, async_client):
        """Message over 1000 chars should be rejected by Pydantic (max_length=1000)."""
        long_msg = "A" * 1100
        response = await async_client.post(
            "/chat",
            json={"user_id": "user_long", "message": long_msg},
        )
        # max_length=1000 in Pydantic → 422
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_handles_tanglish_input(self, async_client):
        """Tanglish (Tamil-English mixed) input should be processed successfully."""
        response = await async_client.post(
            "/chat",
            json={
                "user_id": "tanglish_user",
                "message": "Vanakkam! Curify pathi sollunga.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response"]

    @pytest.mark.asyncio
    async def test_chat_sql_injection_in_message_handled(self, async_client):
        """SQL injection in message body should be handled safely."""
        response = await async_client.post(
            "/chat",
            json={
                "user_id": "safe_user",
                "message": "'; DROP TABLE messages; -- hello",
            },
        )
        # Should either return 200 (sanitised) or 422, but never 500
        assert response.status_code in (200, 422)

    @pytest.mark.asyncio
    async def test_chat_sql_injection_in_user_id_rejected(self, async_client):
        """SQL injection in user_id should be rejected with 422."""
        response = await async_client.post(
            "/chat",
            json={
                "user_id": "'; DROP TABLE users; --",
                "message": "Hello",
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_confidence_is_between_0_and_1(self, async_client):
        """Confidence in response should be between 0.0 and 1.0."""
        response = await async_client.post(
            "/chat",
            json={"user_id": "conf_user", "message": "Hello there!"},
        )
        if response.status_code == 200:
            data = response.json()
            assert 0.0 <= data["confidence"] <= 1.0


# ─── Train Endpoint ───────────────────────────────────────────────────────────

class TestTrainEndpoint:
    """Tests for POST /train."""

    @pytest.mark.asyncio
    async def test_train_returns_401_without_api_key(self, async_client):
        """Missing API key should return HTTP 401."""
        response = await async_client.post("/train", json={})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_train_returns_403_with_wrong_api_key(self, async_client):
        """Wrong API key should return HTTP 403."""
        response = await async_client.post(
            "/train",
            json={},
            headers={"X-API-Key": "wrong_key_here"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_train_returns_200_with_correct_api_key(self, async_client):
        """Correct API key should accept the training request."""
        with patch("app.main.asyncio.create_task"):
            response = await async_client.post(
                "/train",
                json={"reset_chroma": False},
                headers={"X-API-Key": settings.train_api_key},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "job_id" in data


# ─── History Endpoint ─────────────────────────────────────────────────────────

class TestHistoryEndpoint:
    """Tests for GET /history/{user_id}."""

    @pytest.mark.asyncio
    async def test_history_returns_200(self, async_client):
        """History endpoint should return 200 for valid user_id."""
        response = await async_client.get("/history/test_user_1")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_history_returns_correct_structure(self, async_client):
        """History response should have 'user_id', 'messages', 'total' keys."""
        response = await async_client.get("/history/test_user_1")
        data = response.json()
        assert "user_id" in data
        assert "messages" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_history_injection_in_user_id_rejected(self, async_client):
        """SQL injection in URL path user_id should return 400."""
        response = await async_client.get(
            "/history/'; DROP TABLE users; --"
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_history_limit_query_param(self, async_client):
        """Limit parameter should be accepted and validated."""
        response = await async_client.get("/history/test_user_1?limit=5")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_history_invalid_limit_rejected(self, async_client):
        """Limit=0 should be rejected (ge=1 constraint)."""
        response = await async_client.get("/history/test_user_1?limit=0")
        assert response.status_code == 422
