"""
app/llm_client.py
─────────────────
Smart response generation — works 100% offline without any API key.

Priority order:
  1. If ChromaDB returned a high-similarity match (≥0.75) → use that answer directly
  2. If medium similarity (0.5-0.75) → blend retrieved context into a template
  3. If low/no match → use intent-based template
  4. (Optional) If GEMINI_API_KEY is set → use Gemini 2.5 Flash instead of above

No external API calls are ever required. The chatbot works fully offline.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Response Cache ────────────────────────────────────────────────────────────
_response_cache: TTLCache = TTLCache(
    maxsize=settings.llm_cache_max_size,
    ttl=settings.llm_cache_ttl_seconds,
)

# ─── Similarity Thresholds ────────────────────────────────────────────────────
HIGH_SIMILARITY = 0.75   # Use retrieved answer directly
MED_SIMILARITY  = 0.50   # Blend retrieved answer with template


# ─── Intent Templates (offline fallback) ──────────────────────────────────────
INTENT_TEMPLATES: dict[str, str] = {
    "general_greeting": (
        "Hello! Welcome to {business_name}. I'm your AI assistant, here to help you "
        "with product information, pricing, orders, and more. How can I assist you today? 😊"
    ),
    "product_inquiry": (
        "Great question! {business_name} offers a range of products and services designed "
        "to meet your needs. Could you share more details about what you're looking for so "
        "I can give you the most accurate information?"
    ),
    "pricing": (
        "I'd be happy to help with pricing information! {business_name} offers flexible plans "
        "to suit different needs. Please share what product or service you're interested in, "
        "and I'll provide the exact pricing details."
    ),
    "order_status": (
        "I can help you track your order! Please share your Order ID and registered email "
        "address. Our team will provide an update within a few minutes."
    ),
    "complaint": (
        "I sincerely apologise for the inconvenience you're experiencing. Your concern is "
        "important to us and I've noted your issue. Our support team will follow up with you "
        "within 2 hours. Thank you for your patience. 🙏"
    ),
    "refund_request": (
        "I understand your request for a refund. Refunds are typically processed within "
        "5-7 business days for eligible purchases. Please share your Order ID and I'll "
        "initiate the refund process right away."
    ),
    "ai_advice": (
        "I'd be happy to provide guidance! To give you the most relevant advice, could you "
        "share more details about your specific needs or goals? Our team is here to help you "
        "make the best decisions."
    ),
    "account_support": (
        "I can help with your account! For login issues, please try the 'Forgot Password' "
        "option. If you're facing other account problems, please share your registered email "
        "and describe the issue — I'll assist you immediately."
    ),
    "general_farewell": (
        "Thank you for reaching out to {business_name}! It was a pleasure assisting you. "
        "Have a wonderful day and feel free to return anytime. Goodbye! 👋😊"
    ),
    "escalate_human": (
        "Absolutely! I'm connecting you with a live support agent now. Please hold for "
        "approximately 2-3 minutes. Our agents are available Monday–Saturday, 9 AM–7 PM. "
        "Your conversation history has been shared with the agent."
    ),
    "default": (
        "Thank you for your message! I'm here to help with information about {business_name}'s "
        "products, pricing, orders, and services. Could you please clarify what you need "
        "assistance with? I'll do my best to help! 🌟"
    ),
}

# Business name placeholder (overridden from config or knowledge base)
BUSINESS_NAME = "Curify AI Advisor"


# ─── Offline Context-Aware Responder ──────────────────────────────────────────

class OfflineResponder:
    """
    Generates intelligent responses entirely without any external API.

    Uses retrieved ChromaDB answers and intent templates to compose
    contextually relevant, helpful responses.
    """

    def __init__(self, business_name: str = BUSINESS_NAME) -> None:
        """
        Initialise the offline responder.

        Args:
            business_name: The name of the business (injected into templates).
        """
        self.business_name = business_name

    def _fill_template(self, template: str) -> str:
        """Fill template placeholders with actual values."""
        return template.replace("{business_name}", self.business_name)

    def generate(
        self,
        user_message: str,
        intent: str,
        retrieved_hits: list[dict[str, Any]],
    ) -> tuple[str, bool]:
        """
        Generate a response using retrieved context and/or templates.

        Strategy:
          1. High-similarity hit (≥0.75) → return that document's response directly
          2. Medium-similarity (0.5-0.75) → prefix template with relevant context
          3. Low/no match → return pure intent template
          4. No intent match → return default template

        Args:
            user_message: The user's sanitised message.
            intent: Classified intent label.
            retrieved_hits: List of ChromaDB hit dicts with 'metadata', 'similarity'.

        Returns:
            Tuple of (response_text, used_fallback_flag).
            used_fallback_flag is True when using pure template (no retrieval).
        """
        # ── Try high-confidence direct answer ─────────────────────────────────
        if retrieved_hits:
            top_hit = retrieved_hits[0]
            similarity = top_hit.get("similarity", 0.0)
            stored_response = top_hit.get("metadata", {}).get("response", "")

            if similarity >= HIGH_SIMILARITY and stored_response.strip():
                logger.info(
                    "High-similarity hit (%.3f) — returning stored answer directly.",
                    similarity
                )
                return stored_response.strip(), False

            # ── Medium confidence — blend context with template ────────────────
            if similarity >= MED_SIMILARITY and stored_response.strip():
                logger.info(
                    "Medium-similarity hit (%.3f) — blending context with template.",
                    similarity
                )
                base = self._fill_template(
                    INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["default"])
                )
                blended = f"{stored_response.strip()}\n\n{base}"
                return blended, False

        # ── Pure template fallback ─────────────────────────────────────────────
        logger.info("No strong retrieval match — using intent template for '%s'.", intent)
        template = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["default"])
        return self._fill_template(template), True


# ─── Optional Gemini Client ───────────────────────────────────────────────────

class GeminiClient:
    """
    Optional Gemini 2.5 Flash wrapper — only active if GEMINI_API_KEY is set.
    Falls back gracefully to OfflineResponder if not configured.
    """

    def __init__(self) -> None:
        self._is_configured: bool = False
        self._model = None
        self._offline = OfflineResponder()

        if not settings.gemini_api_key:
            logger.info("GEMINI_API_KEY not set — using 100% offline mode.")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            self._model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=512,
                ),
            )
            self._is_configured = True
            logger.info("Gemini client ready: %s", settings.gemini_model)
        except Exception as exc:
            logger.warning("Gemini init failed (%s) — offline mode active.", exc)

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def generate(
        self,
        prompt: str = "",
        intent: str = "default",
        use_cache: bool = True,
        user_message: str = "",
        retrieved_hits: list[dict[str, Any]] | None = None,
    ) -> tuple[str, bool]:
        """
        Generate a response. Uses Gemini if configured, else offline mode.

        Args:
            prompt: Full RAG prompt (used by Gemini only).
            intent: Classified intent label.
            use_cache: Check response cache first.
            user_message: Raw user message (for offline mode).
            retrieved_hits: ChromaDB hits (for offline mode).

        Returns:
            Tuple of (response_text, used_fallback_flag).
        """
        hits = retrieved_hits or []

        # ── Always try offline first for high-similarity cache hit ─────────────
        if use_cache and prompt:
            key = self._cache_key(prompt)
            if key in _response_cache:
                logger.debug("Cache hit.")
                return _response_cache[key], False

        # ── Offline mode (no API key) ──────────────────────────────────────────
        if not self._is_configured:
            return self._offline.generate(user_message, intent, hits)

        # ── Gemini mode ───────────────────────────────────────────────────────
        try:
            from tenacity import retry, stop_after_attempt, wait_exponential
            start = time.perf_counter()
            response = self._model.generate_content(prompt)
            elapsed = int((time.perf_counter() - start) * 1000)

            if response and response.text:
                text = response.text.strip()
                logger.info("Gemini responded in %d ms.", elapsed)
                if use_cache and prompt:
                    _response_cache[self._cache_key(prompt)] = text
                return text, False
        except Exception as exc:
            logger.warning("Gemini call failed (%s) — switching to offline mode.", exc)

        # Fallback to offline if Gemini fails
        return self._offline.generate(user_message, intent, hits)

    def clear_cache(self) -> None:
        """Clear the LLM response cache."""
        _response_cache.clear()

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    @property
    def mode(self) -> str:
        return "gemini" if self._is_configured else "offline"


# ─── Module-level singleton ───────────────────────────────────────────────────

_gemini_client: GeminiClient | None = None


def get_llm_client() -> GeminiClient:
    """Return the module-level GeminiClient singleton."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def get_fallback(intent: str) -> str:
    """Return a template response for the given intent (for direct use)."""
    template = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["default"])
    return template.replace("{business_name}", BUSINESS_NAME)


# Keep backward compatibility alias
_get_fallback = get_fallback
