"""
tests/test_classifier.py
────────────────────────
Unit tests for the IntentClassifier.

Tests:
  - Training on minimal dataset completes without error
  - Prediction returns a valid IntentResult
  - High-confidence prediction is correct for clear inputs
  - Low-confidence input triggers fallback flag
  - Empty string returns a safe default
  - Tanglish (Tamil-English mixed) input is handled gracefully
  - Input exceeding 512 chars is truncated and still returns a result
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classifier import IntentClassifier, IntentResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_classifier() -> IntentClassifier:
    """Train a classifier on a small dataset and return it."""
    data = {
        "text": [
            "Hello how are you",
            "Hi there!",
            "Good morning",
            "What is the price?",
            "How much does it cost?",
            "Pricing details please",
            "I have a complaint",
            "Something is broken",
            "This is not working",
            "I want a refund",
            "Give me my money back",
            "Refund request",
            "Bye goodbye",
            "See you later",
            "Thanks bye",
            "Vanakkam eppadi irukeenga",
            "Naan student discount vennum",
            "Ethanai pairam?",
        ],
        "intent": [
            "general_greeting", "general_greeting", "general_greeting",
            "pricing", "pricing", "pricing",
            "complaint", "complaint", "complaint",
            "refund_request", "refund_request", "refund_request",
            "general_farewell", "general_farewell", "general_farewell",
            "general_greeting", "pricing", "pricing",
        ],
    }
    df = pd.DataFrame(data)
    clf = IntentClassifier(confidence_threshold=0.3)
    clf.train(df)
    return clf


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestIntentClassifierTraining:
    """Tests related to classifier training."""

    def test_training_completes_successfully(self, trained_classifier):
        """Classifier should be marked as trained after fitting."""
        assert trained_classifier.is_trained is True

    def test_classes_populated_after_training(self, trained_classifier):
        """Known classes should be non-empty after training."""
        classes = trained_classifier.classes
        assert isinstance(classes, list)
        assert len(classes) > 0

    def test_training_returns_accuracy(self):
        """train() should return a dict with train_accuracy."""
        df = pd.DataFrame({
            "text": ["Hello", "Hi", "Bye", "Goodbye", "What is price?", "Cost?"],
            "intent": ["general_greeting", "general_greeting",
                       "general_farewell", "general_farewell",
                       "pricing", "pricing"],
        })
        clf = IntentClassifier(confidence_threshold=0.3)
        result = clf.train(df)
        assert "train_accuracy" in result
        assert 0.0 <= result["train_accuracy"] <= 1.0

    def test_training_raises_on_empty_dataframe(self):
        """train() should raise ValueError for empty DataFrames."""
        clf = IntentClassifier()
        with pytest.raises(ValueError, match="empty"):
            clf.train(pd.DataFrame())

    def test_training_raises_on_missing_columns(self):
        """train() should raise ValueError if required columns are missing."""
        clf = IntentClassifier()
        df = pd.DataFrame({"text": ["hello"], "wrong_col": ["x"]})
        with pytest.raises(ValueError):
            clf.train(df)


class TestIntentClassifierPrediction:
    """Tests related to classifier prediction."""

    def test_predict_returns_intent_result(self, trained_classifier):
        """predict() should return an IntentResult instance."""
        result = trained_classifier.predict("Hello there!")
        assert isinstance(result, IntentResult)

    def test_predict_greeting_intent(self, trained_classifier):
        """Clear greeting should return 'general_greeting' intent."""
        result = trained_classifier.predict("Hello how are you")
        assert result.intent == "general_greeting"
        assert result.confidence > 0.0

    def test_predict_pricing_intent(self, trained_classifier):
        """Clear pricing query should be classified correctly."""
        result = trained_classifier.predict("What is the price of the product?")
        assert result.intent == "pricing"

    def test_predict_farewell_intent(self, trained_classifier):
        """Clear farewell should be classified correctly."""
        result = trained_classifier.predict("Bye see you later goodbye")
        assert result.intent == "general_farewell"

    def test_predict_confidence_is_float(self, trained_classifier):
        """Confidence should be a float between 0 and 1."""
        result = trained_classifier.predict("Hello!")
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_empty_string_returns_default(self, trained_classifier):
        """Empty input should return a safe default without raising."""
        result = trained_classifier.predict("")
        assert isinstance(result, IntentResult)
        assert result.intent == "general_greeting"
        assert result.fallback_used is True

    def test_predict_whitespace_only_returns_default(self, trained_classifier):
        """Whitespace-only input should return default intent."""
        result = trained_classifier.predict("   ")
        assert result.fallback_used is True

    def test_predict_long_input_truncated(self, trained_classifier):
        """Input over 512 chars should be truncated and still return a result."""
        long_text = "hello " * 200  # 1200 chars
        result = trained_classifier.predict(long_text)
        assert isinstance(result, IntentResult)

    def test_predict_tanglish_input(self, trained_classifier):
        """Tanglish (Tamil-English) input should not raise and return a result."""
        tanglish = "Vanakkam naan student discount vennum"
        result = trained_classifier.predict(tanglish)
        assert isinstance(result, IntentResult)
        # Should classify as pricing or greeting (trained on this data)
        assert result.intent in trained_classifier.classes

    def test_predict_sql_injection_attempt(self, trained_classifier):
        """SQL injection attempts in message text should be handled safely."""
        injection = "'; DROP TABLE users; --"
        result = trained_classifier.predict(injection)
        assert isinstance(result, IntentResult)
        # Should not raise, fallback is acceptable

    def test_predict_special_characters(self, trained_classifier):
        """Special characters in input should not cause errors."""
        special = "Hello! @#$%^&*() 你好 مرحبا"
        result = trained_classifier.predict(special)
        assert isinstance(result, IntentResult)

    def test_predict_repeated_calls_consistent(self, trained_classifier):
        """Repeated identical calls should return the same intent."""
        text = "What is the cost?"
        results = [trained_classifier.predict(text) for _ in range(5)]
        intents = [r.intent for r in results]
        assert len(set(intents)) == 1, "Repeated predictions should be deterministic."


class TestIntentClassifierPersistence:
    """Tests related to model save/load."""

    def test_save_and_load(self, tmp_path, trained_classifier):
        """Model should be saveable and reloadable from disk."""
        trained_classifier.save(model_dir=tmp_path)

        new_clf = IntentClassifier()
        new_clf.load(model_dir=tmp_path)

        assert new_clf.is_trained is True
        result = new_clf.predict("Hello!")
        assert isinstance(result, IntentResult)

    def test_load_raises_if_files_missing(self, tmp_path):
        """load() should raise FileNotFoundError if model files are absent."""
        clf = IntentClassifier()
        with pytest.raises(FileNotFoundError):
            clf.load(model_dir=tmp_path)
