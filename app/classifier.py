"""
app/classifier.py
─────────────────
Intent classification using TF-IDF + Logistic Regression.

Strategy:
  1. Vectorise text with TF-IDF (character n-grams + word n-grams combined).
  2. Classify with Logistic Regression, returning a class + probability.
  3. If max class probability < CONFIDENCE_THRESHOLD, fall back to
     ChromaDB embedding similarity search to determine the intent.
  4. Both the vectoriser and the model are serialised with joblib.

Tanglish (Tamil-English mixed) support:
  - TF-IDF character n-grams naturally capture Tamil transliterated tokens.
  - No explicit language detection is required.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import settings
from app.embeddings import get_chroma_service

logger = logging.getLogger(__name__)

# ─── File Paths ───────────────────────────────────────────────────────────────
MODEL_DIR = Path(settings.model_dir)
PIPELINE_PATH = MODEL_DIR / "intent_pipeline.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"


# ─── Prediction Result ────────────────────────────────────────────────────────

class IntentResult:
    """Structured result returned by the classifier predict() method."""

    __slots__ = ("intent", "confidence", "fallback_used", "method")

    def __init__(
        self,
        intent: str,
        confidence: float,
        fallback_used: bool = False,
        method: str = "classifier",
    ) -> None:
        self.intent = intent
        self.confidence = confidence
        self.fallback_used = fallback_used
        self.method = method  # 'classifier' | 'embedding_fallback' | 'default'

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "fallback_used": self.fallback_used,
            "method": self.method,
        }

    def __repr__(self) -> str:
        return (
            f"<IntentResult intent={self.intent!r} "
            f"conf={self.confidence:.3f} method={self.method!r}>"
        )


# ─── Classifier ───────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    TF-IDF + Logistic Regression intent classifier with embedding fallback.

    The pipeline combines:
      - TfidfVectorizer with both word and character n-grams for robustness
        against Tanglish / code-mixed input.
      - LogisticRegression with balanced class weights to handle uneven
        intent distributions.

    When confidence < threshold, the classifier falls back to ChromaDB
    embedding similarity to find the closest matching intent.
    """

    def __init__(
        self,
        confidence_threshold: float = settings.confidence_threshold,
    ) -> None:
        """
        Initialise the classifier.

        Args:
            confidence_threshold: Minimum probability to trust the classifier.
                                  Below this, embedding fallback is used.
        """
        self.confidence_threshold = confidence_threshold
        self.pipeline: Pipeline | None = None
        self.label_encoder: LabelEncoder | None = None
        self._is_trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def build_pipeline(self) -> Pipeline:
        """
        Construct the sklearn Pipeline (TF-IDF → Logistic Regression).

        Returns:
            A fresh sklearn Pipeline ready for fitting.
        """
        vectoriser = TfidfVectorizer(
            analyzer="char_wb",       # character n-grams — great for Tanglish
            ngram_range=(2, 4),
            max_features=20_000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        # Combine char and word features via a second vectoriser isn't needed
        # because char_wb at (2,4) subsumes word boundaries well for our use-case.

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
            multi_class="multinomial",
        )
        return Pipeline([("tfidf", vectoriser), ("clf", clf)])

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Fit the pipeline on labelled data.

        Args:
            df: DataFrame with 'text' and 'intent' columns.

        Returns:
            Dict with training accuracy.

        Raises:
            ValueError: If required columns are missing or df is empty.
        """
        if df.empty:
            raise ValueError("Training DataFrame is empty.")
        for col in ("text", "intent"):
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in training data.")

        df = df.dropna(subset=["text", "intent"])
        df = df[df["text"].str.strip().astype(bool)]

        X = df["text"].tolist()
        y_raw = df["intent"].tolist()

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw)

        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self._is_trained = True

        train_acc = self.pipeline.score(X, y)
        logger.info("Classifier trained. Training accuracy: %.4f", train_acc)
        return {"train_accuracy": float(train_acc)}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_dir: Path = MODEL_DIR) -> None:
        """
        Persist the trained pipeline and label encoder to disk.

        Args:
            model_dir: Directory where model files will be saved.

        Raises:
            RuntimeError: If the classifier has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before saving.")

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_dir / "intent_pipeline.joblib")
        joblib.dump(self.label_encoder, model_dir / "label_encoder.joblib")
        logger.info("Classifier saved to %s", model_dir)

    def load(self, model_dir: Path = MODEL_DIR) -> None:
        """
        Load a previously trained pipeline and label encoder from disk.

        Args:
            model_dir: Directory containing the saved model files.

        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If loading fails.
        """
        pipeline_path = model_dir / "intent_pipeline.joblib"
        encoder_path = model_dir / "label_encoder.joblib"

        if not pipeline_path.exists():
            raise FileNotFoundError(f"Model file not found: {pipeline_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

        try:
            self.pipeline = joblib.load(pipeline_path)
            self.label_encoder = joblib.load(encoder_path)
            self._is_trained = True
            logger.info("Classifier loaded from %s", model_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to load classifier: {exc}") from exc

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> IntentResult:
        """
        Classify the intent of a user message.

        Steps:
          1. Sanitise input (strip, truncate to 512 chars).
          2. Run classifier pipeline → get probabilities.
          3. If max probability >= threshold → return classifier result.
          4. Else → query ChromaDB for nearest neighbour intent (fallback).
          5. If ChromaDB also fails → return 'general_greeting' as safe default.

        Args:
            text: The raw user message (may be Tanglish / mixed language).

        Returns:
            IntentResult with intent, confidence, and metadata.
        """
        if not self._is_trained or self.pipeline is None:
            logger.warning("Classifier not trained; loading from disk...")
            try:
                self.load()
            except Exception as exc:
                logger.error("Could not load classifier: %s", exc)
                return IntentResult(
                    intent="general_greeting",
                    confidence=0.0,
                    fallback_used=True,
                    method="default",
                )

        # Sanitise
        cleaned = text.strip()[:512] if text else ""
        if not cleaned:
            return IntentResult(
                intent="general_greeting",
                confidence=0.0,
                fallback_used=True,
                method="default",
            )

        try:
            proba = self.pipeline.predict_proba([cleaned])[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            intent = self.label_encoder.inverse_transform([best_idx])[0]

            if confidence >= self.confidence_threshold:
                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    fallback_used=False,
                    method="classifier",
                )

            # ── Embedding Fallback ────────────────────────────────────────────
            logger.debug(
                "Low confidence (%.3f < %.3f) for %r. Trying embedding fallback.",
                confidence, self.confidence_threshold, cleaned,
            )
            return self._embedding_fallback(cleaned, classifier_confidence=confidence)

        except Exception as exc:
            logger.error("Classifier predict() failed for %r: %s", cleaned, exc)
            return IntentResult(
                intent="general_greeting",
                confidence=0.0,
                fallback_used=True,
                method="default",
            )

    def _embedding_fallback(
        self, text: str, classifier_confidence: float
    ) -> IntentResult:
        """
        Use ChromaDB similarity search to determine intent when classifier
        confidence is too low.

        Args:
            text: The sanitised user message.
            classifier_confidence: The classifier's best probability score.

        Returns:
            IntentResult using embedding-retrieved intent.
        """
        try:
            chroma = get_chroma_service()
            hits = chroma.query(text, top_k=1)

            if hits:
                top_hit = hits[0]
                intent = top_hit["metadata"].get("intent", "general_greeting")
                similarity = top_hit.get("similarity", 0.0)
                logger.debug(
                    "Embedding fallback: intent=%r, similarity=%.4f", intent, similarity
                )
                return IntentResult(
                    intent=intent,
                    confidence=similarity,
                    fallback_used=True,
                    method="embedding_fallback",
                )
        except Exception as exc:
            logger.error("Embedding fallback failed: %s", exc)

        # Hard default
        return IntentResult(
            intent="general_greeting",
            confidence=classifier_confidence,
            fallback_used=True,
            method="default",
        )

    @property
    def is_trained(self) -> bool:
        """True if the classifier has been trained or loaded."""
        return self._is_trained

    @property
    def classes(self) -> list[str]:
        """Return sorted list of known intent labels."""
        if self.label_encoder is None:
            return []
        return list(self.label_encoder.classes_)


# ─── Module-level singleton ───────────────────────────────────────────────────

_classifier: IntentClassifier | None = None


def get_classifier() -> IntentClassifier:
    """
    Return the module-level IntentClassifier singleton.

    Automatically attempts to load a pre-trained model from disk on first call.
    """
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
        try:
            _classifier.load()
            logger.info("IntentClassifier loaded from saved models.")
        except FileNotFoundError:
            logger.warning(
                "No saved classifier found at %s. "
                "Run 'python models/train.py' to train the model.",
                MODEL_DIR,
            )
    return _classifier
