"""
models/train.py
───────────────
Orchestrates the full training pipeline:

  1. Load training CSV from data/raw/
  2. Preprocess and encode labels
  3. Train TF-IDF + Logistic Regression pipeline
  4. Evaluate with cross-validation (accuracy, precision, recall, F1)
  5. Save model artefacts to models/saved/
  6. Ingest training data into ChromaDB (re-index)
  7. Print a summary report

Usage:
  python models/train.py                         # default CSV path from config
  python models/train.py --csv data/raw/my.csv   # custom CSV
  python models/train.py --reset-chroma          # clear ChromaDB before ingesting
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Allow running from project root: python models/train.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.classifier import IntentClassifier
from app.embeddings import ingest_csv_to_chroma, ChromaService, get_embedding_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate the training CSV.

    Args:
        csv_path: Path to the CSV file with columns [text, intent, response].

    Returns:
        Cleaned DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or data is empty.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    logger.info("Loading training data from: %s", csv_path)
    df = pd.read_csv(csv_path)

    required = {"text", "intent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["text", "intent"])
    df = df[df["text"].str.strip().astype(bool)]
    after = len(df)

    if after == 0:
        raise ValueError("No valid rows remain after cleaning.")

    logger.info("Loaded %d rows (%d dropped during cleaning).", after, before - after)

    intent_counts = df["intent"].value_counts()
    logger.info("Intent distribution:\n%s", intent_counts.to_string())

    return df


def evaluate_pipeline(
    classifier: IntentClassifier,
    df: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate the classifier with 5-fold stratified cross-validation.

    Args:
        classifier: A fresh (untrained) IntentClassifier instance.
        df: The full labelled dataset.

    Returns:
        Dict with mean accuracy, precision, recall, and F1 (macro average).
    """
    logger.info("Running 5-fold stratified cross-validation...")

    from sklearn.preprocessing import LabelEncoder
    X = df["text"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df["intent"].tolist())

    pipeline = classifier.build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="macro", zero_division=0),
        "recall": make_scorer(recall_score, average="macro", zero_division=0),
        "f1": make_scorer(f1_score, average="macro", zero_division=0),
    }

    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
    )

    metrics = {
        "accuracy": float(cv_results["test_accuracy"].mean()),
        "precision_macro": float(cv_results["test_precision"].mean()),
        "recall_macro": float(cv_results["test_recall"].mean()),
        "f1_macro": float(cv_results["test_f1"].mean()),
    }

    logger.info(
        "CV Results: Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
    )
    return metrics


def print_report(metrics: dict[str, float], elapsed: float) -> None:
    """
    Print a formatted training summary to stdout.

    Args:
        metrics: Cross-validation metrics dict.
        elapsed: Total training time in seconds.
    """
    border = "=" * 60
    print(f"\n{border}")
    print("  CURIFY AI ADVISOR — MODEL TRAINING REPORT")
    print(border)
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*40}")
    for metric, value in metrics.items():
        label = metric.replace("_", " ").title()
        print(f"  {label:<30} {value:>9.4f}")
    print(f"  {'-'*40}")
    print(f"  {'Training Time':<30} {elapsed:>8.2f}s")
    print(f"{border}\n")


def main(args: argparse.Namespace) -> None:
    """
    Main training entry point.

    Args:
        args: Parsed command-line arguments.
    """
    csv_path = Path(args.csv)
    start_time = time.time()

    # 1. Load data
    df = load_data(csv_path)

    # 2. Cross-validate FIRST (uses a fresh pipeline each fold)
    eval_classifier = IntentClassifier()
    metrics = evaluate_pipeline(eval_classifier, df)

    # 3. Train final model on full dataset
    logger.info("Training final model on full dataset...")
    final_classifier = IntentClassifier()
    train_result = final_classifier.train(df)
    logger.info("Full-dataset training accuracy: %.4f", train_result["train_accuracy"])

    # 4. Save model
    model_dir = Path(settings.model_dir)
    final_classifier.save(model_dir)
    logger.info("Model saved to: %s", model_dir)

    # 5. Ingest into ChromaDB
    logger.info("Ingesting training data into ChromaDB...")
    embedding_service = get_embedding_service()
    chroma_service = ChromaService(embedding_service=embedding_service)
    count = ingest_csv_to_chroma(
        csv_path=csv_path,
        chroma_service=chroma_service,
        reset=args.reset_chroma,
    )
    logger.info("ChromaDB: %d documents indexed.", count)

    elapsed = time.time() - start_time

    # 6. Print summary
    metrics["train_accuracy"] = train_result["train_accuracy"]
    print_report(metrics, elapsed)

    logger.info("Training complete. ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Curify AI Advisor intent classifier."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=settings.training_data_path,
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        default=False,
        help="Clear the ChromaDB collection before ingesting new data.",
    )
    parsed = parser.parse_args()
    main(parsed)
