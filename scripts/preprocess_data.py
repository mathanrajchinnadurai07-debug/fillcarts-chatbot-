"""
scripts/preprocess_data.py
──────────────────────────
Standalone preprocessing script for raw training data.

Responsibilities:
  1. Load the raw CSV from data/raw/
  2. Clean and deduplicate text samples
  3. Normalise Tanglish / Unicode text
  4. Validate intent labels against the known intents list
  5. Export a cleaned CSV to data/processed/
  6. Print a summary statistics report

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --input data/raw/my_data.csv --output data/processed/clean.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")

# ─── Known intents ─────────────────────────────────────────────────────────────
KNOWN_INTENTS: set[str] = {
    "general_greeting",
    "product_inquiry",
    "pricing",
    "order_status",
    "complaint",
    "refund_request",
    "ai_advice",
    "account_support",
    "general_farewell",
    "escalate_human",
}


# ─── Text Normalisation ────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """
    Normalise a raw text sample for consistent model training.

    Steps:
      - Unicode NFC normalisation (handles Tamil script characters)
      - Strip leading/trailing whitespace
      - Collapse consecutive whitespace to a single space
      - Remove control characters (null bytes, form feeds, etc.)
      - Lowercase (preserves Tamil characters correctly)

    Args:
        text: Raw input string, may contain Tanglish, Tamil, or English.

    Returns:
        Normalised string.
    """
    if not isinstance(text, str):
        return ""

    # NFC normalisation handles Tamil Unicode correctly
    text = unicodedata.normalize("NFC", text)

    # Remove control characters (keep printable + whitespace)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " ")
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_intent(intent: str) -> bool:
    """
    Check whether an intent label belongs to the known intents set.

    Args:
        intent: Intent label string from the CSV.

    Returns:
        True if the intent is recognised, False otherwise.
    """
    return intent.strip().lower() in KNOWN_INTENTS


# ─── Main Preprocessing ────────────────────────────────────────────────────────

def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Load, clean, validate, and deduplicate the training CSV.

    Args:
        input_path: Path to the raw CSV with [text, intent, response] columns.
        output_path: Path where the cleaned CSV will be written.

    Returns:
        Cleaned DataFrame.

    Raises:
        FileNotFoundError: If input CSV does not exist.
        ValueError: If required columns are missing.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("Loading raw data from: %s", input_path)
    df = pd.read_csv(input_path)

    # ── Validate columns ───────────────────────────────────────────────────────
    required = {"text", "intent", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    initial_count = len(df)
    logger.info("Initial row count: %d", initial_count)

    # ── Drop nulls ─────────────────────────────────────────────────────────────
    df = df.dropna(subset=["text", "intent"])
    logger.info("After dropping nulls: %d rows", len(df))

    # ── Normalise text ─────────────────────────────────────────────────────────
    df["text"] = df["text"].apply(normalise_text)
    df["response"] = df["response"].fillna("").apply(normalise_text)
    df["intent"] = df["intent"].str.strip().str.lower()

    # ── Remove empty text rows ─────────────────────────────────────────────────
    df = df[df["text"].str.len() > 0]
    logger.info("After removing empty text rows: %d rows", len(df))

    # ── Validate intent labels ─────────────────────────────────────────────────
    invalid_mask = ~df["intent"].apply(is_valid_intent)
    if invalid_mask.any():
        invalid_intents = df.loc[invalid_mask, "intent"].unique().tolist()
        logger.warning(
            "Found %d rows with unknown intents: %s. Dropping them.",
            invalid_mask.sum(), invalid_intents,
        )
        df = df[~invalid_mask]
    logger.info("After intent validation: %d rows", len(df))

    # ── Deduplicate on text ────────────────────────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    dropped = before_dedup - len(df)
    if dropped:
        logger.info("Deduplicated %d duplicate text rows.", dropped)

    # ── Reset index ────────────────────────────────────────────────────────────
    df = df[["text", "intent", "response"]].reset_index(drop=True)

    # ── Save output ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned data saved to: %s (%d rows)", output_path, len(df))

    # ── Summary report ─────────────────────────────────────────────────────────
    _print_summary(df, initial_count)
    return df


def _print_summary(df: pd.DataFrame, initial_count: int) -> None:
    """
    Print a formatted preprocessing summary to stdout.

    Args:
        df: The cleaned DataFrame.
        initial_count: The original row count before cleaning.
    """
    border = "=" * 55
    print(f"\n{border}")
    print("  CURIFY AI ADVISOR — DATA PREPROCESSING REPORT")
    print(border)
    print(f"  {'Initial rows':<30} {initial_count:>10}")
    print(f"  {'Final rows':<30} {len(df):>10}")
    print(f"  {'Rows removed':<30} {initial_count - len(df):>10}")
    print(f"\n  Intent Distribution:")
    print(f"  {'-'*40}")
    for intent, count in df["intent"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {intent:<32} {count:>5}  ({pct:.1f}%)")
    print(f"{border}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess Curify AI Advisor training data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=settings.training_data_path,
        help="Path to the raw input CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/clean_training_data.csv",
        help="Path for the cleaned output CSV.",
    )
    args = parser.parse_args()

    try:
        preprocess(
            input_path=Path(args.input),
            output_path=Path(args.output),
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
