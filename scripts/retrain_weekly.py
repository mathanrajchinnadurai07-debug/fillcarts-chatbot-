"""
scripts/retrain_weekly.py
──────────────────────────
Automated weekly retraining script — designed to run via cron or GitHub Actions.

Workflow:
  1. Query SQLite for newly logged messages since the last retrain
  2. Filter for messages that are diverse and high-quality (not pure fallbacks)
  3. Append valid new samples to the training CSV
  4. Run preprocessing (deduplication, normalisation)
  5. Trigger models/train.py to retrain the classifier + re-index ChromaDB
  6. Write a retraining log file with timestamp and metrics

Cron setup (run every Sunday at 2 AM):
    0 2 * * 0 cd /app && python scripts/retrain_weekly.py >> logs/retrain.log 2>&1

GitHub Actions: see .github/workflows/deploy.yml (schedule trigger)

Usage:
    python scripts/retrain_weekly.py
    python scripts/retrain_weekly.py --min-samples 5 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal, Message, create_db_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain_weekly")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_CSV = PROJECT_ROOT / settings.training_data_path.lstrip("./")
RETRAIN_LOG = PROJECT_ROOT / "logs" / "retrain_log.jsonl"
TRAIN_SCRIPT = PROJECT_ROOT / "models" / "train.py"


# ─── DB Query ─────────────────────────────────────────────────────────────────

async def fetch_new_messages(
    since_hours: int = 168,  # default: last 7 days
) -> list[dict]:
    """
    Fetch recent messages from SQLite that can be used as new training data.

    Criteria for inclusion:
      - Created within the last `since_hours` hours
      - Not a pure fallback (fallback_used=False OR confidence > 0.5)
      - Have a non-empty intent and user_text

    Args:
        since_hours: How many hours back to look for new messages.

    Returns:
        List of dicts with keys: text, intent, response.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    logger.info("Fetching messages since %s (last %d hours)...", cutoff.isoformat(), since_hours)

    try:
        await create_db_tables()
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Message).where(
                    Message.created_at >= cutoff,
                    Message.intent.isnot(None),
                    Message.intent != "",
                )
            )
            messages = result.scalars().all()
    except Exception as exc:
        logger.error("Failed to fetch messages from DB: %s", exc)
        return []

    new_samples: list[dict] = []
    for msg in messages:
        # Skip if it's a pure default fallback with zero confidence
        if msg.fallback_used and (msg.confidence is None or msg.confidence < 0.1):
            continue
        # Skip very short user texts (unlikely to be useful training examples)
        if len(msg.user_text.strip()) < 5:
            continue

        new_samples.append({
            "text": msg.user_text.strip(),
            "intent": msg.intent,
            "response": msg.bot_response.strip(),
        })

    logger.info("Found %d candidate new samples from DB.", len(new_samples))
    return new_samples


# ─── CSV Appending ────────────────────────────────────────────────────────────

def append_to_csv(new_samples: list[dict], csv_path: Path) -> int:
    """
    Append new training samples to the existing CSV file.

    Deduplicates against existing texts to avoid double-training.

    Args:
        new_samples: List of new sample dicts with text/intent/response.
        csv_path: Path to the training CSV file.

    Returns:
        Number of rows actually appended (after deduplication).
    """
    if not new_samples:
        return 0

    # Load existing texts for deduplication
    existing_texts: set[str] = set()
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_texts.add(row.get("text", "").strip().lower())
        except Exception as exc:
            logger.warning("Could not read existing CSV for dedup: %s", exc)

    # Filter out duplicates
    to_append = [
        s for s in new_samples
        if s["text"].strip().lower() not in existing_texts
    ]

    if not to_append:
        logger.info("No new unique samples to append (all are duplicates).")
        return 0

    # Append to CSV
    file_exists = csv_path.exists()
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "intent", "response"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(to_append)
        logger.info("Appended %d new rows to %s.", len(to_append), csv_path)
    except Exception as exc:
        logger.error("Failed to append to CSV: %s", exc)
        return 0

    return len(to_append)


# ─── Retraining Trigger ───────────────────────────────────────────────────────

def run_training_script(reset_chroma: bool = False) -> bool:
    """
    Run the models/train.py script as a subprocess.

    Args:
        reset_chroma: If True, pass --reset-chroma flag to the training script.

    Returns:
        True if training completed successfully, False otherwise.
    """
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--csv", str(TRAINING_CSV)]
    if reset_chroma:
        cmd.append("--reset-chroma")

    logger.info("Running training script: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout
        )
        if result.returncode == 0:
            logger.info("Training completed successfully.")
            if result.stdout:
                logger.info("Training output:\n%s", result.stdout[-2000:])
            return True
        else:
            logger.error(
                "Training failed (rc=%d):\n%s",
                result.returncode, result.stderr[-2000:]
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error("Training script timed out after 600 seconds.")
        return False
    except Exception as exc:
        logger.error("Unexpected error running training script: %s", exc)
        return False


# ─── Retrain Log ──────────────────────────────────────────────────────────────

def write_retrain_log(
    new_sample_count: int,
    appended_count: int,
    training_success: bool,
    dry_run: bool,
) -> None:
    """
    Write a JSONL log entry for this retraining run.

    Args:
        new_sample_count: Total new samples fetched from DB.
        appended_count: Samples actually appended to the CSV.
        training_success: Whether training completed without error.
        dry_run: Whether this was a dry run (no actual training).
    """
    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "new_samples_found": new_sample_count,
        "samples_appended": appended_count,
        "training_triggered": not dry_run and appended_count > 0,
        "training_success": training_success,
        "dry_run": dry_run,
    }
    try:
        with RETRAIN_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("Retrain log written to: %s", RETRAIN_LOG)
    except Exception as exc:
        logger.warning("Could not write retrain log: %s", exc)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    """
    Main async retraining orchestrator.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info("=" * 55)
    logger.info("CURIFY AI ADVISOR — Weekly Retraining Script")
    logger.info("Run time: %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 55)

    # 1. Fetch new samples from DB
    new_samples = await fetch_new_messages(since_hours=args.since_hours)
    new_count = len(new_samples)

    # 2. Check minimum threshold
    min_samples = args.min_samples
    if new_count < min_samples:
        logger.info(
            "Only %d new samples found (minimum: %d). Skipping retrain.",
            new_count, min_samples,
        )
        write_retrain_log(
            new_sample_count=new_count,
            appended_count=0,
            training_success=False,
            dry_run=args.dry_run,
        )
        return

    # 3. Append to CSV
    appended = 0
    if not args.dry_run:
        appended = append_to_csv(new_samples, TRAINING_CSV)
    else:
        logger.info("[DRY RUN] Would append %d samples to CSV.", new_count)
        appended = new_count

    # 4. Trigger retraining
    training_success = False
    if appended > 0 and not args.dry_run:
        training_success = run_training_script(reset_chroma=args.reset_chroma)
    elif args.dry_run:
        logger.info("[DRY RUN] Would trigger models/train.py now.")
        training_success = True

    # 5. Write log
    write_retrain_log(
        new_sample_count=new_count,
        appended_count=appended,
        training_success=training_success,
        dry_run=args.dry_run,
    )

    # 6. Summary
    logger.info("-" * 55)
    logger.info("Retrain summary:")
    logger.info("  New samples found : %d", new_count)
    logger.info("  Samples appended  : %d", appended)
    logger.info("  Training success  : %s", training_success)
    logger.info("  Dry run           : %s", args.dry_run)
    logger.info("=" * 55)


def main() -> None:
    """Parse CLI arguments and run the async orchestrator."""
    parser = argparse.ArgumentParser(
        description="Weekly retraining automation for Curify AI Advisor."
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=settings.min_new_samples_for_retrain,
        help="Minimum new samples required to trigger retraining.",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=168,
        help="How many hours back to look for new messages (default: 168 = 7 days).",
    )
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        default=False,
        help="Clear ChromaDB before re-indexing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate the process without actually writing files or training.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
