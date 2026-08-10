# tests/conftest.py
"""
Shared pytest configuration and fixtures.

- Sets PYTHONPATH so all test files can import from project root.
- Configures pytest-asyncio mode to 'auto' for async tests.
- Provides a shared .env override so tests never need real API keys.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Override critical env vars so tests never call real APIs or touch prod DB
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("TRAIN_API_KEY", "test_ci_key_12345")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_curify.db")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./test_chroma_db")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("MODEL_DIR", "./models/saved")
os.environ.setdefault("TRAINING_DATA_PATH", "./data/raw/sample_training_data.csv")


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (skipped by default)")
    config.addinivalue_line("markers", "integration: marks tests that require live services")
