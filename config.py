from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SOURCES_DIR = PROJECT_DIR / "sources"
MEMORY_DIR = PROJECT_DIR / "memory"
DATA_DIR = PROJECT_DIR / "data"
DATABASE_PATH = DATA_DIR / "research.db"
LONG_TERM_MEMORY_PATH = MEMORY_DIR / "long_term_memory.jsonl"
SESSION_MEMORY_PATH = MEMORY_DIR / "session_memory.json"

CHUNK_WORDS = 90
TOP_K = 5
MIN_SCORE = 0.08
TOOL_TIMEOUT_SECONDS = 5
