from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import LONG_TERM_MEMORY_PATH, MEMORY_DIR, SESSION_MEMORY_PATH
from rag import cosine_similarity, embed_text


@dataclass
class ShortTermContext:
    steps: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seen_tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_step(self, message: str) -> None:
        self.steps.append(message)


def load_session_memory() -> dict[str, Any]:
    if not SESSION_MEMORY_PATH.exists():
        return {"turns": [], "last_topic": None, "last_citations": []}

    try:
        return json.loads(SESSION_MEMORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"turns": [], "last_topic": None, "last_citations": []}


def save_session_turn(topic: str, report: dict[str, Any]) -> dict[str, Any]:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    session = load_session_memory()
    session.setdefault("turns", []).append(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "topic": topic,
            "answer": report.get("answer", ""),
            "citations": report.get("citations", []),
        }
    )
    session["turns"] = session["turns"][-10:]
    session["last_topic"] = topic
    session["last_citations"] = report.get("citations", [])
    SESSION_MEMORY_PATH.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    return session


def load_long_term_memory() -> list[dict[str, Any]]:
    if not LONG_TERM_MEMORY_PATH.exists():
        return []

    memories = []
    for line in LONG_TERM_MEMORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            memories.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return memories


def remember(topic: str, report: dict[str, Any]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memory_item = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "topic": topic,
        "summary": report["answer"],
        "citations": report["citations"],
    }
    with LONG_TERM_MEMORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(memory_item, ensure_ascii=False) + "\n")


def recall_related_memory(topic: str, limit: int = 3) -> list[dict[str, Any]]:
    query_vector = embed_text(topic)
    scored = []
    for memory in load_long_term_memory():
        memory_text = f"{memory.get('topic', '')} {memory.get('summary', '')}"
        score = cosine_similarity(query_vector, embed_text(memory_text))
        if score >= 0.08:
            scored.append({"score": round(score, 4), "memory": memory})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]
