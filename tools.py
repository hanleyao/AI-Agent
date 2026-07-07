from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import sqlite3
import statistics
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from config import DATABASE_PATH, DATA_DIR, SOURCES_DIR
from rag import build_index, chunk_to_dict, load_markdown_sources, parse_front_matter, retrieve


@dataclass
class ToolResult:
    ok: bool
    tool: str
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "result": self.result,
            "error": self.error,
        }


TOOL_SCHEMAS = [
    {
        "name": "search_sources",
        "description": "Search local RAG chunks from sources/*.md and return cited evidence.",
        "parameters": {"query": "string", "top_k": "integer, optional"},
    },
    {
        "name": "read_source_file",
        "description": "Read a markdown file from sources/ by filename.",
        "parameters": {"filename": "string"},
    },
    {
        "name": "query_database",
        "description": "Run a read-only SELECT query against the local SQLite research database.",
        "parameters": {"sql": "string"},
    },
    {
        "name": "browse_url",
        "description": "Fetch a simple text preview for http(s) URLs or local://source_id URLs.",
        "parameters": {"url": "string"},
    },
    {
        "name": "run_python",
        "description": "Run a restricted Python calculation snippet with no imports, file access, or loops.",
        "parameters": {"code": "string"},
    },
]


def safe_tool_call(name: str, arguments: dict[str, Any]) -> ToolResult:
    if name not in TOOL_HANDLERS:
        return ToolResult(ok=False, tool=name, error=f"Unknown tool: {name}")
    try:
        return ToolResult(ok=True, tool=name, result=TOOL_HANDLERS[name](**arguments))
    except Exception as exc:
        return ToolResult(ok=False, tool=name, error=f"{type(exc).__name__}: {exc}")


def search_sources(query: str, top_k: int = 5) -> dict[str, Any]:
    documents = load_markdown_sources()
    chunks = build_index(documents)
    matches = retrieve(query, chunks, top_k=top_k)
    deduped_matches = dedupe_retrieved_matches(matches)
    return {
        "query": query,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "matches": [chunk_to_dict(item["chunk"], item["score"]) for item in deduped_matches],
    }


def dedupe_retrieved_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen_chunk_ids = set()
    for item in matches:
        chunk_id = item["chunk"].chunk_id
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        deduped.append(item)
    return deduped


def read_source_file(filename: str) -> dict[str, Any]:
    target = (SOURCES_DIR / filename).resolve()
    allowed_root = SOURCES_DIR.resolve()
    if allowed_root not in target.parents and target != allowed_root:
        raise PermissionError("read_source_file can only read files inside sources/")
    if not target.is_file():
        raise FileNotFoundError(f"No such source file: {filename}")

    raw = target.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw)
    return {
        "filename": filename,
        "metadata": metadata,
        "content": body.strip(),
    }


def query_database(sql: str) -> dict[str, Any]:
    statement = sql.strip().rstrip(";")
    if not statement.lower().startswith("select"):
        raise PermissionError("Only SELECT queries are allowed.")
    if ";" in statement:
        raise PermissionError("Only one SQL statement is allowed.")

    ensure_database()
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(statement).fetchmany(20)
    return {"sql": statement, "rows": [dict(row) for row in rows], "row_count": len(rows)}


def ensure_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS project_notes (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT
            )
            """
        )
        count = connection.execute("SELECT COUNT(*) FROM project_notes").fetchone()[0]
        if count:
            return
        connection.executemany(
            "INSERT INTO project_notes (title, category, content, url) VALUES (?, ?, ?, ?)",
            [
                (
                    "GPT Researcher",
                    "research_assistant",
                    "Reference for topic-driven research, source gathering, and cited report generation.",
                    "https://github.com/assafelovic/gpt-researcher",
                ),
                (
                    "RAGFlow",
                    "rag_system",
                    "Reference for document ingestion, chunking, indexing, retrieval, and citations.",
                    "https://github.com/infiniflow/ragflow",
                ),
                (
                    "mem0",
                    "memory",
                    "Reference for long-term memory in AI agents and assistants.",
                    "https://github.com/mem0ai/mem0",
                ),
                (
                    "LangGraph",
                    "agent_framework",
                    "Reference for explicit agent state, workflows, persistence, and tool orchestration.",
                    "https://github.com/langchain-ai/langgraph",
                ),
            ],
        )


def browse_url(url: str) -> dict[str, Any]:
    if url.startswith("local://"):
        source_id = url.removeprefix("local://")
        filename = f"{source_id}.md"
        return {"url": url, "content": read_source_file(filename)["content"][:1200]}

    if not url.startswith(("http://", "https://")):
        raise ValueError("browse_url only supports http(s):// or local:// URLs.")

    request = urllib.request.Request(url, headers={"User-Agent": "stage2-research-assistant/0.1"})
    with urllib.request.urlopen(request, timeout=5) as response:
        html = response.read(100_000).decode("utf-8", errors="replace")
    parser = TextPreviewParser()
    parser.feed(html)
    return {"url": url, "content": parser.text()[:1200]}


class TextPreviewParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip = False

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def run_python(code: str) -> dict[str, Any]:
    tree = ast.parse(code, mode="exec")
    validate_python_ast(tree)
    stdout = io.StringIO()
    env = {
        "__builtins__": {
            "abs": abs,
            "len": len,
            "max": max,
            "min": min,
            "print": print,
            "round": round,
            "sum": sum,
        },
        "math": math,
        "statistics": statistics,
    }
    with contextlib.redirect_stdout(stdout):
        exec(compile(tree, "<stage2-run-python>", "exec"), env, {})
    return {"stdout": stdout.getvalue().strip(), "code": code}


def validate_python_ast(tree: ast.AST) -> None:
    blocked_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.While,
        ast.For,
        ast.AsyncFor,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Lambda,
        ast.FunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
    )
    blocked_names = {"eval", "exec", "open", "__import__", "input", "compile", "globals", "locals", "vars"}
    for node in ast.walk(tree):
        if isinstance(node, blocked_nodes):
            raise PermissionError(f"Blocked Python syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in blocked_names:
            raise PermissionError(f"Blocked Python name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PermissionError("Dunder attributes are blocked.")


TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "search_sources": search_sources,
    "read_source_file": read_source_file,
    "query_database": query_database,
    "browse_url": browse_url,
    "run_python": run_python,
}
