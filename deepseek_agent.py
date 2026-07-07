from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

from memory import ShortTermContext, load_session_memory, recall_related_memory, remember, save_session_turn
from research_assistant import call_tool_with_guards, validate_citations


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


SYSTEM_PROMPT = """
You are a careful Stage 2 research assistant.

Use only the provided evidence and memory. Do not invent citations, URLs, titles,
or chunk IDs. Every factual claim should be supported by a citation like [1].
If the evidence is weak or empty, say so clearly.

Return only valid JSON with these fields:
- answer: concise cited answer
- used_citations: list of citation IDs used in the answer, such as ["[1]"]
- limitations: list of important evidence gaps
- confidence: number from 0 to 1
""".strip()


def load_dotenv_if_present() -> None:
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("Missing Python package: openai. Run: pip install -r requirements.txt")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Set it in your terminal or .env file.")
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def build_evidence_prompt(topic: str, retrieved: list[dict[str, Any]], memories: list[dict[str, Any]]) -> str:
    citations = []
    for index, item in enumerate(retrieved, start=1):
        citations.append(
            {
                "id": f"[{index}]",
                "title": item["title"],
                "url": item["url"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "text": item["text"],
            }
        )

    memory_summaries = [
        {
            "topic": item["memory"].get("topic"),
            "summary": item["memory"].get("summary"),
            "citations": item["memory"].get("citations", []),
        }
        for item in memories
    ]

    return json.dumps(
        {
            "task": "Answer the research topic using only the evidence below.",
            "topic": topic,
            "evidence": citations,
            "related_memory": memory_summaries,
            "rules": [
                "Use citation IDs exactly as provided, for example [1].",
                "Do not cite memory unless the same claim is supported by evidence.",
                "Do not invent new citation IDs.",
                "If evidence is insufficient, explain the limitation.",
            ],
            "required_json_shape": {
                "answer": "string",
                "used_citations": ["[1]"],
                "limitations": ["string"],
                "confidence": 0.0,
            },
        },
        ensure_ascii=False,
    )


def call_deepseek_json(client: Any, topic: str, retrieved: list[dict[str, Any]], memories: list[dict[str, Any]]) -> dict[str, Any]:
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_evidence_prompt(topic, retrieved, memories)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        timeout=30,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            "answer": content,
            "used_citations": [],
            "limitations": [f"Model returned invalid JSON: {exc}"],
            "confidence": 0.2,
        }


def normalize_model_report(topic: str, model_report: dict[str, Any], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    citations = []
    for index, item in enumerate(retrieved, start=1):
        citations.append(
            {
                "id": f"[{index}]",
                "title": item["title"],
                "url": item["url"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
            }
        )

    return {
        "topic": topic,
        "answer": str(model_report.get("answer", "")).strip(),
        "citations": citations,
        "used_citations": model_report.get("used_citations", []),
        "limitations": model_report.get("limitations", []),
        "confidence": float(model_report.get("confidence", 0.0) or 0.0),
    }


def run_deepseek_research(topic: str, save_memory: bool = True) -> dict[str, Any]:
    load_dotenv_if_present()
    context = ShortTermContext()
    session = load_session_memory()
    context.add_step(f"Loaded session memory with {len(session.get('turns', []))} turns.")

    search_result = call_tool_with_guards("search_sources", {"query": topic, "top_k": 5}, context)
    retrieved = search_result.result["matches"] if search_result.ok else []
    context.retrieved_chunks = retrieved
    context.add_step(f"Retrieved {len(retrieved)} chunks for DeepSeek evidence prompt.")

    memories = recall_related_memory(topic)
    context.add_step(f"Recalled {len(memories)} related long-term memories.")

    if not retrieved:
        report = {
            "topic": topic,
            "answer": "I could not find enough local evidence to ask DeepSeek for a cited answer.",
            "citations": [],
            "used_citations": [],
            "limitations": ["No retrieved chunks from local sources."],
            "confidence": 0.2,
        }
    else:
        client = make_client()
        context.add_step(f"Calling DeepSeek model {os.getenv('DEEPSEEK_MODEL', DEFAULT_DEEPSEEK_MODEL)}.")
        model_report = call_deepseek_json(client, topic, retrieved, memories)
        report = normalize_model_report(topic, model_report, retrieved)

    report["session_memory"] = {
        "last_topic": session.get("last_topic"),
        "turn_count": len(session.get("turns", [])),
    }
    report["memory_used"] = [item["memory"] for item in memories]
    report["tool_calls"] = context.tool_calls
    report["errors"] = context.errors
    report["citation_validation"] = validate_citations(report, retrieved)
    report["steps"] = context.steps

    if save_memory:
        remember(topic, report)
        save_session_turn(topic, report)
        report["steps"].append("Saved this DeepSeek research summary to session and long-term memory.")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 DeepSeek RAG assistant.")
    parser.add_argument("topic", nargs="*", help="Research topic, for example: memory for RAG assistants")
    parser.add_argument("--no-memory", action="store_true", help="Do not save session or long-term memory.")
    args = parser.parse_args()

    topic = " ".join(args.topic).strip()
    if not topic:
        print('Usage: python deepseek_agent.py "memory for RAG assistants"')
        return 1

    try:
        report = run_deepseek_research(topic, save_memory=not args.no_memory)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "hint": "Check DEEPSEEK_API_KEY, DEEPSEEK_MODEL, network access, and pip install -r requirements.txt.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
