from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from config import TOOL_TIMEOUT_SECONDS
from memory import ShortTermContext, load_session_memory, recall_related_memory, remember, save_session_turn
from rag import first_relevant_sentence
from tools import TOOL_SCHEMAS, ToolResult, safe_tool_call


def call_tool_with_guards(name: str, arguments: dict[str, Any], context: ShortTermContext) -> ToolResult:
    call_key = json.dumps({"name": name, "arguments": arguments}, sort_keys=True, ensure_ascii=False)
    if call_key in context.seen_tool_calls:
        cached = context.seen_tool_calls[call_key]
        context.add_step(f"Skipped duplicate tool call: {name}({arguments}).")
        return ToolResult(ok=True, tool=name, result=cached["result"], error=None)

    context.add_step(f"Calling tool {name} with {arguments}.")
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(safe_tool_call, name, arguments)
        result = future.result(timeout=TOOL_TIMEOUT_SECONDS)
    except TimeoutError:
        result = ToolResult(ok=False, tool=name, error=f"Tool timed out after {TOOL_TIMEOUT_SECONDS}s")
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown(wait=True)

    result_dict = result.to_dict()
    context.tool_calls.append({"name": name, "arguments": arguments, "result": result_dict})
    context.seen_tool_calls[call_key] = result_dict

    if not result.ok:
        context.errors.append(f"{name}: {result.error}")
        context.add_step(f"Tool {name} failed: {result.error}")
    else:
        context.add_step(f"Tool {name} succeeded.")

    return result


def answer_with_citations(topic: str, retrieved: list[dict[str, Any]], memories: list[dict[str, Any]]) -> dict[str, Any]:
    if not retrieved:
        return {
            "topic": topic,
            "answer": "I could not find enough evidence in the available sources. Add sources, broaden the topic, or try a different query.",
            "citations": [],
            "memory_used": [item["memory"] for item in memories],
            "confidence": 0.2,
        }

    claims = []
    citations = []
    seen_sources = set()

    for item in retrieved:
        citation_id = f"[{len(citations) + 1}]"
        sentence = first_relevant_sentence(topic, item["text"]) or item["text"][:220]
        claims.append(f"{sentence} {citation_id}")
        citations.append(
            {
                "id": citation_id,
                "title": item["title"],
                "url": item["url"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
            }
        )
        seen_sources.add(item["source_id"])

    memory_note = ""
    if memories:
        topics = [item["memory"].get("topic", "") for item in memories]
        memory_note = f" Related prior memory: {', '.join(topic for topic in topics if topic)}."

    answer = f"Research summary for '{topic}': " + " ".join(claims) + memory_note
    confidence = min(0.9, 0.45 + 0.1 * len(retrieved) + 0.05 * len(seen_sources))
    return {
        "topic": topic,
        "answer": answer,
        "citations": citations,
        "memory_used": [item["memory"] for item in memories],
        "confidence": round(confidence, 2),
    }


def validate_citations(report: dict[str, Any], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    citation_ids = {citation["id"] for citation in report.get("citations", [])}
    retrieved_chunk_ids = {item["chunk_id"] for item in retrieved}
    answer_ids = set(re.findall(r"\[\d+\]", report.get("answer", "")))

    missing_from_citations = sorted(answer_ids - citation_ids)
    unsupported_chunks = [
        citation["chunk_id"]
        for citation in report.get("citations", [])
        if citation.get("chunk_id") not in retrieved_chunk_ids
    ]

    validation = {
        "ok": not missing_from_citations and not unsupported_chunks,
        "missing_from_citations": missing_from_citations,
        "unsupported_chunks": unsupported_chunks,
    }
    if not validation["ok"]:
        report["confidence"] = min(report.get("confidence", 0.0), 0.3)
    return validation


def run_research(topic: str, save_memory: bool = True, use_session: bool = True) -> dict[str, Any]:
    context = ShortTermContext()
    session = load_session_memory() if use_session else {"turns": [], "last_topic": None, "last_citations": []}
    context.add_step(f"Loaded session memory with {len(session.get('turns', []))} turns.")

    search_result = call_tool_with_guards("search_sources", {"query": topic, "top_k": 5}, context)
    retrieved = []
    if search_result.ok:
        retrieved = search_result.result["matches"]
        context.retrieved_chunks = retrieved
        context.add_step(f"Retrieved {len(retrieved)} chunks from local sources.")
    else:
        context.add_step("Continuing without source search results.")

    memories = recall_related_memory(topic)
    context.add_step(f"Recalled {len(memories)} related long-term memories.")

    db_result = call_tool_with_guards(
        "query_database",
        {
            "sql": (
                "SELECT title, category, content, url FROM project_notes "
                "WHERE content LIKE '%research%' OR content LIKE '%RAG%' OR content LIKE '%memory%'"
            )
        },
        context,
    )
    database_rows = db_result.result["rows"] if db_result.ok else []

    if retrieved:
        top_url = retrieved[0]["url"]
        top_filename = f"{retrieved[0]['source_id']}.md"
        call_tool_with_guards("read_source_file", {"filename": top_filename}, context)
        call_tool_with_guards("browse_url", {"url": top_url}, context)

        scores = [item["score"] for item in retrieved]
        score_code = f"scores = {scores!r}\nprint(round(sum(scores) / len(scores), 4))"
        call_tool_with_guards("run_python", {"code": score_code}, context)

    report = answer_with_citations(topic, retrieved, memories)
    report["session_memory"] = {
        "last_topic": session.get("last_topic"),
        "turn_count": len(session.get("turns", [])),
    }
    report["database_evidence"] = database_rows
    report["tool_calls"] = context.tool_calls
    report["errors"] = context.errors
    report["citation_validation"] = validate_citations(report, retrieved)
    report["steps"] = context.steps

    if save_memory:
        remember(topic, report)
        save_session_turn(topic, report)
        report["steps"].append("Saved this research summary to session and long-term memory.")

    return report


def run_self_check() -> dict[str, Any]:
    context = ShortTermContext()
    first = call_tool_with_guards("search_sources", {"query": "agent memory", "top_k": 1}, context)
    second = call_tool_with_guards("search_sources", {"query": "agent memory", "top_k": 1}, context)
    bad_sql = call_tool_with_guards("query_database", {"sql": "DROP TABLE project_notes"}, context)
    empty_search = call_tool_with_guards("search_sources", {"query": "zzzxqv nonexistent topic", "top_k": 3}, context)
    fake_report = {
        "answer": "This claim has a fake citation [99].",
        "citations": [],
        "confidence": 0.9,
    }
    citation_validation = validate_citations(fake_report, [])

    return {
        "duplicate_call": {
            "first_ok": first.ok,
            "second_ok": second.ok,
            "skipped_duplicate": any("Skipped duplicate tool call" in step for step in context.steps),
        },
        "tool_failure": bad_sql.to_dict(),
        "empty_result_count": len(empty_search.result["matches"]) if empty_search.ok else None,
        "fake_citation_validation": citation_validation,
        "steps": context.steps,
        "errors": context.errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 local research assistant with tools, RAG, and memory.")
    parser.add_argument("topic", nargs="*", help="Research topic, for example: agentic RAG memory")
    parser.add_argument("--no-memory", action="store_true", help="Do not save the result to session or long-term memory.")
    parser.add_argument("--show-tools", action="store_true", help="Print available tool schemas and exit.")
    parser.add_argument("--self-check", action="store_true", help="Run guardrail checks for duplicate calls, failures, empty results, and citations.")
    args = parser.parse_args()

    if args.show_tools:
        print(json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2))
        return 0

    if args.self_check:
        print(json.dumps(run_self_check(), ensure_ascii=False, indent=2))
        return 0

    topic = " ".join(args.topic).strip()
    if not topic:
        print('Usage: python research_assistant.py "agentic RAG memory"')
        return 1

    report = run_research(topic, save_memory=not args.no_memory, use_session=not args.no_memory)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
