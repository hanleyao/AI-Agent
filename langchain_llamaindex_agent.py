from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Callable

from config import TOOL_TIMEOUT_SECONDS
from config import SOURCES_DIR
from langchain_agent import (
    chat_completion,
    classify_chat_intent,
    create_agent,
    extract_final_content,
    load_dotenv_if_present,
    make_langchain_model,
    parse_model_json,
    route_with_model,
    tool,
)
from llamaindex_rag import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, build_llamaindex_retriever, configure_llamaindex, node_to_match
from memory import ShortTermContext, load_session_memory, recall_related_memory, remember, save_session_turn
from reporting import render_markdown_report, save_markdown_report
from rag import parse_front_matter
from research_assistant import call_tool_with_guards, validate_citations


SYSTEM_PROMPT = """
You are a Stage 2 research assistant that combines LangChain and LlamaIndex.

LangChain controls the agent loop and tool calls. LlamaIndex handles document
loading, node parsing, vector indexing, and retrieval.

Use llamaindex_search_sources first for research questions. Use read_source_file
when a full source file is needed. Use query_database or browse_url only when
extra structured or URL evidence is useful.

Return only valid JSON:
{
  "answer": "concise answer with citations like [1]",
  "used_citations": ["[1]"],
  "limitations": ["evidence gap or tool failure"],
  "confidence": 0.0
}

Do not invent citations, URLs, titles, or chunk IDs. Citation IDs come from the
llamaindex_search_sources tool result.
""".strip()


_LLAMAINDEX_CACHE: dict[int, tuple[Any, list[Any]]] = {}


def get_cached_llamaindex_retriever(top_k: int) -> tuple[Any, list[Any]]:
    if top_k not in _LLAMAINDEX_CACHE:
        configure_llamaindex()
        _LLAMAINDEX_CACHE[top_k] = build_llamaindex_retriever(top_k)
    return _LLAMAINDEX_CACHE[top_k]


def llamaindex_search_sources(
    query: str,
    context: ShortTermContext,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> dict[str, Any]:
    call_key = json.dumps(
        {"name": "llamaindex_search_sources", "arguments": {"query": query, "top_k": top_k, "min_score": min_score}},
        sort_keys=True,
        ensure_ascii=False,
    )
    if call_key in context.seen_tool_calls:
        context.add_step(f"Skipped duplicate LlamaIndex search for {query!r}.")
        return context.seen_tool_calls[call_key]["result"]

    context.add_step(
        f"Calling LlamaIndex retriever with query={query!r}, top_k={top_k}, min_score={min_score}."
    )

    def run_search() -> dict[str, Any]:
        retriever, documents = get_cached_llamaindex_retriever(top_k)
        retrieved_nodes = retriever.retrieve(query)
        filtered_nodes = [item for item in retrieved_nodes if float(item.score or 0.0) >= min_score]
        matches = assign_global_citation_ids(
            context,
            [node_to_match(item, index) for index, item in enumerate(filtered_nodes, start=1)],
        )
        return {
            "query": query,
            "document_count": len(documents),
            "retrieved_count": len(retrieved_nodes),
            "kept_count": len(matches),
            "top_k": top_k,
            "min_score": min_score,
            "index": "LlamaIndex VectorStoreIndex",
            "embedding": "HashEmbedding(local learning stub)",
            "matches": matches,
        }

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        result = executor.submit(run_search).result(timeout=TOOL_TIMEOUT_SECONDS)
    except TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"LlamaIndex retrieval timed out after {TOOL_TIMEOUT_SECONDS}s")
    else:
        executor.shutdown(wait=True)

    result_dict = {"ok": True, "tool": "llamaindex_search_sources", "result": result, "error": None}
    context.tool_calls.append(
        {
            "name": "llamaindex_search_sources",
            "arguments": {"query": query, "top_k": top_k, "min_score": min_score},
            "result": result_dict,
        }
    )
    context.seen_tool_calls[call_key] = result_dict
    context.add_step(f"LlamaIndex search kept {result['kept_count']} of {result['retrieved_count']} retrieved nodes.")
    return result


def assign_global_citation_ids(
    context: ShortTermContext,
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_by_chunk_id = {item["chunk_id"]: item for item in context.retrieved_chunks}
    assigned = []
    for match in matches:
        chunk_id = match["chunk_id"]
        if chunk_id in existing_by_chunk_id:
            assigned.append(existing_by_chunk_id[chunk_id])
            continue

        item = dict(match)
        item["citation_id"] = f"[{len(context.retrieved_chunks) + 1}]"
        context.retrieved_chunks.append(item)
        existing_by_chunk_id[chunk_id] = item
        assigned.append(item)
    return assigned


def build_integrated_tools(context: ShortTermContext) -> list[Callable[..., str]]:
    if tool is None:
        raise RuntimeError("Missing LangChain packages. Run: pip install -r requirements.txt")

    @tool
    def llamaindex_search_sources(query: str, top_k: int = DEFAULT_TOP_K, min_score: float = DEFAULT_MIN_SCORE) -> str:
        """Search local source documents with a LlamaIndex VectorStoreIndex and return cited evidence nodes."""
        try:
            result = globals()["llamaindex_search_sources"](query, context, top_k=top_k, min_score=min_score)
            payload = {"ok": True, "tool": "llamaindex_search_sources", "result": result, "error": None}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            context.errors.append(f"llamaindex_search_sources: {error}")
            context.add_step(f"LlamaIndex search failed: {error}")
            payload = {"ok": False, "tool": "llamaindex_search_sources", "result": None, "error": error}
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def read_source_file(filename: str) -> str:
        """Read a full source file. Accepts filename, source_id, or source title."""
        result = call_tool_with_guards("read_source_file", {"filename": resolve_source_filename(filename)}, context)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    @tool
    def query_database(sql: str) -> str:
        """Run a read-only SELECT query against the local SQLite research database."""
        result = call_tool_with_guards("query_database", {"sql": sql}, context)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    @tool
    def browse_url(url: str) -> str:
        """Fetch a text preview for an http(s) URL or local://source_id URL."""
        result = call_tool_with_guards("browse_url", {"url": url}, context)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    return [llamaindex_search_sources, read_source_file, query_database, browse_url]


def resolve_source_filename(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return candidate

    direct = SOURCES_DIR / candidate
    if direct.is_file():
        return direct.name

    if not candidate.endswith(".md"):
        direct_md = SOURCES_DIR / f"{candidate}.md"
        if direct_md.is_file():
            return direct_md.name

    lowered = candidate.lower()
    for path in SOURCES_DIR.glob("*.md"):
        metadata, _body = parse_front_matter(path.read_text(encoding="utf-8"))
        title = metadata.get("title", "")
        if lowered in {path.name.lower(), path.stem.lower(), title.lower()}:
            return path.name

    return candidate


def citations_from_retrieved(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("citation_id", f"[{index}]"),
            "title": item["title"],
            "url": item["url"],
            "chunk_id": item["chunk_id"],
            "score": item["score"],
        }
        for index, item in enumerate(retrieved, start=1)
    ]


def normalize_report(
    topic: str,
    model_report: dict[str, Any],
    context: ShortTermContext,
    memories: list[dict[str, Any]],
    session: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "topic": topic,
        "answer": str(model_report.get("answer", "")).strip(),
        "citations": citations_from_retrieved(context.retrieved_chunks),
        "used_citations": model_report.get("used_citations", []),
        "limitations": model_report.get("limitations", []),
        "confidence": float(model_report.get("confidence", 0.0) or 0.0),
        "memory_used": [item["memory"] for item in memories],
        "session_memory": {
            "last_topic": session.get("last_topic"),
            "turn_count": len(session.get("turns", [])),
        },
        "tool_calls": context.tool_calls,
        "errors": context.errors,
        "steps": context.steps,
        "frameworks": {
            "agent_loop": "LangChain create_agent",
            "retrieval": "LlamaIndex VectorStoreIndex",
        },
    }
    report["citation_validation"] = validate_citations(report, context.retrieved_chunks)
    if context.errors and not report["limitations"]:
        report["limitations"] = context.errors
    return report


def run_langchain_llamaindex_agent(topic: str, save_memory: bool = True) -> dict[str, Any]:
    load_dotenv_if_present()
    if create_agent is None:
        raise RuntimeError("LangChain could not be imported. Run: pip install -r requirements.txt")

    context = ShortTermContext()
    session = load_session_memory()
    memories = recall_related_memory(topic)
    context.add_step(f"Loaded session memory with {len(session.get('turns', []))} turns.")
    context.add_step(f"Recalled {len(memories)} related long-term memories.")

    model = make_langchain_model()
    agent = create_agent(
        model=model,
        tools=build_integrated_tools(context),
        system_prompt=SYSTEM_PROMPT,
    )

    user_payload = {
        "topic": topic,
        "session_memory": {
            "last_topic": session.get("last_topic"),
            "turn_count": len(session.get("turns", [])),
        },
        "related_long_term_memory": [
            {"topic": item["memory"].get("topic"), "summary": item["memory"].get("summary")}
            for item in memories
        ],
    }
    context.add_step("Invoking LangChain create_agent with LlamaIndex retrieval tool.")
    result = agent.invoke({"messages": [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}]})
    context.add_step("Integrated LangChain + LlamaIndex agent returned final state.")

    model_report = parse_model_json(extract_final_content(result))
    report = normalize_report(topic, model_report, context, memories, session)

    if save_memory:
        remember(topic, report)
        save_session_turn(topic, report)
        report["steps"].append("Saved this integrated research summary to session and long-term memory.")

    return report


def print_chat_help() -> None:
    print("Interactive LangChain + LlamaIndex research assistant. Type 'exit' to quit.")
    print("Examples:")
    print("  research how should I choose chunk size and embeddings for RAG")
    print("  save report: LangChain and LlamaIndex integration before Stage 3")
    print("  markdown summary for Stage 2 to Stage 3 learning path")
    print("  hello, explain what you can do")


def run_chat_loop() -> int:
    load_dotenv_if_present()
    try:
        model = make_langchain_model()
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

    print_chat_help()
    while True:
        try:
            user_message = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        if not user_message:
            continue

        try:
            intent = route_with_model(model, user_message)
        except Exception as exc:
            print(f"Router error, using keyword fallback: {type(exc).__name__}: {exc}")
            intent = classify_chat_intent(user_message)

        if intent["action"] == "exit":
            print("bye")
            return 0

        if intent["action"] == "chat":
            try:
                print("\nAssistant>")
                print(chat_completion(model, intent["message"]))
            except Exception as exc:
                print(f"Chat error: {type(exc).__name__}: {exc}")
            continue

        topic = intent["topic"]
        print(f"\n[research] topic: {topic}")
        print(f"[research] output: {intent['format']}, save_report={intent['save_report']}")
        if intent.get("router_reason"):
            print(f"[router] {intent['router_reason']}")

        try:
            report = run_langchain_llamaindex_agent(topic, save_memory=True)
        except Exception as exc:
            print(f"Research error: {type(exc).__name__}: {exc}")
            continue

        if intent["save_report"]:
            path = save_markdown_report(report)
            report["report_path"] = str(path)

        if intent["format"] == "markdown":
            print("\nAssistant>")
            print(render_markdown_report(report))
            if intent["save_report"]:
                print(f"Saved report: {report['report_path']}")
        else:
            print("\nAssistant>")
            print(report.get("answer", ""))
            if report.get("citations"):
                print("\nSources:")
                for citation in report["citations"]:
                    print(f"- {citation['id']} {citation['title']} {citation['url']}")
            if report.get("limitations"):
                print("\nLimitations:")
                for limitation in report["limitations"]:
                    print(f"- {limitation}")
            if intent["save_report"]:
                print(f"\nSaved report: {report['report_path']}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LangChain agent with LlamaIndex retrieval.")
    parser.add_argument("topic", nargs="*", help="Research topic, for example: memory for RAG assistants")
    parser.add_argument("--no-memory", action="store_true", help="Do not save session or long-term memory.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format.")
    parser.add_argument("--save-report", action="store_true", help="Save a Markdown report under reports/.")
    parser.add_argument("--chat", action="store_true", help="Start an interactive assistant.")
    args = parser.parse_args()

    if args.chat:
        return run_chat_loop()

    topic = " ".join(args.topic).strip()
    if not topic:
        print('Usage: python langchain_llamaindex_agent.py "memory for RAG assistants"')
        return 1

    try:
        report = run_langchain_llamaindex_agent(topic, save_memory=not args.no_memory)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "hint": "Check DEEPSEEK_API_KEY and run: pip install -r requirements.txt",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    if args.save_report:
        path = save_markdown_report(report)
        report["report_path"] = str(path)

    if args.format == "markdown":
        print(render_markdown_report(report))
        if args.save_report:
            print(f"\nSaved report: {report['report_path']}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
