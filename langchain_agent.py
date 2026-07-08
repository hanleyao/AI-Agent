from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

try:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_openai import ChatOpenAI
except (ImportError, ModuleNotFoundError) as exc:
    create_agent = None
    tool = None
    ChatOpenAI = None
    LANGCHAIN_IMPORT_ERROR = exc
else:
    LANGCHAIN_IMPORT_ERROR = None

from memory import ShortTermContext, load_session_memory, recall_related_memory, remember, save_session_turn
from reporting import render_markdown_report, save_markdown_report
from research_assistant import call_tool_with_guards, validate_citations


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


SYSTEM_PROMPT = """
You are a Stage 2 research assistant built with LangChain.

Use tools to gather evidence before answering research questions. Prefer
search_sources first, then read_source_file or query_database if more detail is
needed. Use browse_url only when a URL preview is useful, and run_python only for
small calculations.

Return only valid JSON:
{
  "answer": "concise answer with citations like [1]",
  "used_citations": ["[1]"],
  "limitations": ["evidence gap or tool failure"],
  "confidence": 0.0
}

Do not invent citations, URLs, titles, or chunk IDs. Citation IDs come from the
search_sources tool result: first match is [1], second match is [2], and so on.
""".strip()


CHAT_SYSTEM_PROMPT = """
You are a concise, friendly assistant in an interactive Stage 2 LangChain
research tool. Answer normal conversation directly. Do not claim to have
searched unless the agent explicitly used research tools.
""".strip()


ROUTER_SYSTEM_PROMPT = """
You route user messages for a Stage 2 research assistant.

Return only valid JSON:
{
  "action": "chat" | "research" | "exit",
  "topic": "clean research topic or empty string",
  "format": "json" | "markdown",
  "save_report": false,
  "reason": "short reason"
}

Rules:
- Use "exit" only for clear quit requests such as exit, quit, bye, or q.
- Use "research" when the user asks to study, research, compare, analyze,
  summarize with sources, find citations, create a report, or asks about RAG,
  agents, tools, memory, LangChain, DeepSeek, open-source projects, papers, or
  technical learning topics that benefit from evidence.
- Use "chat" for greetings, casual questions, or direct explanations that do
  not require source retrieval.
- Set format to "markdown" when the user asks for markdown, report, article,
  note, or document-style output. Otherwise use "json".
- Set save_report to true when the user asks to save, export, write to file, or
  generate a saved report.
- The topic should remove command words such as save, export, markdown, report,
  please, and research, but keep the actual subject.
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


def make_langchain_model() -> Any:
    if ChatOpenAI is None:
        raise RuntimeError(
            "LangChain could not be imported. "
            "Use a clean Stage 2 virtual environment and run pip install -r requirements.txt. "
            f"Original import error: {type(LANGCHAIN_IMPORT_ERROR).__name__}: {LANGCHAIN_IMPORT_ERROR}"
        )
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Set it in your terminal or .env file.")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
        temperature=0.2,
        timeout=30,
    )


def chat_completion(model: Any, user_message: str) -> str:
    response = model.invoke(
        [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    content = getattr(response, "content", "")
    return content if isinstance(content, str) else str(content)


def route_with_model(model: Any, user_message: str) -> dict[str, Any]:
    response = model.invoke(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    content = getattr(response, "content", "")
    if not isinstance(content, str):
        content = str(content)

    route = parse_router_json(content)
    return normalize_route(route, user_message)


def parse_router_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(extract_json_text(content))
    except json.JSONDecodeError:
        return {}


def normalize_route(route: dict[str, Any], user_message: str) -> dict[str, Any]:
    action = route.get("action")
    if action not in {"chat", "research", "exit"}:
        return classify_chat_intent(user_message)

    if action == "exit":
        return {"action": "exit"}

    if action == "chat":
        return {
            "action": "chat",
            "message": user_message.strip(),
            "router_reason": str(route.get("reason", "")).strip(),
        }

    output_format = route.get("format")
    if output_format not in {"json", "markdown"}:
        output_format = "json"

    topic = str(route.get("topic", "")).strip() or clean_research_topic(user_message)
    return {
        "action": "research",
        "format": output_format,
        "save_report": bool(route.get("save_report", False)),
        "topic": topic,
        "router_reason": str(route.get("reason", "")).strip(),
    }


def add_citation_ids(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen_chunk_ids = set()
    for match in matches:
        chunk_id = match.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        item = dict(match)
        item["citation_id"] = f"[{len(deduped) + 1}]"
        deduped.append(item)
    return deduped


def build_langchain_tools(context: ShortTermContext) -> list[Callable[..., str]]:
    if tool is None:
        raise RuntimeError("Missing LangChain packages. Run: pip install -r requirements.txt")

    @tool
    def search_sources(query: str, top_k: int = 5) -> str:
        """Search local RAG chunks from sources/*.md and return cited evidence."""
        result = call_tool_with_guards("search_sources", {"query": query, "top_k": top_k}, context)
        payload = result.to_dict()
        if payload.get("ok"):
            matches = add_citation_ids(payload.get("result", {}).get("matches", []))
            payload["result"]["matches"] = matches
            context.retrieved_chunks = matches
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def read_source_file(filename: str) -> str:
        """Read a full markdown source file from sources/ by filename."""
        result = call_tool_with_guards("read_source_file", {"filename": filename}, context)
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

    @tool
    def run_python(code: str) -> str:
        """Run restricted Python for small calculations."""
        result = call_tool_with_guards("run_python", {"code": code}, context)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    return [search_sources, read_source_file, query_database, browse_url, run_python]


def extract_final_content(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    latest = messages[-1]
    content = getattr(latest, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(latest, dict):
        return latest.get("content", "")
    return str(content or "")


def parse_model_json(content: str) -> dict[str, Any]:
    content = extract_json_text(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "answer": content,
            "used_citations": [],
            "limitations": ["LangChain agent returned non-JSON final content."],
            "confidence": 0.25,
        }


def extract_json_text(content: str) -> str:
    stripped = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        return object_match.group(0).strip()

    return stripped


def citations_from_retrieved(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for index, item in enumerate(retrieved, start=1):
        citations.append(
            {
                "id": item.get("citation_id", f"[{index}]"),
                "title": item["title"],
                "url": item["url"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
            }
        )
    return citations


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
    }
    report["citation_validation"] = validate_citations(report, context.retrieved_chunks)
    if context.errors and not report["limitations"]:
        report["limitations"] = context.errors
    return report


def run_langchain_agent(topic: str, save_memory: bool = True) -> dict[str, Any]:
    load_dotenv_if_present()
    if create_agent is None:
        raise RuntimeError(
            "LangChain could not be imported. "
            "Use a clean Stage 2 virtual environment and run pip install -r requirements.txt. "
            f"Original import error: {type(LANGCHAIN_IMPORT_ERROR).__name__}: {LANGCHAIN_IMPORT_ERROR}"
        )

    context = ShortTermContext()
    session = load_session_memory()
    memories = recall_related_memory(topic)
    context.add_step(f"Loaded session memory with {len(session.get('turns', []))} turns.")
    context.add_step(f"Recalled {len(memories)} related long-term memories.")

    model = make_langchain_model()
    agent = create_agent(
        model=model,
        tools=build_langchain_tools(context),
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
    context.add_step("Invoking LangChain create_agent.")
    result = agent.invoke({"messages": [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}]})
    context.add_step("LangChain agent returned final state.")

    model_report = parse_model_json(extract_final_content(result))
    report = normalize_report(topic, model_report, context, memories, session)

    if save_memory:
        remember(topic, report)
        save_session_turn(topic, report)
        report["steps"].append("Saved this LangChain research summary to session and long-term memory.")

    return report


def classify_chat_intent(user_message: str) -> dict[str, Any]:
    text = user_message.strip()
    lowered = text.lower()

    if lowered in {"exit", "quit", "q", "bye"}:
        return {"action": "exit"}

    wants_report = any(keyword in lowered for keyword in ["report", "markdown", "md"])
    wants_save = any(keyword in lowered for keyword in ["save", "export", "write report"])
    wants_research = wants_report or any(
        keyword in lowered
        for keyword in [
            "research",
            "study",
            "compare",
            "analyze",
            "summarize",
            "sources",
            "citation",
            "citations",
            "rag",
            "agent",
            "memory",
            "stage",
            "tool",
            "langchain",
            "deepseek",
            "open source",
            "opensource",
        ]
    )

    if wants_research:
        return {
            "action": "research",
            "format": "markdown" if wants_report else "json",
            "save_report": wants_save,
            "topic": clean_research_topic(text),
        }

    return {"action": "chat", "message": text}


def clean_research_topic(text: str) -> str:
    cleanup_phrases = [
        "markdown",
        "Markdown",
        "report",
        "save",
        "export",
        "please",
    ]
    topic = text
    for phrase in cleanup_phrases:
        topic = topic.replace(phrase, " ")
    topic = " ".join(topic.split())
    return topic or text


def print_chat_help() -> None:
    print("Interactive LangChain research assistant. Type 'exit' to quit.")
    print("Examples:")
    print("  research Stage 2 RAG memory open source projects")
    print("  save report: Stage 2 RAG memory open source projects")
    print("  markdown summary for RAG failure modes")
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
            report = run_langchain_agent(topic, save_memory=True)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="LangChain create_agent version of the Stage 2 research assistant.")
    parser.add_argument("topic", nargs="*", help="Research topic, for example: memory for RAG assistants")
    parser.add_argument("--no-memory", action="store_true", help="Do not save session or long-term memory.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format.")
    parser.add_argument("--save-report", action="store_true", help="Save a Markdown report under reports/.")
    parser.add_argument("--chat", action="store_true", help="Start an interactive assistant that routes chat vs research/report requests.")
    args = parser.parse_args()

    if args.chat:
        return run_chat_loop()

    topic = " ".join(args.topic).strip()
    if not topic:
        print('Usage: python langchain_agent.py "memory for RAG assistants"')
        return 1

    try:
        report = run_langchain_agent(topic, save_memory=not args.no_memory)
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
