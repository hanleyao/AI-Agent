from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

from memory import ShortTermContext, load_session_memory, recall_related_memory, remember, save_session_turn
from reporting import render_markdown_report, save_markdown_report
from research_assistant import call_tool_with_guards, validate_citations


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
MAX_AGENT_STEPS = 8


SYSTEM_PROMPT = """
You are a Stage 2 research assistant.

Goal:
Given a user topic, automatically gather evidence, filter useful information,
summarize it, and output citation links.

Rules:
- You may call tools to search local sources, read files, query a database,
  browse URLs, and run safe Python calculations.
- For research topics, start by calling search_sources.
- If search_sources returns useful chunks, cite them with IDs [1], [2], etc.
  The first match is [1], the second match is [2], and so on.
- Do not invent citations, URLs, titles, or chunk IDs.
- If a tool fails, continue with the best available evidence and mention the limitation.
- When ready, return only valid JSON:
  {
    "answer": "concise answer with citations like [1]",
    "used_citations": ["[1]"],
    "limitations": ["evidence gap or tool failure"],
    "confidence": 0.0
  }
""".strip()


CHAT_SYSTEM_PROMPT = """
You are a concise, friendly assistant in an interactive Stage 2 research tool.
Answer normal conversation directly. Do not claim to have searched unless the
program explicitly provides research results.
""".strip()


DEEPSEEK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_sources",
            "description": "Search local RAG chunks from sources/*.md and return evidence chunks. Use this first for research topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query or research topic."},
                    "top_k": {"type": "integer", "description": "Maximum chunks to return, usually 3 to 5."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source_file",
            "description": "Read a full markdown file from sources/ after search_sources finds a relevant source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename inside sources/, for example open_source_projects.md."}
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Run a read-only SELECT query against the local SQLite research database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement. Non-SELECT statements are blocked."}
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_url",
            "description": "Fetch a text preview for an http(s) URL or a local://source_id URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to browse, such as local://rag_failure_modes or https://example.com."}
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a restricted Python calculation snippet. Use for simple arithmetic or aggregate score calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Restricted Python code with no imports, file access, functions, or loops."}
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
]


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


def chat_completion(client: Any, user_message: str) -> str:
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        timeout=30,
    )
    return response.choices[0].message.content or ""


def serialize_tool_result(tool_name: str, result: dict[str, Any], context: ShortTermContext) -> str:
    if result.get("ok") and tool_name == "search_sources":
        matches = dedupe_matches(result.get("result", {}).get("matches", []))
        result["result"]["matches"] = matches
        for index, match in enumerate(matches, start=1):
            match["citation_id"] = f"[{index}]"
        context.retrieved_chunks = matches
    return json.dumps(result, ensure_ascii=False)


def dedupe_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen_chunk_ids = set()
    for match in matches:
        chunk_id = match.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        deduped.append(match)
    return deduped


def parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}


def build_initial_messages(topic: str, memories: list[dict[str, Any]], session: dict[str, Any]) -> list[dict[str, Any]]:
    memory_summaries = [
        {
            "topic": item["memory"].get("topic"),
            "summary": item["memory"].get("summary"),
        }
        for item in memories
    ]
    user_payload = {
        "topic": topic,
        "session_memory": {
            "last_topic": session.get("last_topic"),
            "turn_count": len(session.get("turns", [])),
        },
        "related_long_term_memory": memory_summaries,
        "instruction": "Research the topic with tools, then produce the required JSON answer with citation IDs.",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def extract_final_json(content: str | None) -> dict[str, Any]:
    if not content:
        return {
            "answer": "",
            "used_citations": [],
            "limitations": ["Model returned empty content."],
            "confidence": 0.0,
        }
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "answer": content,
            "used_citations": [],
            "limitations": ["Model returned non-JSON final content."],
            "confidence": 0.25,
        }


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


def normalize_final_report(
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


def run_deepseek_tool_agent(topic: str, save_memory: bool = True) -> dict[str, Any]:
    load_dotenv_if_present()
    client = make_client()
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)

    context = ShortTermContext()
    session = load_session_memory()
    memories = recall_related_memory(topic)
    context.add_step(f"Loaded session memory with {len(session.get('turns', []))} turns.")
    context.add_step(f"Recalled {len(memories)} related long-term memories.")

    messages = build_initial_messages(topic, memories, session)
    model_report: dict[str, Any] | None = None

    for step in range(1, MAX_AGENT_STEPS + 1):
        context.add_step(f"Calling DeepSeek model step {step}/{MAX_AGENT_STEPS}.")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=DEEPSEEK_TOOLS,
            tool_choice="auto",
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=30,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ]
        messages.append(assistant_message)

        if not tool_calls:
            model_report = extract_final_json(message.content)
            context.add_step("DeepSeek returned final JSON.")
            break

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = parse_tool_arguments(tool_call.function.arguments)
            result = call_tool_with_guards(tool_name, arguments, context)
            result_text = serialize_tool_result(tool_name, result.to_dict(), context)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

    if model_report is None:
        model_report = {
            "answer": "The agent reached the maximum step limit before producing a final answer.",
            "used_citations": [],
            "limitations": [f"Reached MAX_AGENT_STEPS={MAX_AGENT_STEPS}."],
            "confidence": 0.2,
        }

    report = normalize_final_report(topic, model_report, context, memories, session)

    if save_memory:
        remember(topic, report)
        save_session_turn(topic, report)
        report["steps"].append("Saved this tool-agent research summary to session and long-term memory.")

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
    print("Interactive DeepSeek research assistant. Type 'exit' to quit.")
    print("Examples:")
    print("  research Stage 2 RAG memory open source projects")
    print("  save report: Stage 2 RAG memory open source projects")
    print("  markdown summary for RAG failure modes")
    print("  hello, explain what you can do")


def run_chat_loop() -> int:
    load_dotenv_if_present()
    try:
        client = make_client()
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

        intent = classify_chat_intent(user_message)
        if intent["action"] == "exit":
            print("bye")
            return 0

        if intent["action"] == "chat":
            try:
                print("\nAssistant>")
                print(chat_completion(client, intent["message"]))
            except Exception as exc:
                print(f"Chat error: {type(exc).__name__}: {exc}")
            continue

        topic = intent["topic"]
        print(f"\n[research] topic: {topic}")
        print(f"[research] output: {intent['format']}, save_report={intent['save_report']}")
        try:
            report = run_deepseek_tool_agent(topic, save_memory=True)
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
    parser = argparse.ArgumentParser(description="DeepSeek tool-calling Stage 2 research assistant.")
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
        print('Usage: python deepseek_tool_agent.py "memory for RAG assistants"')
        return 1

    try:
        report = run_deepseek_tool_agent(topic, save_memory=not args.no_memory)
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
