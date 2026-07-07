from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import PROJECT_DIR


REPORTS_DIR = PROJECT_DIR / "reports"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "research-report"


def render_markdown_report(report: dict[str, Any]) -> str:
    topic = report.get("topic", "Untitled topic")
    confidence = report.get("confidence", 0)
    validation = report.get("citation_validation", {})
    citations = report.get("citations", [])
    limitations = report.get("limitations", [])
    errors = report.get("errors", [])
    tool_calls = report.get("tool_calls", [])
    steps = report.get("steps", [])

    lines = [
        f"# Research Report: {topic}",
        "",
        f"- Confidence: {confidence}",
        f"- Citation validation: {'passed' if validation.get('ok') else 'needs review'}",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        report.get("answer", "").strip() or "No answer was generated.",
        "",
        "## Sources",
        "",
    ]

    if citations:
        for citation in citations:
            lines.append(
                f"- {citation['id']} [{citation['title']}]({citation['url']}) "
                f"`{citation['chunk_id']}` score={citation['score']}"
            )
    else:
        lines.append("- No citations were produced.")

    lines.extend(["", "## Limitations", ""])
    if limitations:
        for limitation in limitations:
            lines.append(f"- {limitation}")
    else:
        lines.append("- No explicit limitations were reported.")

    if errors:
        lines.extend(["", "## Tool Errors", ""])
        for error in errors:
            lines.append(f"- {error}")

    lines.extend(["", "## Tool Calls", ""])
    if tool_calls:
        for index, call in enumerate(tool_calls, start=1):
            result = call.get("result", {})
            status = "ok" if result.get("ok") else "failed"
            lines.append(f"{index}. `{call.get('name')}` {status}")
            lines.append(f"   - arguments: `{call.get('arguments')}`")
            if result.get("error"):
                lines.append(f"   - error: {result['error']}")
    else:
        lines.append("- No tool calls were recorded.")

    lines.extend(["", "## Execution Steps", ""])
    for step in steps:
        lines.append(f"- {step}")

    return "\n".join(lines).rstrip() + "\n"


def save_markdown_report(report: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}-{slugify(report.get('topic', 'research-report'))}.md"
    path = REPORTS_DIR / filename
    path.write_text(render_markdown_report(report), encoding="utf-8")
    return path
