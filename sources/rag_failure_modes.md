---
title: RAG And Tool Failure Modes
url: local://rag_failure_modes
---

RAG can fail when chunks are too large, too small, poorly overlapped, or missing important metadata. Retrieval can fail when the query uses different words from the source, when the embedding model is weak, or when top-k is too small. An agent should detect empty retrieval results and ask for better sources or broaden the query.

Citation hallucination happens when an answer includes a citation that was not actually retrieved. A safer design creates citations only from retrieved chunks. The answer writer should not invent URLs, titles, or quote locations.

Tool calls can fail because of bad arguments, timeouts, permission errors, network errors, and empty results. An agent loop should catch errors, return structured tool outputs, and let the model decide whether to retry, use another tool, or stop with a clear limitation.

Repeated tool calls are another common failure. Production agents use maximum step counts, duplicate-call detection, budgets, and tool-specific retry rules. A research assistant should log steps so the developer can inspect what happened.
