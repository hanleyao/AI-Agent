---
title: Agent Learning Hub Stage 2 Notes
url: https://github.com/datawhalechina/Agent-Learning-Hub
---

Stage 2 focuses on tool use, retrieval augmented generation, and memory. A practical agent should retrieve evidence before answering knowledge-heavy questions. The core retrieval pipeline is chunk, embed, retrieve, and answer with citations. Chunking splits large documents into smaller units. Embedding converts text into vectors or searchable representations. Retrieval selects the most relevant chunks. The final answer should cite the evidence it used.

Stage 2 also expands tool use beyond toy examples. Search tools help discover candidate sources. File tools read local documents. Database tools query structured data. Browser tools inspect web pages. Code execution tools run calculations or data processing. A useful research assistant connects several tools, handles failures, and reports uncertainty when evidence is weak.

Memory has several layers. Short-term context is the current conversation or current agent run. Session memory persists across turns in the same session. Long-term memory stores durable facts, preferences, or prior research summaries. Good agents separate these memory types so old information does not silently pollute current answers.

Reliable research agents need guardrails against empty results, repeated tool calls, tool errors, and hallucinated citations. A citation should point to a real retrieved source, not a source invented after the answer is written.

MCP help agent to connect external tools and data, is a kind of interface. The tools the agent can use such as database, research, calendar,Notion, Figma,IDE and so on.
