---
title: Open Source Projects To Study For Stage 2
url: https://github.com/datawhalechina/Agent-Learning-Hub
---

GPT Researcher and Open Deep Research are useful references for building a research assistant that searches, filters, synthesizes, and produces cited reports. They are bigger than a Stage 2 toy project, but their product shape is close to the goal: input a topic, gather evidence, and output a structured research summary.

STORM is a useful reference for long-form research and report generation. It shows how a system can collect multiple perspectives, organize evidence, and write an article-like output. It is most useful after the basic retrieve-and-cite loop is clear.

RAGFlow, Onyx, AnythingLLM, and GraphRAG are useful references for retrieval augmented generation. They show production concerns such as ingestion, chunking, indexing, retrieval quality, document management, and citations. They are larger systems, so study their architecture rather than copying them directly.

mem0, Letta, and Khoj are useful references for agent memory. They help distinguish short-term context, session state, long-term memory, and retrieval over previous interactions. A Stage 2 project can start with a simple JSONL memory file before adopting a memory framework.

LangChain, LangGraph, and LlamaIndex are framework references. LangGraph is useful for explicit agent state and workflow control. LlamaIndex is useful for data-centric RAG and agentic retrieval. LangChain gives common abstractions for models, tools, retrievers, and agents.
