---
title: LangChain, LlamaIndex, And Stage 3 Learning Path
url: local://framework_decision_path
---

LangChain and LlamaIndex overlap, but their main focus is different. LangChain is strongest as a general agent and tool orchestration framework. It helps connect models, tools, prompts, routers, memory, and agent loops. LlamaIndex is strongest as a data-centric RAG framework. It helps load documents, parse them into nodes, build indexes, retrieve evidence, and expose query engines over private data.

For a Stage 2 research assistant, LangChain can decide what action to take, while LlamaIndex can provide better retrieval inside a tool. A practical architecture is: user input goes to a router, research requests go to a LangChain agent, the agent calls a search tool, and that search tool uses LlamaIndex to retrieve cited nodes from the local knowledge base. The program then validates citations and saves memory.

The hand-written DeepSeek tool agent is best for learning the raw tool call loop. The LangChain agent is best for learning how frameworks package model-tool orchestration. The LlamaIndex RAG starter is best for learning how frameworks package document loading, chunking, indexing, and retrieval. These three versions should be compared rather than treated as competing replacements.

Before moving fully to Stage 3, it is useful to build one integrated Stage 2 version: LangChain handles the agent loop and model router, while LlamaIndex handles retrieval. This connects tool use, RAG, citations, memory, and framework abstractions in one small project. After that, Stage 3 with LangGraph will feel natural because LangGraph focuses on explicit state, nodes, edges, persistence, and controllable workflows.

Stage 3 should begin when the Stage 2 flow is clear: the learner can explain how a model chooses a tool, how the program executes the tool, how retrieved evidence becomes citations, how memory is separated, and how failures such as empty results, duplicate calls, timeouts, and hallucinated citations are handled.

LangGraph is appropriate when the agent process needs explicit control: routing node, retrieval node, tool node, critique node, report node, memory node, and human approval node. It is less about better retrieval and more about making agent state and workflow transitions visible, testable, and resumable.
