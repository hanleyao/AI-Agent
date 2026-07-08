# Stage 2 Research Assistant

This is a Stage 2 learning project for tool use, RAG, citations, memory, and a DeepSeek tool-calling agent.

It demonstrates:

- Loading local source documents
- Chunking documents
- Creating simple local text embeddings
- Retrieving relevant chunks
- Answering with citations from retrieved chunks
- Tool use: local search, file reading, SQLite query, URL browsing, restricted Python execution
- Short-term context for one run
- Session memory across recent turns
- Long-term memory saved as JSONL
- Empty-result handling
- Tool failure handling
- Duplicate tool-call detection
- Citation validation

## Setup

```powershell
cd E:\agent\aiagent\ai-ai-agent\outputs\stage2_research_assistant
deactivate
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your DeepSeek API key:

```powershell
$env:DEEPSEEK_API_KEY="your DeepSeek API key"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

`deepseek-v4-flash` is the default low-cost model for this starter. You can override it with `DEEPSEEK_MODEL`.

## Offline Mode

The offline assistant is useful for learning the mechanics without API calls:

```powershell
python research_assistant.py "agentic RAG memory"
python research_assistant.py --show-tools
python research_assistant.py --self-check
```

## DeepSeek RAG-First Mode

This version lets the program retrieve evidence first, then asks DeepSeek to write the final cited answer:

```powershell
python deepseek_agent.py --no-memory "memory for RAG assistants"
```

Flow:

```text
user topic -> search_sources -> evidence prompt -> DeepSeek JSON answer -> citation validation
```

## DeepSeek Tool-Calling Agent Mode

This is the Stage 2 target shape: the model receives tool schemas, chooses tools, the program executes them, and tool results are sent back to the model until it returns final JSON.

```powershell
python deepseek_tool_agent.py --no-memory "which open source projects should I study for Stage 2 RAG memory"
```

Interactive chat mode:

```powershell
python deepseek_tool_agent.py --chat
```

Example chat requests:

```text
hello, explain what you can do
research Stage 2 RAG memory open source projects
markdown summary for RAG failure modes
save report: Stage 2 RAG memory open source projects
exit
```

Markdown output:

```powershell
python deepseek_tool_agent.py --no-memory --format markdown "which open source projects should I study for Stage 2 RAG memory"
```

Save a report under `reports/`:

```powershell
python deepseek_tool_agent.py --save-report "which open source projects should I study for Stage 2 RAG memory"
```

Tool-agent flow:

```text
user topic
-> DeepSeek chooses a tool
-> program executes search_sources/read_source_file/query_database/browse_url/run_python
-> tool result goes back to DeepSeek
-> DeepSeek continues or returns final JSON
-> program validates citations and records memory
```

## LangChain create_agent Mode

This version recreates the same tool-calling research assistant with LangChain's
`create_agent`, while reusing the same local tools, memory, RAG, reporting, and
citation validation code.

```powershell
python langchain_agent.py --no-memory "which open source projects should I study for Stage 2 RAG memory"
```

Interactive chat mode:

```powershell
python langchain_agent.py --chat
```

In chat mode, a small model-router call first classifies each message as
`chat`, `research`, or `exit`. Research requests are then sent to the LangChain
tool-calling agent, while casual messages go to a normal chat completion.

Markdown output:

```powershell
python langchain_agent.py --no-memory --format markdown "which open source projects should I study for Stage 2 RAG memory"
```

Mapping from the hand-written agent to LangChain:

```text
DEEPSEEK_TOOLS            -> @tool wrappers in langchain_agent.py
manual messages/tool loop -> create_agent(...)
tool result messages      -> LangChain ToolMessage handling
tools.py implementations  -> reused directly
validate_citations()      -> still our own guardrail
memory.py                 -> still our own memory layer
```

If Windows blocks `xxhash` while importing LangChain, first make sure you are
using this project's `.venv`, not the Stage 1 environment:

```powershell
where python
python -c "import sys; print(sys.executable)"
```

If it still points to `stage1_openai_agent\.venv`, close the terminal or run
`deactivate`, then activate `stage2_research_assistant\.venv`.

## Output Shape

```json
{
  "topic": "...",
  "answer": "... [1]",
  "citations": [],
  "memory_used": [],
  "session_memory": {},
  "tool_calls": [],
  "errors": [],
  "citation_validation": {},
  "confidence": 0.75,
  "steps": []
}
```

## File Map

```text
config.py              Shared paths and constants
rag.py                 Document loading, chunking, embedding, retrieval
tools.py               Tool schemas and tool implementations
memory.py              Short-term context, session memory, long-term memory
research_assistant.py  Offline orchestration and guardrails
deepseek_agent.py      RAG-first DeepSeek answer generation
deepseek_tool_agent.py DeepSeek chooses tools in an agent loop
langchain_agent.py     LangChain create_agent version using the same tools
reporting.py           Markdown report rendering and saving
sources/               Local documents used as RAG sources
data/research.db       Auto-created SQLite database for the database tool
reports/               Saved Markdown research reports
```

## Guardrails

- Tool failure: `query_database` rejects non-SELECT SQL.
- Empty results: no retrieved chunks means no citations are generated.
- Duplicate calls: repeated tool name + same arguments are skipped.
- Citation validation: every `[n]` in `answer` must exist in `citations`, and every citation must point to a retrieved chunk.

## Open-Source Projects Worth Studying

- GPT Researcher: topic-driven research and cited report generation.
- Open Deep Research / Local Deep Researcher: iterative search and reflection loops.
- STORM: long-form research and report generation.
- RAGFlow, Onyx, AnythingLLM, GraphRAG: production RAG and document ingestion.
- mem0, Letta, Khoj: long-term memory and stateful assistants.
- LangGraph and LlamaIndex: explicit agent state/workflow control and data-centric RAG.
