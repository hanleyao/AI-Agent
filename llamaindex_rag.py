from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import MockLLM
except (ImportError, ModuleNotFoundError) as exc:
    Settings = None
    SimpleDirectoryReader = None
    VectorStoreIndex = None
    BaseEmbedding = object
    MockLLM = None
    LLAMAINDEX_IMPORT_ERROR = exc
else:
    LLAMAINDEX_IMPORT_ERROR = None

from config import SOURCES_DIR
from memory import ShortTermContext, load_session_memory, recall_related_memory, remember, save_session_turn
from rag import first_relevant_sentence, parse_front_matter
from research_assistant import validate_citations


DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.03


class HashEmbedding(BaseEmbedding):
    """Tiny local embedding for learning LlamaIndex without paid API calls."""

    embed_dim: int = 384

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._hash_embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.embed_dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.embed_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def configure_llamaindex() -> None:
    if Settings is None or MockLLM is None:
        raise RuntimeError(
            "LlamaIndex could not be imported. Run: pip install -r requirements.txt. "
            f"Original import error: {type(LLAMAINDEX_IMPORT_ERROR).__name__}: {LLAMAINDEX_IMPORT_ERROR}"
        )

    Settings.embed_model = HashEmbedding(embed_dim=384)
    Settings.llm = MockLLM(max_tokens=256)
    Settings.chunk_size = 320
    Settings.chunk_overlap = 40


def load_llamaindex_documents() -> list[Any]:
    if SimpleDirectoryReader is None:
        raise RuntimeError("Missing LlamaIndex. Run: pip install -r requirements.txt")
    return SimpleDirectoryReader(input_dir=str(SOURCES_DIR), required_exts=[".md"]).load_data()


def build_llamaindex_retriever(top_k: int) -> tuple[Any, list[Any]]:
    documents = load_llamaindex_documents()
    index = VectorStoreIndex.from_documents(documents, show_progress=False)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever, documents


def metadata_for_filename(filename: str) -> dict[str, str]:
    path = SOURCES_DIR / filename
    if not path.exists():
        return {"title": filename, "url": f"local://{Path(filename).stem}"}
    metadata, _body = parse_front_matter(path.read_text(encoding="utf-8"))
    return {
        "title": metadata.get("title", filename),
        "url": metadata.get("url", f"local://{Path(filename).stem}"),
    }


def node_to_match(item: Any, index: int) -> dict[str, Any]:
    node = item.node
    filename = node.metadata.get("file_name") or Path(node.metadata.get("file_path", "source.md")).name
    source_id = Path(filename).stem
    metadata = metadata_for_filename(filename)
    raw_text = node.get_content(metadata_mode="none")
    _metadata, text = parse_front_matter(raw_text)
    score = float(item.score or 0.0)
    return {
        "citation_id": f"[{index}]",
        "title": metadata["title"],
        "url": metadata["url"],
        "source_id": source_id,
        "chunk_id": node.node_id,
        "score": round(score, 4),
        "text": text,
    }


def answer_with_citations(topic: str, matches: list[dict[str, Any]], memories: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {
            "answer": "I could not find enough evidence in the LlamaIndex index.",
            "used_citations": [],
            "limitations": ["No retrieved nodes from LlamaIndex."],
            "confidence": 0.2,
        }

    claims = []
    for match in matches:
        sentence = first_relevant_sentence(topic, match["text"]) or match["text"][:220]
        claims.append(f"{sentence} {match['citation_id']}")

    if memories:
        memory_topics = [item["memory"].get("topic", "") for item in memories]
        claims.append("Related prior memory: " + ", ".join(topic for topic in memory_topics if topic))

    confidence = min(0.9, 0.45 + 0.1 * len(matches))
    return {
        "answer": f"LlamaIndex RAG summary for '{topic}': " + " ".join(claims),
        "used_citations": [match["citation_id"] for match in matches],
        "limitations": ["This starter uses a local hash embedding, not a production embedding model."],
        "confidence": round(confidence, 2),
    }


def citations_from_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["citation_id"],
            "title": item["title"],
            "url": item["url"],
            "chunk_id": item["chunk_id"],
            "score": item["score"],
        }
        for item in matches
    ]


def run_llamaindex_rag(
    topic: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    save_memory: bool = True,
) -> dict[str, Any]:
    configure_llamaindex()
    context = ShortTermContext()
    session = load_session_memory()
    memories = recall_related_memory(topic)

    context.add_step(f"Loaded session memory with {len(session.get('turns', []))} turns.")
    context.add_step(f"Recalled {len(memories)} related long-term memories.")
    context.add_step("Loading sources with LlamaIndex SimpleDirectoryReader.")
    retriever, documents = build_llamaindex_retriever(top_k)
    context.add_step(f"Built LlamaIndex VectorStoreIndex from {len(documents)} documents.")

    retrieved_nodes = retriever.retrieve(topic)
    filtered_nodes = [item for item in retrieved_nodes if float(item.score or 0.0) >= min_score]
    matches = [node_to_match(item, index) for index, item in enumerate(filtered_nodes, start=1)]
    context.retrieved_chunks = matches
    context.add_step(f"Retrieved {len(retrieved_nodes)} nodes with LlamaIndex retriever.")
    context.add_step(f"Kept {len(matches)} nodes with score >= {min_score}.")

    model_report = answer_with_citations(topic, matches, memories)
    report = {
        "topic": topic,
        "answer": model_report["answer"],
        "citations": citations_from_matches(matches),
        "used_citations": model_report["used_citations"],
        "limitations": model_report["limitations"],
        "confidence": model_report["confidence"],
        "memory_used": [item["memory"] for item in memories],
        "session_memory": {
            "last_topic": session.get("last_topic"),
            "turn_count": len(session.get("turns", [])),
        },
        "tool_calls": [],
        "errors": context.errors,
        "steps": context.steps,
        "llamaindex": {
            "reader": "SimpleDirectoryReader",
            "index": "VectorStoreIndex",
            "embedding": "HashEmbedding(local learning stub)",
            "top_k": top_k,
            "min_score": min_score,
        },
    }
    report["citation_validation"] = validate_citations(report, context.retrieved_chunks)

    if save_memory:
        remember(topic, report)
        save_session_turn(topic, report)
        report["steps"].append("Saved this LlamaIndex RAG summary to session and long-term memory.")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 LlamaIndex RAG starter.")
    parser.add_argument("topic", nargs="*", help="Research topic, for example: memory for RAG assistants")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of LlamaIndex nodes to retrieve.")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Minimum retrieval score to keep.")
    parser.add_argument("--no-memory", action="store_true", help="Do not save session or long-term memory.")
    args = parser.parse_args()

    topic = " ".join(args.topic).strip()
    if not topic:
        print('Usage: python llamaindex_rag.py "memory for RAG assistants"')
        return 1

    try:
        report = run_llamaindex_rag(topic, top_k=args.top_k, min_score=args.min_score, save_memory=not args.no_memory)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "hint": "Run pip install -r requirements.txt and retry.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
