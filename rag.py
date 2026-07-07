from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import CHUNK_WORDS, MIN_SCORE, SOURCES_DIR, TOP_K


@dataclass
class SourceDocument:
    source_id: str
    title: str
    path: Path
    url: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    url: str
    text: str
    vector: Counter[str]


def load_markdown_sources() -> list[SourceDocument]:
    documents = []
    for path in sorted(SOURCES_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = parse_front_matter(raw)
        documents.append(
            SourceDocument(
                source_id=path.stem,
                title=metadata.get("title", path.stem),
                path=path,
                url=metadata.get("url", str(path)),
                text=body.strip(),
            )
        )
    return documents


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw

    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw

    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, parts[2]


def tokenize(text: str) -> list[str]:
    return [normalize_token(token) for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*|\d+", text.lower())]


def normalize_token(token: str) -> str:
    aliases = {
        "agents": "agent",
        "chunks": "chunk",
        "chunking": "chunk",
        "citations": "citation",
        "cited": "citation",
        "embedding": "embed",
        "embeddings": "embed",
        "embedded": "embed",
        "fail": "failure",
        "fails": "failure",
        "failed": "failure",
        "failures": "failure",
        "hallucinated": "hallucination",
        "hallucinating": "hallucination",
        "hallucinations": "hallucination",
        "memories": "memory",
        "retrieval": "retrieve",
        "retrieved": "retrieve",
        "retrieves": "retrieve",
        "tools": "tool",
    }
    return aliases.get(token, token)


def embed_text(text: str) -> Counter[str]:
    tokens = tokenize(text)
    vector = Counter(tokens)
    for left, right in zip(tokens, tokens[1:]):
        vector[f"{left}_{right}"] += 1
    return vector


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    common = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def chunk_document(document: SourceDocument) -> list[Chunk]:
    sentences = split_sentences(document.text)
    if not sentences:
        return []

    chunks = []
    current_sentences: list[str] = []
    current_word_count = 0
    index = 1

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_sentences and current_word_count + sentence_word_count > CHUNK_WORDS:
            chunks.append(make_chunk(document, index, current_sentences))
            index += 1
            current_sentences = current_sentences[-1:]
            current_word_count = sum(len(item.split()) for item in current_sentences)

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:
        chunks.append(make_chunk(document, index, current_sentences))

    return chunks


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]


def make_chunk(document: SourceDocument, index: int, sentences: list[str]) -> Chunk:
    text = " ".join(sentences)
    return Chunk(
        chunk_id=f"{document.source_id}#{index}",
        source_id=document.source_id,
        title=document.title,
        url=document.url,
        text=text,
        vector=embed_text(text),
    )


def build_index(documents: list[SourceDocument]) -> list[Chunk]:
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


def retrieve(query: str, chunks: list[Chunk], top_k: int = TOP_K) -> list[dict[str, Any]]:
    query_vector = embed_text(query)
    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk.vector)
        if score >= MIN_SCORE:
            scored.append({"score": round(score, 4), "chunk": chunk})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def first_relevant_sentence(topic: str, text: str) -> str | None:
    query_terms = set(tokenize(topic))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    ranked = []
    for sentence in sentences:
        terms = set(tokenize(sentence))
        overlap = len(query_terms & terms)
        if overlap:
            ranked.append((overlap, sentence.strip()))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else None


def chunk_to_dict(chunk: Chunk, score: float | None = None) -> dict[str, Any]:
    item = {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "title": chunk.title,
        "url": chunk.url,
        "text": chunk.text,
    }
    if score is not None:
        item["score"] = score
    return item
