---
title: RAG Chunking, Embeddings, And Retrieval Quality
url: local://rag_chunking_embeddings
---

RAG quality depends heavily on chunking, embedding, and retrieval settings. The model can only answer from the evidence that retrieval brings back. If the wrong chunks are retrieved, even a strong language model may produce weak, incomplete, or unsupported answers.

Chunking splits documents into smaller searchable units. In LlamaIndex these units are often called nodes. A document becomes many nodes, each node receives metadata from the original document, and each node can be embedded and retrieved independently. Chunk size controls how much context is inside one node. Chunk overlap repeats some text between adjacent nodes so important context is not lost at a boundary.

Small chunks are more precise, but they can lose surrounding context. Large chunks preserve more context, but they may mix unrelated ideas and make retrieval less focused. For tutorials and articles, a practical starting point is 300 to 800 tokens with 50 to 100 tokens of overlap. For technical documentation, section-aware chunking based on headings is often better than blind token splitting. For code, function-level or class-level chunking is usually better than normal text chunking. For FAQ data, each question-answer pair can often be one chunk.

Embedding converts text into a vector representation. In a vector RAG system, every chunk is embedded and stored. The user query is embedded too. Retrieval compares the query vector with chunk vectors and returns the most similar chunks. A production embedding model can capture semantic similarity, so related phrases can match even when they use different words. A simple hash embedding is useful for learning the pipeline but is not a production semantic model.

The embedding model directly affects recall. Weak embeddings may miss relevant chunks when the query and document use different vocabulary. Stronger embedding models, query rewriting, metadata filters, hybrid search, and reranking can improve retrieval quality. The best answer generation cannot fix missing evidence if retrieval never selected the right chunks.

Important retrieval settings include top_k, similarity threshold, metadata filters, and reranking. A low top_k may miss useful evidence. A high top_k may include too much noise. A minimum score threshold can prevent very weak chunks from being cited. Reranking can reorder retrieved chunks using a stronger model after the first retrieval pass.

Good RAG evaluation starts by inspecting retrieval before inspecting the final answer. For each test query, check whether the retrieved chunks actually contain the evidence needed to answer. If retrieval is poor, adjust chunk size, overlap, metadata, embedding model, top_k, query rewriting, or reranking before blaming the answer model.

In the current learning project, LlamaIndex uses SimpleDirectoryReader to load source documents, VectorStoreIndex to index nodes, and a local HashEmbedding to avoid paid embedding calls. This is useful for understanding the mechanics, but a production assistant should replace HashEmbedding with a real embedding model and evaluate retrieval quality with representative questions.
