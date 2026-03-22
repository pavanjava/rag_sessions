# What is Qdrant?

Qdrant (pronounced "quadrant") is an open-source, high-performance **vector database and vector similarity search engine** built in Rust. It is designed to store, manage, and search high-dimensional vector embeddings — the numerical representations that modern AI and machine learning models generate from text, images, audio, and other data.

At its core, Qdrant answers the question: *"Given a query, which items in my dataset are most semantically similar?"* Unlike traditional databases that match records by exact values or keywords, Qdrant finds results based on **meaning and context** — making it the backbone of modern AI-powered search applications.

---

## Why Vector Search?

Traditional keyword search works by matching exact words. Semantic search, powered by vector embeddings, understands *intent*. For example, a query like `"alien invasion"` will surface books about Martian attacks or interstellar civilizations — even if none of them contain those exact two words — because the underlying vectors capture conceptual similarity.

Qdrant stores these vectors as **points** in a collection, each carrying:

- A **unique ID**
- A **vector** (the embedding of your content)
- An optional **payload** (structured metadata like title, author, year, tags)

This design makes Qdrant a natural fit for RAG (Retrieval-Augmented Generation) pipelines, recommendation systems, multimodal search, and any application that needs to retrieve contextually relevant data at scale.

---

## Key Concepts

### Collections
A collection is the primary organizational unit in Qdrant — analogous to a table in a relational database. Each collection holds points with vectors of a fixed dimensionality and a configured distance metric (e.g., Cosine, Dot Product, Euclidean).

### Points
Each record stored in Qdrant is a point. A point consists of an ID, one or more vectors, and an optional payload containing arbitrary JSON metadata.

### Payload & Filtering
Payloads allow you to attach structured metadata to each point. Qdrant supports **filtered search** — you can narrow results by payload fields (e.g., "only books published after 2000") without sacrificing vector search performance. Creating a **payload index** on frequently filtered fields is recommended for production use.

### Distance Metrics
Qdrant supports multiple distance functions: **Cosine** (most common for NLP embeddings), **Dot Product**, and **Euclidean**. The choice depends on how your embedding model was trained.

---

## Semantic Search — Getting Started

The simplest Qdrant workflow involves three steps: create a collection, upload points with embeddings, and query by vector similarity.

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, cloud_inference=True)

# Create a collection
client.create_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

# Upload points
client.upload_points(
    collection_name="my_books",
    points=[
        models.PointStruct(
            id=idx,
            vector=models.Document(text=doc["description"], model="sentence-transformers/all-minilm-l6-v2"),
            payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)

# Query
hits = client.query_points(
    collection_name="my_books",
    query=models.Document(text="alien invasion", model="sentence-transformers/all-minilm-l6-v2"),
    limit=3,
).points
```

The `cloud_inference=True` flag delegates embedding generation to Qdrant Cloud, removing the need to manage your own embedding infrastructure. For a full walkthrough, see the [Semantic Search 101 tutorial](https://qdrant.tech/documentation/tutorials-basics/search-beginners/).

---

## Hybrid Search with Reranking

For production-grade search accuracy, Qdrant supports **hybrid search** — combining dense semantic vectors with sparse keyword vectors (BM25) — and **reranking** using late interaction models like ColBERT.

### The Three-Layer Architecture

| Layer | Model Type | Purpose |
|---|---|---|
| Dense Embeddings | `all-MiniLM-L6-v2` | Captures semantic meaning |
| Sparse Embeddings | `BM25` | Handles keyword precision |
| Late Interaction | `ColBERT` | Reranks for contextual relevance |

### How It Works

**Ingestion:** Documents are encoded into all three embedding types and stored in a multi-vector Qdrant collection.

**Retrieval:** A hybrid query runs dense and sparse sub-queries in parallel using the `prefetch` parameter, combining their results.

**Reranking:** The fused candidate set is then reranked by ColBERT's late interaction mechanism, which computes fine-grained token-level similarity between the query and each candidate document.

```python
# Hybrid prefetch (dense + sparse in parallel)
prefetch = [
    models.Prefetch(query=dense_vectors, using="all-MiniLM-L6-v2", limit=20),
    models.Prefetch(query=models.SparseVector(**sparse_vectors.as_object()), using="bm25", limit=20),
]

# Rerank with ColBERT
results = client.query_points(
    "hybrid-search",
    prefetch=prefetch,
    query=late_vectors,
    using="colbertv2.0",
    with_payload=True,
    limit=10,
)
```

For the complete implementation, refer to the [Hybrid Search with Reranking tutorial](https://qdrant.tech/documentation/tutorials-search-engineering/reranking-hybrid-search/).

---

## When to Use Qdrant

- **RAG Pipelines** — Retrieve semantically relevant context for LLM generation
- **Recommendation Systems** — Find similar items based on user behavior vectors
- **Semantic Document Search** — Search codebases, research papers, support tickets by meaning
- **Multimodal Search** — Combine text, image, and audio embeddings in one collection
- **Anomaly Detection** — Surface outliers by finding vectors with low similarity to their neighbors

---

## Deployment Options

Qdrant can be run in multiple environments:

- **Qdrant Cloud** — Managed, forever-free tier available; no infrastructure management required
- **Docker / Self-hosted** — Full control over your cluster and data
- **Qdrant Edge** — Lightweight on-device deployments for mobile and embedded scenarios

---

## References

- [Semantic Search 101 — Qdrant Docs](https://qdrant.tech/documentation/tutorials-basics/search-beginners/)
- [Hybrid Search with Reranking — Qdrant Docs](https://qdrant.tech/documentation/tutorials-search-engineering/reranking-hybrid-search/)
- [Qdrant Cloud](https://cloud.qdrant.io)
- [Qdrant GitHub](https://github.com/qdrant/qdrant)

---

## Connect with Qdrant

> Scan the QR codes below to connect, follow along, or explore related work.

<!-- Replace the image paths below with your actual QR code image files -->

| | |
|:---:|:---:|
| ![QR Code 1](./resources/qdrant_cloud_register.png) | ![QR Code 2](./resources/qdrant_docs.png) |
| *[Qdrant Cloud]* | *[Qdrant Docs]* |

---

*Built with ❤️ using Qdrant — the vector database for the next generation of AI applications.*