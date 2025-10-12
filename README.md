# Advanced RAG

A comprehensive collection of Retrieval-Augmented Generation (RAG) techniques and implementations, designed to help you understand and experiment with various RAG patterns and optimizations.

## Project Overview

This project demonstrates advanced RAG concepts through practical, hands-on implementations using LlamaIndex, various chunking strategies, and retrieval techniques. Each module is self-contained and focuses on a specific RAG concept.

## What You'll Learn

### 1. **Chunking Strategies** (`chunk_strategies/`)
Explore different approaches to splitting documents into meaningful chunks:
- **Token Chunking**: Fixed-size token-based splitting with overlap
- **Sentence Chunking**: Semantic chunking at sentence boundaries
- **Recursive Chunking**: Hierarchical splitting using recursive rules
- **Semantic Chunking**: Content-aware chunking based on semantic similarity
- **Late Chunking**: Contextual embedding generation after chunking
- **Neural Chunking**: ML-based intelligent chunk boundary detection
- **Slumber Chunking**: LLM-guided adaptive chunking

### 2. **Context Enrichment** (`context_enriched_retrieval/`)
Learn how to enhance retrieved context using the PrevNextNodePostprocessor to include surrounding chunks for better context understanding.

### 3. **Fusion RAG** (`fusion_rag/`)
Implement hybrid retrieval by combining:
- Vector-based semantic search
- BM25 keyword-based retrieval
- Weighted fusion strategies for optimal results

### 4. **HyDE (Hypothetical Document Embeddings)** (`hyde/`)
Generate hypothetical answers and use them to improve retrieval accuracy through query transformation.

### 5. **Reranking** (`reranking_rag/`)
Improve retrieval precision by reranking initial results using LLM-based reranking to select the most relevant chunks.

### 6. **RAG Evaluation** (`rag_evals/`)
Measure and improve your RAG system with:
- Relevancy evaluation
- Faithfulness evaluation
- Automated dataset generation for testing

## Key Features

- **Multiple Vector Stores**: Qdrant integration for efficient vector storage
- **Flexible LLM Support**: OpenAI, Anthropic, Ollama, and local models
- **Production-Ready**: Environment-based configuration using `.env` files
- **Evaluation Framework**: Built-in tools to measure RAG performance

## Prerequisites

```bash
pip install -r requirements.txt
```

Required services:
- Qdrant (vector database) running on `localhost:6333`
- Ollama (optional, for local embeddings)

## Project Structure

```
advanced_rag/
├── chunk_strategies/       # Various chunking implementations
├── context_enriched_retrieval/  # Context expansion techniques
├── fusion_rag/            # Hybrid retrieval methods
├── hyde/                  # Query transformation with HyDE
├── reranking_rag/         # Result reranking strategies
├── rag_evals/             # Evaluation and testing tools
├── data/                  # Your PDF documents
└── evals/                 # Evaluation results
```

## Getting Started

1. **Set up environment variables** (create a `.env` file):
```env
openai_llm_api_key=your_key
anthropic_llm_api_key=your_key
google_llm_api_key=your_key
embedding_model_id=nomic-embed-text
embedding_host=http://localhost:11434
```

2. **Add your documents** to the `data/` folder (PDF format)

3. **Start with a chunking strategy**:
```bash
python chunk_strategies/token_chunking.py
```

4. **Experiment with retrieval techniques**:
```bash
python fusion_rag/fusion_rag.py
python hyde/hyde.py
python reranking_rag/reranking_rag.py
```

5. **Evaluate your RAG system**:
```bash
python rag_evals/rag_evals.py
```

## Use Cases

- Building production RAG applications
- Comparing different chunking strategies for your use case
- Implementing hybrid search with multiple retrieval methods
- Evaluating and optimizing RAG performance
- Learning advanced RAG patterns and best practices

## Notes

- Each script is independent and can be run separately
- Vector stores are persisted in Qdrant with unique collection names
- The project uses climate change documents as examples, but works with any PDF content

## Contributing

Feel free to experiment with different configurations, add new RAG techniques, or improve existing implementations!