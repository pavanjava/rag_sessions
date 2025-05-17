from chonkie import TokenChunker
from doc_reader_util import get_documents
from sentence_transformers import SentenceTransformer
from advanced_rag.qdrant_util.vector_store import ChonkieVectorStore


vector_store = ChonkieVectorStore(collection_prefix="token_chunker_")
collection_name = vector_store.create_collection(collection_name="demo", vector_size=384)
print(f"collection: {collection_name} created")

# Basic initialization with default parameters
chunker = TokenChunker(
    tokenizer="gpt2",  # Supports string identifiers
    chunk_size=128,  # Maximum tokens per chunk, experiment with [128, 512, 1024, 2048]
    chunk_overlap=30  # Overlap between chunks
)

encoder = SentenceTransformer("all-MiniLM-L6-v2")
docs = get_documents()

for doc in docs:
    chunks = chunker.chunk(doc.text)
    doc.extra_info["text"] = doc.text
    # doc.extra_info = {'page_label': '1', 'file_name': 'Understanding_Climate_Change.pdf', 'file_path': '../data/Understanding_Climate_Change.pdf', 'file_type': 'application/pdf', 'file_size': 206372, 'creation_date': '2025-05-04', 'last_modified_date': '2025-05-04'}
    for chunk in chunks:
        # print(f"Chunk text: {chunk.text}")
        # print(f"Token count: {chunk.token_count}")
        # print(f"Start index: {chunk.start_index}")
        # print(f"End index: {chunk.end_index}")
        vectors = encoder.encode([chunk.text])
        payload = doc.extra_info
        _id = vector_store.upsert_point(collection_name=collection_name, vector=vectors[0], payload=payload)
        print(f"point {_id} created successfully.")