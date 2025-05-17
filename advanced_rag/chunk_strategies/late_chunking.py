from chonkie import LateChunker, RecursiveRules
from doc_reader_util import get_documents

chunker = LateChunker(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=128,
    rules=RecursiveRules(),
    min_characters_per_chunk=30,
)

docs = get_documents()

for doc in docs:
    chunks = chunker.chunk(doc.text)

    for chunk in chunks:
        print(f"Chunk text: {chunk.text}")
        print(f"Token count: {chunk.token_count}")
        print(f"Start index: {chunk.start_index}")
        print(f"End index: {chunk.end_index}")