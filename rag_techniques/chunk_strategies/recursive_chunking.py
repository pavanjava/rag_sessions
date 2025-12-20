from chonkie import RecursiveChunker, RecursiveRules
from doc_reader_util import get_documents

# Basic initialization with default parameters
chunker = RecursiveChunker(
    tokenizer_or_token_counter="gpt2",  # Supports string identifiers
    chunk_size=128,    # Maximum tokens per chunk, experiment with [128, 512, 1024, 2048]
    rules=RecursiveRules() # rules on which chunking should be done.
)

docs = get_documents()

for doc in docs:
    chunks = chunker.chunk(doc.text)

    for chunk in chunks:
        print(f"Chunk text: {chunk.text}")
        print(f"Token count: {chunk.token_count}")
        print(f"Start index: {chunk.start_index}")
        print(f"End index: {chunk.end_index}")
