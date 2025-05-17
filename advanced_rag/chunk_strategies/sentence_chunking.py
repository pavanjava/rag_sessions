from chonkie import SentenceChunker
from doc_reader_util import get_documents

# Basic initialization with default parameters
chunker = SentenceChunker(
    tokenizer_or_token_counter="gpt2",  # Supports string identifiers
    chunk_size=128,    # Maximum tokens per chunk, experiment with [128, 512, 1024, 2048]
    chunk_overlap=30,  # Overlap between chunks
    min_sentences_per_chunk=1, # Minimum number of sentences per chunk
)

docs = get_documents()

for doc in docs:
    chunks = chunker.chunk(doc.text)

    for chunk in chunks:
        print(f"Chunk text: {chunk.text}")
        print(f"Token count: {chunk.token_count}")
        print(f"Start index: {chunk.start_index}")
        print(f"End index: {chunk.end_index}")
