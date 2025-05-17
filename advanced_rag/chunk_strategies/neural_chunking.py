from chonkie import NeuralChunker
from doc_reader_util import get_documents

# Basic initialization with default parameters
chunker = NeuralChunker(
    model="mirth/chonky_modernbert_base_1",  # Default model
    device_map="mps",                        # Device to run the model on ('cpu', 'cuda', 'mps' etc.)
    min_characters_per_chunk=10,             # Minimum characters for a chunk
    return_type="chunks"                     # Output type
)

docs = get_documents()

for doc in docs:
    chunks = chunker.chunk(doc.text)

    for chunk in chunks:
        print(f"Chunk text: {chunk.text}")
        print(f"Token count: {chunk.token_count}")
        print(f"Start index: {chunk.start_index}")
        print(f"End index: {chunk.end_index}")