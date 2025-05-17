from chonkie import SlumberChunker
from chonkie.genie import GeminiGenie, OpenAIGenie
from doc_reader_util import get_documents
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# Optional: Initialize Genie
gemini_genie = GeminiGenie("gemini-2.5-flash-preview-04-17", api_key=os.environ.get("google_llm_api_key"))
openai_genie = OpenAIGenie("o3-mini", api_key=os.environ.get("openai_llm_api_key"))


# Basic initialization
chunker = SlumberChunker(
    genie=gemini_genie,                 # Genie interface to use
    tokenizer_or_token_counter="gpt2",  # Tokenizer or token counter to use
    chunk_size=128,                     # Maximum chunk size
    candidate_size=128,                 # How many tokens Genie looks at for potential splits
    min_characters_per_chunk=30,        # Minimum number of characters per chunk
    verbose=True                        # See the progress bar for the chunking process
)

docs = get_documents()

for doc in docs:
    chunks = chunker.chunk(doc.text)

    for chunk in chunks:
        print(f"Chunk text: {chunk.text}")
        print(f"Token count: {chunk.token_count}")
        print(f"Start index: {chunk.start_index}")
        print(f"End index: {chunk.end_index}")