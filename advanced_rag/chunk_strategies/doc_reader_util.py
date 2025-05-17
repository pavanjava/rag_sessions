from llama_index.core import SimpleDirectoryReader, Document
from typing import List

def get_documents() -> List[Document]:
    # Understanding_Climate_Change.pdf, rag.pdf
    docs: List[Document] = SimpleDirectoryReader(input_files=["../data/Understanding_Climate_Change.pdf"]).load_data(show_progress=True)

    return docs