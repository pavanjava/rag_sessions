import logging
import os
import sys

from qdrant_client import qdrant_client

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

Settings.embed_model =  OllamaEmbedding(model_name=os.environ.get("ollama_embedding_model_id"),
                                        base_url=os.environ.get("ollama_embedding_host"))
Settings.llm = Anthropic(model=os.environ.get("anthropic_llm_model_id"), api_key=os.environ.get("anthropic_llm_api_key"))
Settings.chunk_size = 128
Settings.chunk_overlap = 30

def load_document():
    # load documents
    _documents = SimpleDirectoryReader(input_dir="../data", required_exts=[".pdf"]).load_data()
    return _documents

documents = load_document()

client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # host="localhost"
    # otherwise set Qdrant instance with host and port:
    url="http://localhost:6333",
    # set API KEY for Qdrant Cloud
    api_key="th3s3cr3tk3y",
)

vector_store = QdrantVectorStore(client=client, collection_name="hyde_collection")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
if not client.collection_exists(collection_name="hyde_collection"):
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
else:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_str = "What are the key factors of climatic change?"
query_engine = index.as_query_engine()
response = query_engine.query(query_str)
logging.info(f"response before HyDE: {response}")

hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query(query_str)
logging.info(f"response after HyDE: {response}")