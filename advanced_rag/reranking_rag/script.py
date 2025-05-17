import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import Settings, StorageContext
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
import qdrant_client

# Original path append replaced for Colab compatibility
# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get('openai_llm_api_key')

# Llamaindex global settings for llm and embeddings
Settings.llm = OpenAI(model=os.environ.get("openai_llm_model_id"), temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name=os.environ.get("embedding_model_id"),
                                       base_url=os.environ.get("embedding_host"))
Settings.chunk_size = 128
Settings.chunk_overlap = 128 // 5

path = "../data"
reader = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = reader.load_data()

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

vector_store = QdrantVectorStore(client=client, collection_name="reranking_collection")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if not client.collection_exists(collection_name="reranking_collection"):
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
else:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine_w_llm_rerank = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[LLMRerank(top_n=5)]
)

resp = query_engine_w_llm_rerank.query("What are the key factors of climatic change?")
print(resp)