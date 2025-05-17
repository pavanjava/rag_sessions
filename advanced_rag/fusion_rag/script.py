import os
from dotenv import load_dotenv, find_dotenv
from typing import List
from llama_index.core import Settings, StorageContext
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
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

vector_store = QdrantVectorStore(client=client, collection_name="fusion_collection")
storage_context = StorageContext.from_defaults(vector_store=vector_store)


class TextCleaner(TransformComponent):
    """
    Transformation to be used within the ingestion pipeline.
    Cleans clutters from texts.
    """

    def __call__(self, nodes, **kwargs) -> List[BaseNode]:
        for node in nodes:
            node.text = node.text.replace('\t', ' ')  # Replace tabs with spaces
            node.text = node.text.replace(' \n', ' ')  # Replace paragprah seperator with spacaes

        return nodes


# Pipeline instantiation with:
# node parser, custom transformer, vector store and documents
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        TextCleaner()
    ],
    vector_store=vector_store,
    documents=documents
)

# Run the pipeline to get nodes
nodes = pipeline.run()
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

# Run the vector store to get vector index
if not client.collection_exists(collection_name="fusion_collection"):
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
else:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

vector_retriever = index.as_retriever(similarity_top_k=2)

retriever = QueryFusionRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    retriever_weights=[

        0.6,  # vector retriever weight
        0.4  # BM25 retriever weight
    ],
    num_queries=5,
    mode=FUSION_MODES.DIST_BASED_SCORE, # FUSION_MODES.RELATIVE_SCORE, FUSION_MODES.RECIPROCAL_RANK, FUSION_MODES.SIMPLE
    use_async=False
)

# Query
query = "What are the key factors of climatic change?"

# Perform fusion retrieval
response = retriever.retrieve(query)
print(response)

for node in response:
    print(f"Node Score: {node.score:.2}")
    print(f"Node Content: {node.text}")
    print("-" * 100)
