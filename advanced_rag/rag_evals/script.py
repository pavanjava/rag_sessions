import logging
import os
import sys
import pandas as pd

import qdrant_client
from llama_index.core import Settings, StorageContext
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator, FaithfulnessEvaluator, EvaluationResult
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv(find_dotenv())
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(name=__name__)

os.environ["OPENAI_API_KEY"] = os.environ.get("openai_llm_api_key")

llm = OpenAI(model=os.environ.get("openai_llm_model_id"), temperature=0)
Settings.llm = llm
reader = SimpleDirectoryReader(input_dir="../data/", required_exts=[".pdf"])
documents = reader.load_data(show_progress=True)

data_generator = DatasetGenerator.from_documents(documents)
eval_questions = data_generator.generate_questions_from_nodes(num=15)
logger.info(eval_questions)

relevancy_evaluator_gpt4 = RelevancyEvaluator(llm=llm)
faithfulness_evaluator_gpt4 = FaithfulnessEvaluator(llm=llm)

# create vector index
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

vector_store = QdrantVectorStore(client=client, collection_name="openai_evals_collection")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
if not client.collection_exists(collection_name="openai_evals_collection"):
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
else:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


# define jupyter display function
def display_relevancy_eval_df(query: str, response: Response, eval_result: str) -> None:
    # Get only the first source if there are multiple
    source_content = ""
    if hasattr(response, 'source_nodes') and response.source_nodes:
        source_content = response.source_nodes[0].node.get_content()[:1000] + "..."

    eval_df = pd.DataFrame(
        {
            "Query": [query],  # Note: Wrap in list
            "Response": [str(response)],  # Wrap in list
            "Source": [source_content],  # Wrap in list
            "Evaluation Result": [eval_result]  # Wrap in list
        }
    )

    eval_df.head()


# define jupyter display function
def display_faithfulness_eval_df(response: Response, eval_result: EvaluationResult) -> None:
    if response.source_nodes == []:
        print("no response!")
        return
    eval_df = pd.DataFrame(
        {
            "Response": str(response),
            "Source": response.source_nodes[0].node.text[:1000] + "...",
            "Evaluation Result": "Pass" if eval_result.passing else "Fail",
            "Reasoning": eval_result.feedback,
        },
        index=[0]
    )

    eval_df.head()


query_engine = index.as_query_engine(similarity_top_k=10)
response_vector = query_engine.query(eval_questions[1])
relevancy_eval_result = relevancy_evaluator_gpt4.evaluate_response(
    query=eval_questions[1], response=response_vector
)

faithfulness_eval_result = faithfulness_evaluator_gpt4.evaluate_response(response=response_vector)

display_relevancy_eval_df(eval_questions[1], response_vector, relevancy_eval_result)
display_faithfulness_eval_df(response_vector, faithfulness_eval_result)
