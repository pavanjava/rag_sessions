from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.prompts import PromptType
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import os

load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"] = os.environ.get("openai_llm_api_key")

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge"
    "--------------------------------------------------\n"
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
    "--------------------------------------------------\n"
    "FINALLY, NOT TO GENERATE SIMILAR THINGS: Okay class, here are five questions to prepare you for our "
    "upcoming quiz based on the provided text about Understanding Climate Change. Please read each question "
    "carefully"
)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

# a set of documents loaded by using for example a Reader
reader = SimpleDirectoryReader(input_dir="../data/", required_exts=[".pdf"])
documents = reader.load_data(show_progress=True)

llm = Ollama(model="gemma3:4b", base_url="http://localhost:11434")

print("Dataset generation initiated.")
dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=llm,
    num_questions_per_chunk=5,  # set the number of questions per nodes
    workers=3,
    text_qa_template=DEFAULT_TEXT_QA_PROMPT,
    show_progress=True,
)

rag_dataset:LabelledRagDataset = dataset_generator.generate_dataset_from_nodes()
df:pd.DataFrame = rag_dataset.to_pandas()
df.to_csv(path_or_buf='ground_truth_dataset.csv')