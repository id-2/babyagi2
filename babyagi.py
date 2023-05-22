#!/usr/bin/env python3

# The embedded document Q&A retrieval functionality is from: https://github.com/imartinez/privateGPT.git
# The main functionality from file privateGPT.py has been integrated in BabyAGI
# Many thanks to https://github.com/imartinez for the great work!

from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

import os
import time
import logging
from collections import deque
from typing import Dict, List
import importlib
import openai
import chromadb
import tiktoken as tiktoken
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import re

# default opt out of chromadb telemetry.
from chromadb.config import Settings

# Engine configuration
# --------------------
# Model: GPT, LLAMA, HUMAN, etc.
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()
LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", 0.9))
LLAMA_CONTEXT = int(os.getenv("LLAMA_CONTEXT", 2000))
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "")
LLAMA_CTX_MAX = int(os.getenv("LLAMA_CTX_MAX", 1024))
LLAMA_THREADS_NUM = int(os.getenv("LLAMA_THREADS_NUM", 8))
LLAMA_FAILSAFE = os.getenv("LLAMA_FAILSAFE", "false").lower() == "true"

# Internet search key configuration
ENABLE_SEARCH_EXTENSION = os.getenv("ENABLE_SEARCH_EXTENSION", "false").lower() == "true"

# Document embedding vector store configuration
ENABLE_DOCUMENT_EXTENSION = os.getenv("ENABLE_DOCUMENT_EXTENSION", "false").lower() == "true"

# Output to file for report
ENABLE_REPORT_EXTENSION = os.getenv("ENABLE_REPORT_EXTENSION", "false").lower() == "true"

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not (LLM_MODEL.startswith("llama") or LLM_MODEL.startswith("human")):
    assert OPENAI_API_KEY, "\033[91m\033[1m" + "OPENAI_API_KEY environment variable is missing from .env" + "\033[0m\033[0m"

# OpenAI model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
openai.api_key = OPENAI_API_KEY

# Table config
PERSISTENT_STORAGE = os.getenv("PERSISTENT_STORAGE", "false").lower() == "true"
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", os.getenv("TABLE_NAME", ""))
assert RESULTS_STORE_NAME, "\033[91m\033[1m" + "RESULTS_STORE_NAME environment variable is missing from .env" + "\033[0m\033[0m"

# Run configuration
INSTANCE_NAME = os.getenv("INSTANCE_NAME", os.getenv("BABY_NAME", "BabyAGI"))
COOPERATIVE_MODE = os.getenv("COOPERATIVE_MODE", "")
JOIN_EXISTING_OBJECTIVE = False

# Goal configuration
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))


# Extensions support begin
# ------------------------
def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
        os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments

        OBJECTIVE, INITIAL_TASK, LLM_MODEL, DOTENV_EXTENSIONS, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE = parse_arguments()

# Human mode extension
# Gives human input to babyagi
if LLM_MODEL.startswith("human"):
    if can_import("extensions.human_mode"):
        from extensions.human_mode import user_input_await

# Load additional environment variables for enabled extensions
# TODO: This might override the following command line arguments as well:
#    OBJECTIVE, INITIAL_TASK, LLM_MODEL, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions

        load_dotenv_extensions(DOTENV_EXTENSIONS)

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions, but also provide command line
# arguments to override them


# Output to file for creation of step-by-step report
# TODO: This is a bit of a hack, but it works for now
if ENABLE_REPORT_EXTENSION:
    REPORT_FILE = os.getenv("REPORT_FILE", "report.txt")
    REPORT_PROMPT = os.getenv("REPORT_PROMPT", "")
    REPORT_TITLE = os.getenv("REPORT_TITLE", "Report")
    OBJECTIVE_ADDER = os.getenv("OBJECTIVE_ADDER", "")

    # Write to summary file
    def write_report(file: str, text: str, mode: chr):
        try:
            with open(file, mode) as f:
                f.write(text)
                summary = f.read()
                return summary
        except:
            return ""      

    # Read from summary file
    def read_report(file: str, text: str):
        try:
            with open(file, 'r') as f:
                text = f.read()
                return text
        except:
            return ""

    # Check text needs to be written to file
    def check_report(result: str):
        if "Task List:" not in result:
            text = ""
            if "Report-Tag" in result:
                text = result.split("Report-Tag")[0]
            elif "Short Story:" in result:
                text = result.split("Short Story:")[1]
            if text:
                write_report(REPORT_FILE, str(text) + "\n", 'a')
                print("Text written to report file...\n")
                print_to_file("Text written to report file...\n", 'a')

    # Check if summary file exists
    def check_report_file(file: str):
        try:
            with open(file, 'r') as f:
                print(f"Using report file: '{REPORT_FILE}'. File already exists. Text will be appended to the file...")
        except:
            with open(file, 'w') as f:
                print(f"Using report file: '{REPORT_FILE}'. File does not exists, creating file...")
                f.write(f"In this file BabyAGI stores the {REPORT_TITLE}, as configured in .env file under ENABLE_REPORT_EXTENSION.\nThe output is derived step by step and new information is appended to the file...\n\n")
        

# Internet smart search (based on BabyCatAGI)
# - SERPAPI, Google CSE or browser search and fallback strategy for API key limits
# - Works also w/o any API key
# - Search results are summarized by LLM
if ENABLE_SEARCH_EXTENSION:
    if can_import("extensions.smart_search"):
        from extensions.smart_search import web_search_tool

        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
        SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")


# Document embedding with Q&A retrieval using langchain (multiple file types supported)
# Load & embedd all supported document in folder 'source_documents' in chromadb vector store with 'document-loader.py'
# The complete functionality is from: https://github.com/imartinez/privateGPT.git
if ENABLE_DOCUMENT_EXTENSION:
    from langchain.chains import RetrievalQA
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.vectorstores import Chroma
    from langchain.llms import GPT4All, LlamaCpp
    import argparse

    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
    embeddings_temperature = float(os.environ.get("EMBEDDINGS_TEMPERATURE", 0.8))
    persist_directory = os.environ.get("DOC_STORE_NAME", "")
    model_type = os.environ.get("EMBEDDINGS_MODEL_TYPE", "LlamaCpp")
    model_path = os.environ.get("EMBEDDINGS_MODEL_PATH", "")
    model_n_ctx = int(os.environ.get("LLAMA_CTX_MAX", 1024))

    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=persist_directory,
            anonymized_telemetry=False
    )

    # Command line parser function
    def parse_arguments():
        parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                    'using the power of LLMs.')
        parser.add_argument("--hide-source", "-S", action='store_true',
                            help='Use this flag to disable printing of source documents used for answers.')

        parser.add_argument("--mute-stream", "-M",
                            action='store_true',
                            help='Use this flag to disable the streaming StdOut callback for LLMs.')
        return parser.parse_args()
    

    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, temperature=embeddings_temperature)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False, temp=embeddings_temperature)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
# ------------------------
# Extensions support end


# Check if local file exists (and contains OBJECTIVE in case of 'task_list.txt')
def check_file():
    try:
        # Check task list output file
        with open('task_list.txt', 'r') as f:
            lines = f.readlines()
            if OBJECTIVE in lines[2]:
                return 'a'
            else:
                return 'w'
    except:
        return 'w'
    

# Write terminal output to 'task_list.txt''
def print_to_file(text: str, mode: chr):
    with open('task_list.txt', mode) as f:
        f.write(text)


# Print configuration
print("\033[95m\033[1m" + "\n*****CONFIGURATION*****" + "\033[0m\033[0m")
print(f"Name  : {INSTANCE_NAME}")
print(f"Mode  : {'alone' if COOPERATIVE_MODE in ['n', 'none'] else 'local' if COOPERATIVE_MODE in ['l', 'local'] else 'distributed' if COOPERATIVE_MODE in ['d', 'distributed'] else 'undefined'}\n")
print(f"LLM Model         : {LLM_MODEL}")
print(f"LLAMA Temperature : {LLAMA_TEMPERATURE}")
print(f"LLAMA Context Max : {LLAMA_CONTEXT}")
print(f"LLAMA CTX MAX     : {LLAMA_CTX_MAX}")
print(f"LLAMA Failsafe    : {LLAMA_FAILSAFE}\n")
print(f"Smart Internet Search Extension : {ENABLE_SEARCH_EXTENSION}")
print(f"Document Embedding Extension    : {ENABLE_DOCUMENT_EXTENSION}")
print(f"File Report Extension           : {ENABLE_REPORT_EXTENSION}\n")
client = chromadb.Client(Settings(anonymized_telemetry=False))

# Check if we know what we are doing
assert OBJECTIVE, "\033[91m\033[1m" + "OBJECTIVE environment variable is missing from .env" + "\033[0m\033[0m"
assert INITIAL_TASK, "\033[91m\033[1m" + "INITIAL_TASK environment variable is missing from .env" + "\033[0m\033[0m"

if ENABLE_REPORT_EXTENSION:
    check_report_file(REPORT_FILE)

# Setup Llama (evaluation and embedding)
if LLM_MODEL.startswith("llama"):
    if can_import("llama_cpp"):
        from llama_cpp import Llama

        print(f"LLAMA : {LLAMA_MODEL_PATH}" + "\n")
        assert os.path.exists(LLAMA_MODEL_PATH), "\033[91m\033[1m" + f"Model can't be found." + "\033[0m\033[0m"

        print('Initialize model for evaluation')
        llm = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=LLAMA_CTX_MAX,
            n_threads=LLAMA_THREADS_NUM,
            n_batch=512,
            use_mlock=False,
        )
        print('\nInitialize model for embedding')
        llm_embed = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=LLAMA_CTX_MAX,
            n_threads=LLAMA_THREADS_NUM,
            n_batch=512,
            embedding=True,
            use_mlock=False,
        )
        print(
            "\033[91m\033[1m"
            + "\n*****USING LLAMA.CPP. POTENTIALLY SLOW.*****"
            + "\033[0m\033[0m"
        )
    else:
        print(
            "\033[91m\033[1m"
            + "\nLlama LLM requires package llama-cpp. Falling back to GPT-3.5-turbo."
            + "\033[0m\033[0m"
        )
        LLM_MODEL = "gpt-3.5-turbo"

if LLM_MODEL.startswith("gpt-4"):
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

if LLM_MODEL.startswith("human"):
    print(
        "\033[91m\033[1m"
        + "\n*****USING HUMAN INPUT*****"
        + "\033[0m\033[0m"
    )

# Print objective
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")
print_to_file("\n*****OBJECTIVE*****\n" + f"{OBJECTIVE}\n", mode=check_file())
if ENABLE_REPORT_EXTENSION:
    print("\033[93m\033[1m" + "\nObjective adder:" + "\033[0m\033[0m" + f" {OBJECTIVE_ADDER}")
    print_to_file("\nObjective adder:" + f" {OBJECTIVE_ADDER}" + "\n", 'a')
          
if not JOIN_EXISTING_OBJECTIVE:
    print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")
    print_to_file("\nInitial task:" + f" {INITIAL_TASK}" + "\n", 'a')
else:
    print("\033[93m\033[1m" + f"\nJoining to help the objective" + "\033[0m\033[0m")
    print_to_file(f"\nJoining to help the objective", 'a')

if ENABLE_REPORT_EXTENSION:
    print("\033[93m\033[1m" + "\nReport title:" + "\033[0m\033[0m" + f" {REPORT_TITLE}")
    print_to_file("\nReport title:" + f" {REPORT_TITLE}" + "\n", 'a')
    print("\033[93m\033[1m" + "\nReport prompt:" + "\033[0m\033[0m" + f" {REPORT_PROMPT}")
    print_to_file("\nReport title:" + f" {REPORT_PROMPT}" + "\n", 'a')

# Llama embedding function
class LlamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        return

    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = []
        for t in texts:
            e = llm_embed.embed(t)
            embeddings.append(e)
        return embeddings


# Results storage using local ChromaDB
class DefaultResultsStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        # Non-persistent vector storage
        if not PERSISTENT_STORAGE:
            chroma_db_impl="duckdb"            
        # Persistent vector storage
        else:
            chroma_db_impl="duckdb+parquet"

        chroma_client = chromadb.Client(
                settings=chromadb.config.Settings(
                    chroma_db_impl=chroma_db_impl,
                    persist_directory=chroma_persist_dir,
                )
            )
        
        metric = "cosine"
        if LLM_MODEL.startswith("llama"):
            embedding_function = LlamaEmbeddingFunction()
        else:
            embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: str, result_id: str):

        # Break the function if LLM_MODEL starts with "human" (case-insensitive)
        if LLM_MODEL.startswith("human"):
            return
        # Continue with the rest of the function

        embeddings = llm_embed.embed(result) if LLM_MODEL.startswith("llama") else None
        if (
                len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0
        ):  # Check if the result already exists
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={"task": task["task_name"], "result": result},
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
            include=["metadatas"]
        )
        return [item["task"] for item in results["metadatas"][0]]


# Initialize results storage
def try_weaviate():
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_USE_EMBEDDED = os.getenv("WEAVIATE_USE_EMBEDDED", "False").lower() == "true"
    if (WEAVIATE_URL or WEAVIATE_USE_EMBEDDED) and can_import("extensions.weaviate_storage"):
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
        from extensions.weaviate_storage import WeaviateResultsStorage
        print("\nUsing results storage: " + "\033[93m\033[1m" + "Weaviate" + "\033[0m\033[0m")
        return WeaviateResultsStorage(OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_USE_EMBEDDED, LLM_MODEL, LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
    return None

def try_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    if PINECONE_API_KEY and can_import("extensions.pinecone_storage"):
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
        assert (
            PINECONE_ENVIRONMENT
        ), "\033[91m\033[1m" + "PINECONE_ENVIRONMENT environment variable is missing from .env" + "\033[0m\033[0m"
        from extensions.pinecone_storage import PineconeResultsStorage
        print("\nUsing results storage: " + "\033[93m\033[1m" + "Pinecone" + "\033[0m\033[0m")
        return PineconeResultsStorage(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, LLM_MODEL, LLAMA_MODEL_PATH, RESULTS_STORE_NAME, OBJECTIVE)
    return None

def use_chroma():
    print("\nUsing results storage: " + "\033[93m\033[1m" + "Chroma (Default)" + "\033[0m\033[0m")
    return DefaultResultsStorage()

results_storage = try_weaviate() or try_pinecone() or use_chroma()


# Task storage supporting only a single instance of BabyAGI
class SingleTaskListStorage:
    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]


# Initialize tasks storage
tasks_storage = SingleTaskListStorage()
if COOPERATIVE_MODE in ['l', 'local']:
    if can_import("extensions.ray_tasks"):
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parent))
        from extensions.ray_tasks import CooperativeTaskListStorage

        tasks_storage = CooperativeTaskListStorage(OBJECTIVE)
        print("\nReplacing tasks storage: " + "\033[93m\033[1m" + "Ray" + "\033[0m\033[0m")
elif COOPERATIVE_MODE in ['d', 'distributed']:
    pass


def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])


def openai_call(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.lower().startswith("llama"):
                result = llm(prompt[:LLAMA_CTX_MAX],
                             stop=["### Human"],
                             echo=False,
                             temperature=LLAMA_TEMPERATURE,
                             top_k=40,
                             top_p=0.95,
                             repeat_penalty=1.05,
                             max_tokens=400)
                # print('\n*****RESULT JSON DUMP*****\n')
                # print(json.dumps(result))
                # print('\n')
                return result['choices'][0]['text'].strip()
            elif model.lower().startswith("human"):
                return user_input_await(prompt)
            elif not model.lower().startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use 4000 instead of the real limit (4097) to give a bit of wiggle room for the encoding of roles.
                # TODO: different limits for different models.

                trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)

                # Use chat completion API
                messages = [{"role": "system", "content": trimmed_prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def task_creation_agent(
        objective: str, result: Dict, task_description: str, task_list: List[str], internet_result: str, search_result: str
):  
    if LLM_MODEL.startswith("llama"):
        task_description = task_description[0:LLAMA_CONTEXT]
        
    prompt = f"""
You are to use the result from an execution agent to create new tasks with the following objective: {objective}\n
The last completed task has the result: \n{result["data"]}
This result was based on this task description: {task_description}.\n"""

    if task_list:
        prompt += f"These are incomplete tasks: {', '.join(task_list)}\n"
    prompt += "Based on the result, return a list of tasks to be completed in order to meet the objective. "
    if task_list:
        prompt += "These new tasks must not overlap with incomplete tasks. "

    prompt += """
Return one task per line in your response. The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period. If your list is empty, write "There are no tasks to add at this time."
Unless your list is empty, do not include any headers before your numbered list or follow your numbered list with any other output.
\nYour response: """

    print(f"\n*****TASK CREATION AGENT PROMPT****{prompt}")
    print_to_file(f"\n****TASK CREATION AGENT PROMPT****\n{prompt}\n", 'a')
    response = openai_call(prompt, max_tokens=2000)
    if LLM_MODEL.startswith("llama") and len(response) > LLAMA_CONTEXT:
        response = response[0:LLAMA_CONTEXT]

    print(f"\n****TASK CREATION AGENT RESPONSE****{response}")
    print_to_file(f"\n****TASK CREATION AGENT RESPONSE****\n{response}\n", 'a')
    new_tasks = response.split('\n')
    new_tasks_list = []

    # Llama failsafe routine (execute last task again and use failsafe results for creation of new_tasks_list)
    if LLM_MODEL.startswith("llama") and LLAMA_FAILSAFE and (task_description in response or len(response) < int(LLAMA_CONTEXT/5) or not new_tasks):
        print(f'\nContext has been lost and task creation prompt or response is truncated,... create failsafe prompt and request response.\n')
        print_to_file(f'\nContext has been lost and task creation prompt or response is truncated,... create failsafe prompt and request response.\n', 'a')

        if ENABLE_SEARCH_EXTENSION:
            if task_description == "":
                task_description = INITIAL_TASK + " for objective: " + objective
            result, next_task_flag = llama_failsafe_agent(task_description, internet_result, search_result)
        else:
            result = execution_agent(OBJECTIVE, task_description)

        print(result)
        print_to_file(result + "\n", 'a')
        check_report(result)
        print("\033[92m\033[1m" + "\n*****FAILSAFE TASK EXECUTION*****\n" + "\033[0m\033[0m" + result + "\n")
        print_to_file("\n*****FAILSAFE TASK EXECUTION*****\n" + result + "\n", 'a')
        response = openai_call(prompt, max_tokens=2000)
        if len(response) > LLAMA_CONTEXT:
            response = response[0:LLAMA_CONTEXT]
        print("\033[93m\033[1m" + "\n*****FAILSAFE TASK CREATION*****\n" + "\033[0m\033[0m" + "\n" + response + "\n")
        print_to_file("\n*****FAILSAFE TASK CREATION*****\n" + response + "\n", 'a')
        new_tasks = response.split('\n')
        new_tasks_list = []

    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip() and task_id.isnumeric():
                new_tasks_list.append(task_name)
            print('New task created: ' + task_name)

    out = [{"task_name": task_name} for task_name in new_tasks_list]
    return out


def prioritization_agent(enriched_result, task, internet_result, search_result):
    task_names = tasks_storage.get_task_names()
    bullet_string = '\n'

    prompt = f"""
You are tasked with prioritizing the following tasks: {bullet_string + bullet_string.join(task_names)}
Consider the ultimate objective of your team: {OBJECTIVE}.
Tasks should be sorted from highest to lowest priority, where higher-priority tasks are those that act as pre-requisites.
Do not remove any tasks. Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1. The number of each entry must be followed by a period.
Do not include any headers before your ranked list or follow your list with any other output."""

    print(f"\n****TASK PRIORITIZATION AGENT PROMPT****\n{prompt}")
    print_to_file(f"\n****TASK PRIORITIZATION AGENT PROMPT****\n{prompt}\n", 'a')

    if LLM_MODEL.startswith("llama"):
        response = openai_call(prompt, max_tokens=2000)
        if len(response) > LLAMA_CONTEXT:
            response = response[0:LLAMA_CONTEXT]
    else:
        response = openai_call(prompt, max_tokens=2000)

    # Llama failsafe routine (Create new tasks based on last task and add to task_storage)
    if LLM_MODEL.startswith("llama") and LLAMA_FAILSAFE and len(response) < int(LLAMA_CONTEXT/10):
        print(f'\nContext has been lost and task prioritization prompt is truncated,... create new tasks again and add to task storage.\n')
        print_to_file(f'\nContext has been lost and task prioritization prompt is truncated,... create new tasks again and add to task storage.\n', 'a')
        if internet_result:
            enriched_result = {
                    "data": internet_result
                }
        new_tasks = task_creation_agent(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            tasks_storage.get_task_names(),
            internet_result,
            search_result,
        )
        
        response = openai_call(prompt, max_tokens=2000)
        if len(response) > LLAMA_CONTEXT:
            response = response[0:LLAMA_CONTEXT]

    print(f"\n****TASK PRIORITIZATION AGENT RESPONSE****\n{response}")
    print_to_file(f"\n****TASK PRIORITIZATION AGENT RESPONSE****\n{response}\n", 'a')
    if not response:
        print('Received empty response from priotritization agent. Keeping task list unchanged.')
        print_to_file('Received empty response from priotritization agent. Keeping task list unchanged.', 'a')
        #print('Adding new tasks to task_storage')
        return
    
    new_tasks = response.split("\n") if "\n" in response else [response]
    new_tasks_list = []
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
            task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
            if task_name.strip():
                new_tasks_list.append({"task_id": task_id, "task_name": task_name})
                print('New task created: ' + task_name)

    return new_tasks_list


# Execute a task based on the objective and five previous tasks
def execution_agent(objective: str, task: str) -> str:
    """
    Executes a task based on the given objective and previous context.

    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.

    Returns:
        str: The response generated by the AI for the given task.

    """
    context = []
    if INITIAL_TASK not in task:
        if LLM_MODEL.startswith("llama"):
            top_results_num = 3
            result_factor = 3
            context = context_agent(query=objective, top_results_num=top_results_num)
            for i in range(0, len(context)):
                context[i] = context[i][int(i*(LLAMA_CONTEXT/(top_results_num*result_factor))):int((LLAMA_CONTEXT/(top_results_num*result_factor)*(i+1)))]
        else:
            context = context_agent(query=objective, top_results_num=5)
                
    print(f"\033[96m\033[1m\n*****RELEVANT CONTEXT*****\033[0m\033[0m\n{context}")
    print_to_file(f"\n*****RELEVANT CONTEXT*****\n{context}\n", 'a')

    if ENABLE_DOCUMENT_EXTENSION:
        doc_context = ""
        doc_length = int(LLAMA_CONTEXT/3)
        query = f"Consider the length limit of {doc_length} characters for your response on the following query: {task}\n"
        query += f"Take into account also the context from previous tasks: {str(context)}\nYour response: "

        # Retrieve document embedding context via Q&A
        if INITIAL_TASK in task:
            query = ""

        print(f"\033[96m\033[1m\n*****DOCUMENT EMBEDDING CONTEXT*****\033[0m\033[0m")
        print_to_file(f"\n*****DOCUMENT EMBEDDING CONTEXT*****\n", 'a')
        if query:
            res = qa(query)
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']
            if LLM_MODEL.startswith("llama") and len(str(answer)) > doc_length:
                if answer:
                    doc_context = str(answer)[0:doc_length]
            else:
                if answer:
                    doc_context = str(answer)
            print_to_file(doc_context + "\n\n", 'a')
            print()

    if ENABLE_REPORT_EXTENSION:
        objective = objective + " " + OBJECTIVE_ADDER + " " + REPORT_PROMPT

    prompt = f'Perform one task based on the following objective: {objective}\nYour task: {task}\n'
    if context:
        prompt += 'Take into account these previously completed tasks:' + '\n'.join(context)
    if ENABLE_DOCUMENT_EXTENSION and doc_context:
        prompt += f'\nConsider the answer on the task from related document embedding query: {doc_context}'
    if ENABLE_SEARCH_EXTENSION:
        #prompt += f'\nDo your best to complete the task and provide a useful answer. Responding with the task content itself, a variation of it or vague suggestions, is not #useful as well. In this case assume that internet search is required.'
        prompt += f'\nIf internet search is required to complete the task, add "Internet search request: " to the response and add the task redrafted to an optimal concise internet search request.'
    prompt += f'\n\nYour response: '
    return openai_call(prompt, max_tokens=2000)


# Get the top n completed tasks for the objective
def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    results = results_storage.query(query=query, top_results_num=top_results_num)
    # print("****RESULTS****")
    # print(results)
    return results


# Check if google needs to be accessed, based on the the text in last completed task result
def internet_agent(result: str, task: str):
    search_request = ""
    search_results = ""
    prompt = ""
    lines = result.split("\n")
    result = ""
    line_flag = False

    # Do not trigger internet search for INITIAL_TASK
    if INITIAL_TASK not in task:
        for l in lines:
            if "search request: " in l:
                try:
                    search_request = l.split("search request: ")[1]
                    break
                except:
                    line_flag = True
            elif line_flag:
                try:
                    search_request = l.strip()
                    break
                except:
                    break

        # Get top urls from smart search and scrape web pages
        if search_request:
            search_results = ""
            search_extract = ""
            num_results = 5
            if LLM_MODEL.startswith("llama"):
                num_results = 2

            if GOOGLE_API_KEY and GOOGLE_CSE_ID:
                print("\nAccess smart search with Google CSE")
                print_to_file("\nAccess smart search with Google CSE\n", 'a')
                search_results += web_search_tool(OBJECTIVE, task, num_results, "google")
            elif SERPAPI_API_KEY:
                print("\nAccess smart search with SERPAPI")
                print_to_file("\nAccess smart search with SERPAPI\n", 'a')
                search_results += web_search_tool(OBJECTIVE, task, num_results, "serpapi")
            else:
                print("\nAccess smart search with www.duckduckgo.com")
                print_to_file("\nAccess smart search with www.duckduckgo.com\n", 'a')
                search_results += web_search_tool(OBJECTIVE, task, num_results, "browser")

    # Evaluate search results with LLM
    if search_results:
        if LLM_MODEL.startswith("llama"):
            for i in range(0, num_results):
                search_extract += search_results[int(i*(LLAMA_CONTEXT/num_results)):int((LLAMA_CONTEXT/num_results)*(i+1))]
        else:
            search_extract = search_results
            
        # Use extracted search result as result
        result = search_extract
        if LLM_MODEL.startswith("llama") and len(result) > LLAMA_CONTEXT:
            result = result[0:LLAMA_CONTEXT]

        print("\033[93m\033[1m" + "\n*****TASK RESULT WITH SMART SEARCH*****" + "\033[0m\033[0m" + f"\n{result}")
        print_to_file("\n*****TASK RESULT WITH SMART SEARCH*****\n" + f"{result}\n", 'a')
    return(prompt, result, search_results)


# Llama failsafe routine (create new prompt with internet search prompt/results and send prompt)
def llama_failsafe_agent(task: str, internet_result: str, search_result: str):
    #print('\nTruncated task result,... formulate a new prompt with internet search prompt & results and send prompt.')
    #print_to_file('\nTruncated task result,... formulate a new prompt with internet search prompt & results and send prompt.', 'a')
    next_task_flag = False
    objective = OBJECTIVE
    context = context_agent(query=objective, top_results_num=5)

    # Check internet search result
    if search_result == "":
        if task:
            if INITIAL_TASK in task:
                task = INITIAL_TASK + " for the objective: " + OBJECTIVE
            prompt, result, search_result = internet_agent(internet_result, task)

            print('\nInternet search result not available,... repeat last task: ' + task)
        else:
            print('\nInternet search result not available and task is empty, use initial task and objective for new prompt.')
            objective = INITIAL_TASK + " for the objective: " + OBJECTIVE

        next_task_flag=True
    else:
        print('\nInternet search result is available...')
        if len(internet_result) > LLAMA_CONTEXT:
            internet_result = internet_result[0:LLAMA_CONTEXT]
    
    if ENABLE_DOCUMENT_EXTENSION:
        doc_context = ""
        doc_length = int(LLAMA_CONTEXT/3)

        # Get the answer from the chain
        query = f"Consider the length limit of {doc_length} characters for your response on the following query: {task}\n"
        query += f"Take into account also the context from previous tasks: {str(context)}\nYour response: "
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        # Failsafe for llama with limited context length
        if LLM_MODEL.startswith("llama") and len(str(answer)) > doc_length:
            if answer:
                doc_context = str(answer)[0:doc_length]
        else:
            if answer:
                doc_context = str(answer)

    prompt = f'Perform one task based on the following objective: {objective}\nYour task: {task}'
    if context:
        prompt += '\nTake into account these previously completed tasks:' + '\n'.join(context)
    if ENABLE_DOCUMENT_EXTENSION and doc_context:
        prompt += f'\nConsider the answer on the task from related document embedding query: {doc_context}'
    if internet_result:
        prompt += f'\nConsider the internet search result for the task: {internet_result}'
    prompt += f'\n\nYour response: '
    result = openai_call(prompt, max_tokens=2000)

    if len(result) > LLAMA_CONTEXT:
        result = result[0:LLAMA_CONTEXT]
    print("\033[92m\033[1m" + "\n*****FAILSAFE TASK EXECUTION*****\n" + "\033[0m\033[0m" + result + "\n")
    print_to_file("\n*****FAILSAFE TASK EXECUTION*****\n" + result + "\n", 'a')
    print(result)
    print_to_file(result, 'a')
    check_report(result)

    return result, next_task_flag


# Add the initial task if starting new objective
if not JOIN_EXISTING_OBJECTIVE:
    initial_task = {
        "task_id": tasks_storage.next_task_id(),
        "task_name": INITIAL_TASK
    }
    tasks_storage.append(initial_task)


def main():
    loop = True
    llama_failsafe_counter = 0
    while loop:
        # As long as there are tasks in the storage...
        if not tasks_storage.is_empty():
            # Print the task list
            print("\033[95m\033[1m" + "\n*****TASK LIST*****" + "\033[0m\033[0m")
            print_to_file("\n*****TASK LIST*****\n", 'a')
            for t in tasks_storage.get_task_names():
                print(" • " + str(t))
                print_to_file((" • " + str(t)), 'a')
            print_to_file("\n", 'a')

            # Step 1: Pull the first incomplete task
            task = tasks_storage.popleft()
            print("\033[92m\033[1m" + "\n*****NEXT TASK*****" + "\033[0m\033[0m")
            print(str(task["task_name"]))
            print_to_file("\n*****NEXT TASK*****\n" + str(task["task_name"]) + "\n", 'a')

            # Send to execution function to complete the task based on the context
            result = execution_agent(OBJECTIVE, str(task["task_name"]))
            print("\033[93m\033[1m" + "\n*****TASK RESULT*****" + "\033[0m\033[0m\n" + result)
            print_to_file("\n*****TASK RESULT*****\n" + f"{result}\n", 'a')

            # Step 2: Check if internet search is required for conclusion of this task
            internet_prompt = ""
            internet_result = ""
            search_result = ""
            internet_prompt, internet_result, search_result = internet_agent(result, str(task["task_name"]))

            # Failsafe for llama with limited context length
            if LLM_MODEL.startswith("llama") and LLAMA_FAILSAFE and len(result) < int(LLAMA_CONTEXT/10) and len(internet_result) < int(LLAMA_CONTEXT/10):
                result, next_task_flag = llama_failsafe_agent(str(task["task_name"]), internet_result, search_result)

                # Normal procedure with enriched result storage and then create new tasks
                if not next_task_flag or len(result) > int(LLAMA_CONTEXT/100):
                    print(f'\nTask result is OK -> Continue with result storage...')
                    print_to_file(f'\nTask result is OK -> Continue with result storage...\n', 'a')
                    # Step 3: Enrich result and store in the results storage
                    #print(f"\nUpdate embedded vector store with result: {result}")
                    enriched_result = {
                        "data": result
                    }
                    result_id = f"result_{task['task_id']}"
                    results_storage.add(task, result, result_id)

                # Do not execute step 3 (enriched result storage), but select next task and execute step 4 (create new tasks)
                else:
                    print(f'\nTask result is incomprehensible,... select next task.\n')
                    print_to_file(f'\nTask result is incomprehensible,... select next task.\n', 'a')
                    task.update({"task_id": tasks_storage.next_task_id()})

            # Step 3: Enrich result and store in the results storage
            # This is where you should enrich the result if needed
            else:
                if internet_result:
                    result = internet_result
                #print(f"\nUpdate embedded vector store with result: {result}")
                enriched_result = {
                    "data": result
                }
                # extract the actual result from the dictionary
                # since we don't do enrichment currently
                # vector = enriched_result["data"]

                result_id = f"result_{task['task_id']}"
                results_storage.add(task, result, result_id)

                # Check if result includes output to file
                check_report(result)

            # Step 4: Create new tasks and re-prioritize task list
            # only the main instance in cooperative mode does that
            new_tasks = task_creation_agent(
                OBJECTIVE,
                enriched_result,
                task["task_name"],
                tasks_storage.get_task_names(),
                result,
                search_result,
            )
            
            print('Adding new tasks to task_storage...')
            for new_task in new_tasks:
                new_task.update({"task_id": tasks_storage.next_task_id()})
                print(str(new_task))
                tasks_storage.append(new_task)

            if not JOIN_EXISTING_OBJECTIVE:
                prioritized_tasks = prioritization_agent(enriched_result, task, result, search_result)
                if prioritized_tasks:
                    tasks_storage.replace(prioritized_tasks)

            # Sleep a bit before checking the task list again
            time.sleep(5)

        else:
            if LLM_MODEL.startswith("llama") and LLAMA_FAILSAFE and llama_failsafe_counter < 2:
                llama_failsafe_counter+=1
                print(f"\nFailsafe triggered at exit,... trigger failsafe routine. Failsafe counter: {llama_failsafe_counter}")
                result, next_task_flag = llama_failsafe_agent(str(task["task_name"]), result, search_result)

                # Step 3: Enrich result and store in the results storage
                #print(f"\nUpdate embedded vector store with result: {result}")
                enriched_result = {
                    "data": result
                }
                result_id = f"result_{task['task_id']}"
                results_storage.add(task, result, result_id)

                # Check if result includes output to file
                check_report(result)

                if tasks_storage.is_empty():
                    print('Task storage is empty, ...create new tasks.')
                    new_tasks = task_creation_agent(
                        OBJECTIVE,
                        enriched_result,
                        task["task_name"],
                        tasks_storage.get_task_names(),
                        result,
                        search_result,
                    )                 
                    print('Adding new tasks to task_storage...')
                    for new_task in new_tasks:
                        new_task.update({"task_id": tasks_storage.next_task_id()})
                        print(str(new_task))
                        tasks_storage.append(new_task)
                else:
                    print('Task storage is not empty, ...select next task.')
                    task.update({"task_id": tasks_storage.next_task_id()})
                    tasks_storage.append(task)

                # Sleep a bit before checking the task list again
                time.sleep(5)

            else:
                print('Done.')
                loop = False


if __name__ == "__main__":
    main()

