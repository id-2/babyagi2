#!/usr/bin/env python3

# The loading & embedding functionality is from: https://github.com/imartinez/privateGPT.git
# The files constants.py and ingest.py from repo have been combined into this file
# Many thanks to https://github.com/imartinez for the great work!

import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

# Document loading
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings

# Document embedding
import argparse


load_dotenv()

# Load environment variables
llama_type = os.environ.get("EMBEDDINGS_MODEL_PATH", "").split("/")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
embeddings_temperature = float(os.environ.get("EMBEDDINGS_TEMPERATURE", 0.8))
model_type = os.environ.get("EMBEDDINGS_MODEL_TYPE", "LlamaCpp")
model_path = os.environ.get("EMBEDDINGS_MODEL_PATH", "")
model_n_ctx = int(os.environ.get("LLAMA_CTX_MAX", 1024))
chunk_size = 500
chunk_overlap = 50

source_directory = os.environ.get('DOC_SOURCE_PATH', 'source_documents')
scrape_directory = os.environ.get('DOC_SCRAPE_PATH', 'scrape_documents')

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".xhtml": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    print(f"Found {len(filtered_files)} files to load...")

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.append(doc)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    print(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


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


# API: Load documents from directory & embedd in vector store
def document_loader(source_directory: str, persist_directory: str, embeddings: HuggingFaceEmbeddings):
    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=persist_directory,
            anonymized_telemetry=False
    )

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at: {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Document loading & embedding complete! Vectorstore has been setup at: {persist_directory}")


class Document:
    def __init__(self, content, sources):
        self.page_content = content
        self.metadata = sources


def process_text(qa_result: str, links: str) -> List[Document]:
    """
    Load text and split in chunks
    """
    print(f"Appending all available results to document embedding store...")
    
    # Using a dictionary for metadata
    sources = {"source": links if links else llama_type}
    document = Document('\n'.join([qa_result]), sources)

    if not document.page_content:
        print("No new results to load")
        exit(0)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents([document])
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")

    return texts


def check_write(file_path: str, input: str, text: str):
    with open(file_path, 'r') as f:
        f.read()
        if input not in f:
            print(f"Store {text} in Q&A document: {scrape_directory}/qa_memory.txt")
            return True
        else:
            print(f"{text} already stored in Q&A document.")
            return False
        

# API: Write text to file
def text_writer(file_path: str, input: str, text: str):
    try:
        with open(file_path, 'r') as f:
            mode = 'a'
    except:
        mode = 'w'

    write_flag = check_write(file_path, input, text)
    if input and write_flag:
        if input.startswith("As an AI assistant"):
            input = input.split(". ")[1]
        with open(file_path, mode) as f:
            if mode == 'w':
                f.write("# This file contains the list of all completed task results (for Q&A retrieval), if available with web scrape summary, LLM validated summary, source URLs and complete web pages (for later using with 'ingest.py')\n# The write feature can be enabled/disbaled with ENABLE_DOC_UPDATE.\n# The update of document embedding with new result data for presistent entity vector memory (see ENABLE_STORE_UPDATE) works indepently from this file.\n# New result data (which is not yet stored) is appended to this file. If the vectorstore (see DOC_STORE_NAME) is deleted, BabyAGI's persistent entity memory is 'erased', but still exists in this file (and can be loaded again with 'ingest.py')...\n\n")
            f.write(input)
        return input
    elif not write_flag:
        return ""
    else:
        print("Error: Extracting text failed")
        return ""


# API: Load text from e.g. internet result & embedd in vector store
def text_loader(persist_directory: str, qa_result: str, links: str):
    # Define the Chroma settings
    CHROMA_SETTINGS = Settings(
            chroma_db_impl='duckdb+parquet',
            persist_directory=persist_directory,
            anonymized_telemetry=False
    )

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at: {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        if qa_result != "":
            texts = process_text(qa_result, links)
            print(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)
            db.persist()
            db = None
            print(f"Document loading & embedding complete! Vectorstore has been updated at: {persist_directory}")
            return True
        else:
            print("No new results to add to document embeddings")
            return False
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_text(qa_result, links)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
        db.persist()
        db = None
        print(f"Document loading & embedding complete! Vectorstore has been setup at: {persist_directory}")
        return True

