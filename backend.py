import os

from dotenv import load_dotenv

from consts import OCP_PDF_PATH, OCP_INDEX_FILE_NAME, OCP_INDEX_DIR_PATH

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from typing_extensions import Any


def get_indexed_vector_store() -> VectorStore:
    if os.path.exists(OCP_INDEX_DIR_PATH + OCP_INDEX_FILE_NAME):
        print("Loading existing index...")
        load_local_vector_store()
    else:
        print(
            f"Index {OCP_INDEX_DIR_PATH + OCP_INDEX_FILE_NAME} "
            + "does not exist, creating..."
        )
        os.makedirs(OCP_INDEX_DIR_PATH, exist_ok=True)
        return create_local_vector_store()


def load_local_vector_store() -> VectorStore:
    FAISS.load_local(folder_path=OCP_INDEX_DIR_PATH, index_name=OCP_INDEX_FILE_NAME)


def create_local_vector_store() -> VectorStore:
    loader = PyPDFLoader(OCP_PDF_PATH)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)


def run_query(vector_store: VectorStore, query: str) -> Any:

    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})
