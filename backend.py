import os

from dotenv import load_dotenv
from langchain_core.documents import Document

from consts import (
    VECTOR_STORE_PATH,
    VECTOR_STORE_NAME,
)

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from typing_extensions import Any, List, Dict


def get_vector_store() -> VectorStore | None:
    if os.path.exists(VECTOR_STORE_PATH + VECTOR_STORE_NAME + ".faiss"):
        print("Loading existing index at ", VECTOR_STORE_PATH + VECTOR_STORE_NAME)
        return FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            index_name=VECTOR_STORE_NAME,
            embeddings=OpenAIEmbeddings(),
        )
    else:
        print(f"Index {VECTOR_STORE_PATH + VECTOR_STORE_NAME} does not exist")
        return None


def parse_documents(file_path) -> List[Document]:
    loader = PyPDFLoader(file_path, extract_images=False)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")
    return documents


def index_documents(parsed_documents: List[Document]) -> VectorStore:
    if os.path.exists(VECTOR_STORE_PATH + VECTOR_STORE_NAME + ".faiss"):
        print("Loading existing index at ", VECTOR_STORE_PATH + VECTOR_STORE_NAME)
        vector_store = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            index_name=VECTOR_STORE_NAME,
            embeddings=OpenAIEmbeddings(),
        )
        vector_store.add_documents(parsed_documents)
        return vector_store
    else:
        print(
            f"Index {VECTOR_STORE_PATH + VECTOR_STORE_NAME} does not exist, creating..."
        )
        vector_store = FAISS.from_documents(parsed_documents, OpenAIEmbeddings())
        vector_store.save_local(VECTOR_STORE_PATH, VECTOR_STORE_NAME)
        print(
            f"Index successfully created and saved at {VECTOR_STORE_PATH + VECTOR_STORE_NAME}"
        )
        return vector_store


def run_query(
    vector_store: VectorStore, query: str, chat_history: List[Dict[str, Any]]
) -> Any:
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )
    return qa({"question": query, "chat_history": chat_history})
