import os

from dotenv import load_dotenv

from consts import OCP_PDF_PATH, OCP_INDEX_FILE_NAME, OCP_INDEX_DIR_PATH

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from typing_extensions import Any, List, Dict


def get_indexed_vector_store() -> VectorStore:
    # TODO use one vector store, we don't need to create multiple
    if os.path.exists(OCP_INDEX_DIR_PATH + OCP_INDEX_FILE_NAME + ".faiss"):
        print("Loading existing index...")
        return load_local_vector_store()
    else:
        print(
            f"Index {OCP_INDEX_DIR_PATH + OCP_INDEX_FILE_NAME} "
            + "does not exist, creating..."
        )
        os.makedirs(OCP_INDEX_DIR_PATH, exist_ok=True)
        return create_local_vector_store()


def load_local_vector_store() -> VectorStore:
    # TODO use one vector store, we don't need to create multiple
    print(f"Loading vector store {OCP_INDEX_FILE_NAME} at {OCP_INDEX_DIR_PATH}")
    return FAISS.load_local(
        folder_path=OCP_INDEX_DIR_PATH,
        index_name=OCP_INDEX_FILE_NAME,
        embeddings=OpenAIEmbeddings(),
    )


def create_local_vector_store() -> VectorStore:
    loader = PyPDFLoader(OCP_PDF_PATH, extract_images=True)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(
        folder_path=OCP_INDEX_DIR_PATH, index_name=OCP_INDEX_FILE_NAME
    )
    return vector_store


def embed_documents(file_path):
    loader = PyPDFLoader(file_path, extract_images=True)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")
    embeddings = OpenAIEmbeddings()
    # TODO implement get_vector_store()
    vector_store = get_vector_store()
    for doc in documents:
        embedding = embeddings.generate(doc)
        if not is_duplicate(embedding, vector_store):
            vector_store.add(embedding)


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
