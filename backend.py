from dotenv import load_dotenv

load_dotenv()

from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from typing_extensions import Any


def get_indexed_vector_store() -> VectorStore:

    loader = PyPDFLoader("./ocp-study-guide.pdf")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)


def run(vector_store: VectorStore, query: str) -> Any:

    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})
