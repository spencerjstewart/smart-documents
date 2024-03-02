from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStore

load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
