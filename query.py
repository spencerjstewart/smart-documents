from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from typing_extensions import Any


def run(vector_store: VectorStore, query: str) -> Any:

    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})
