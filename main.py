import tempfile

import streamlit as st
from streamlit_chat import message
from typing_extensions import List

from backend import get_indexed_vector_store, run_query, embed_documents


def create_sources_str(sources: List[str]) -> str:
    if not sources:
        return ""
    sources.sort()
    sources_str = "sources:\n"
    for i, source in enumerate(sources):
        sources_str += f"{i+1}. {source}\n"
    return sources_str


if __name__ == "__main__":
    st.header("Smart Documents Bot")
    prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

    # Add a file uploader to allow PDF uploads
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Display a message confirming the upload
        st.write("Uploaded file:", uploaded_file.name)
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(
                delete_on_close=True, suffix=".pdf"
            ) as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                file_path = tmpfile.name
                embed_documents(file_path)
            st.write("Successfully saved the file to the vector store.")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if prompt:
        with st.spinner("Generating response..."):
            response = run_query(
                vector_store=get_indexed_vector_store(),
                query=prompt,
                chat_history=st.session_state["chat_history"],
            )
            sources = [doc.metadata["source"] for doc in response["source_documents"]]
            formatted_response = (
                f"{response['answer']}\n\n {create_sources_str(sources)}"
            )
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append((prompt, response["answer"]))
    if st.session_state["chat_answers_history"]:
        for response, query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            message(query, is_user=True)
            message(response)
