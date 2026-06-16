import os
import streamlit as st
import ollama

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="centered"
)


# -----------------------------
# Cache Vector Database
# -----------------------------

@st.cache_resource
def load_vector_db():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="vector_db",
        embedding_function=embedding_model
    )

    return vectordb


vectordb = load_vector_db()


# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Header
# -----------------------------

st.title("🏢 Enterprise Knowledge Assistant")

st.markdown("""
Ask questions about company policies using
**RAG (Retrieval Augmented Generation)**,
**Llama 3**, and **ChromaDB**.
""")

st.divider()


# -----------------------------
# Display Previous Messages
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            st.markdown("**Sources:**")

            for source in message["sources"]:

                st.write(
                    f"📄 {os.path.basename(source)}"
                )


# -----------------------------
# User Input
# -----------------------------

question = st.chat_input("Ask a question")


# -----------------------------
# RAG Pipeline
# -----------------------------

if question:

    # Show User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # Retrieve Relevant Chunks

    results = vectordb.similarity_search(
        question,
        k=3
    )

    context = "\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
reply exactly:

I don't have enough information in the knowledge base.
"""

    # Generate Answer

    with st.spinner("Generating answer..."):

        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    answer = response["message"]["content"]

    # Clean Sources

    sources = list(
        set(
            [
                doc.metadata["source"]
                for doc in results
            ]
        )
    )

    # Save Assistant Message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )

    # Display Assistant Message

    with st.chat_message("assistant"):

        st.write(answer)

        st.markdown("**Sources:**")

        for source in sources:

            st.write(
                f"📄 {os.path.basename(source)}"
            )