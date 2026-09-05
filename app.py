import os
import streamlit as st
import ollama

from google import genai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="centered"
)


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

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


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# HEADER
# =========================================================

st.title("🏢 Enterprise Knowledge Assistant")

st.markdown(
    """
    Ask questions about company policies using:

    **RAG (Retrieval-Augmented Generation)**  
    **ChromaDB**  
    **Semantic Search**  
    **AI Language Models**
    """
)

st.divider()


# =========================================================
# DISPLAY PREVIOUS MESSAGES
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):

            st.markdown("**Sources:**")

            for source in message["sources"]:

                st.write(
                    f"📄 {os.path.basename(source)}"
                )


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(prompt):

    # -----------------------------------------------------
    # Check for Gemini API key
    # -----------------------------------------------------

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None


    # -----------------------------------------------------
    # CLOUD MODE
    # Gemini API
    # -----------------------------------------------------

    if api_key:

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text


    # -----------------------------------------------------
    # LOCAL MODE
    # Ollama + Llama 3
    # -----------------------------------------------------

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================================================
# USER INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about company policies..."
)


# =========================================================
# RAG PIPELINE
# =========================================================

if question:

    # -----------------------------------------------------
    # DISPLAY USER QUESTION
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)


    # -----------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # -----------------------------------------------------

    with st.spinner("Searching knowledge base..."):

        results_with_scores = (
            vectordb.similarity_search_with_relevance_scores(
                question,
                k=3
            )
        )


    # -----------------------------------------------------
    # FILTER IRRELEVANT DOCUMENTS
    # -----------------------------------------------------

    relevance_threshold = 0.35

    results = [
        doc
        for doc, score in results_with_scores
        if score >= relevance_threshold
    ]


    # -----------------------------------------------------
    # NO RELEVANT INFORMATION
    # -----------------------------------------------------

    if not results:

        answer = (
            "I don't have enough information "
            "in the knowledge base."
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": []
            }
        )

        with st.chat_message("assistant"):

            st.write(answer)

        st.stop()


    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context = "\n\n".join(
        [
            doc.page_content
            for doc in results
        ]
    )


    # -----------------------------------------------------
    # RAG PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are an Enterprise Knowledge Assistant.

Your job is to answer questions about company policies.

IMPORTANT RULES:

1. Answer ONLY using the provided knowledge-base context.
2. Do NOT use outside knowledge.
3. Do NOT make up information.
4. Do NOT assume information that is not explicitly available.
5. If the answer is not available in the context, reply exactly:

I don't have enough information in the knowledge base.

6. Keep the answer clear and concise.

--------------------------------------------------

KNOWLEDGE BASE CONTEXT:

{context}

--------------------------------------------------

USER QUESTION:

{question}

--------------------------------------------------

ANSWER:
"""


    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    with st.spinner("Generating answer..."):

        try:

            answer = generate_answer(prompt)

        except Exception as e:

            answer = (
                "Unable to generate an answer. "
                "Please check the AI model configuration."
            )

            st.error(
                f"Model error: {e}"
            )


    # -----------------------------------------------------
    # GET SOURCES
    # -----------------------------------------------------

    sources = list(
        set(
            [
                doc.metadata.get(
                    "source",
                    "Unknown source"
                )
                for doc in results
            ]
        )
    )


    # -----------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )


    # -----------------------------------------------------
    # DISPLAY ANSWER
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write(answer)

        if sources:

            st.markdown("**Sources:**")

            for source in sources:

                st.write(
                    f"📄 {os.path.basename(source)}"
                )