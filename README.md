\# Enterprise Knowledge Assistant



\## Overview



Enterprise Knowledge Assistant is a Retrieval-Augmented Generation (RAG) chatbot built using Llama 3, LangChain, ChromaDB, Sentence Transformers, and Streamlit.



The application allows users to ask questions about company policies and receive grounded answers from a knowledge base with source attribution.



\---



\## Features



\- Retrieval-Augmented Generation (RAG)

\- Document loading and chunking

\- Semantic search

\- Sentence Transformer embeddings

\- ChromaDB vector database

\- Llama 3 local inference using Ollama

\- Source attribution

\- Hallucination prevention

\- Interactive Streamlit chat interface



\---



\## Tech Stack



\- Python

\- LangChain

\- Ollama

\- Llama 3

\- ChromaDB

\- Sentence Transformers

\- Streamlit



\---



\## Architecture



```text

Company Policy Documents

&#x20;         ↓

&#x20;      Chunking

&#x20;         ↓

&#x20;     Embeddings

&#x20;         ↓

&#x20;      ChromaDB

&#x20;         ↓

&#x20;  Semantic Retrieval

&#x20;         ↓

&#x20;   Relevant Context

&#x20;         ↓

&#x20;      Llama 3

&#x20;         ↓

&#x20;   Answer + Sources

Knowledge Base



The application currently contains company policy documents covering:



HR Policy

Leave Policy

Medical Benefits

Security Policy

Travel Policy

Example Questions

How many annual leave days do employees get?

What is the probation period?

Who is covered under health insurance?

When should passwords be changed?

What is the maternity leave policy?

Run the Application Locally

1\. Install dependencies

pip install -r requirements.txt

2\. Make sure Ollama is installed and Llama 3 is available

ollama pull llama3

3\. Run the Streamlit application

python -m streamlit run app.py



The application will open in your browser.



Project Structure

enterprise-rag-assistant/

│

├── knowledge\_base/

│   ├── hr\_policy.txt

│   ├── leave\_policy.txt

│   ├── medical\_benefits.txt

│   ├── security\_policy.txt

│   └── travel\_policy.txt

│

├── vector\_db/

│   └── ChromaDB files

│

├── app.py

├── ingest.py

├── rag\_chat.py

├── retriever.py

├── vector\_store.py

├── requirements.txt

├── README.md

└── .gitignore

RAG Workflow

Company policy documents are loaded from the knowledge base.

Documents are split into smaller chunks.

Chunks are converted into vector embeddings.

Embeddings are stored in ChromaDB.

A user asks a question.

ChromaDB performs semantic similarity search.

Relevant document chunks are retrieved.

The retrieved context is passed to Llama 3.

Llama 3 generates an answer based only on the retrieved context.

The application displays the answer along with its source documents.

Hallucination Prevention



The application instructs the LLM to answer only using the retrieved knowledge-base context.



If the required information is not available, the application responds:



I don't have enough information in the knowledge base.



This helps keep responses grounded in the available company policy documents.



Author



Mayuri Gulave



AI Engineer | Data Scientist

