\# Enterprise Knowledge Assistant



\## Overview



Enterprise Knowledge Assistant is a Retrieval-Augmented Generation (RAG) chatbot built using Llama 3, LangChain, ChromaDB, and Streamlit.



The application allows users to ask questions about company policies and receive accurate answers from a knowledge base with source attribution.



\---



\## Features



\* Local Llama 3 inference using Ollama

\* Document loading and chunking

\* Embedding generation using Sentence Transformers

\* ChromaDB vector database

\* Semantic search retrieval

\* Source attribution

\* Hallucination prevention

\* Interactive Streamlit chat interface



\---



\## Tech Stack



\* Python

\* LangChain

\* Ollama

\* Llama 3

\* ChromaDB

\* Sentence Transformers

\* Streamlit



\---



\## Architecture



Documents



↓



Chunking



↓



Embeddings



↓



ChromaDB



↓



Retriever



↓



Llama 3



↓



Answer + Sources



\---



\## Example Questions



\* How many annual leave days do employees get?

\* What is the probation period?

\* Who is covered under health insurance?

\* When should passwords be changed?

\* What is the maternity leave policy?



\---



\## Run the Application



Create Vector Database:



python vector\_store.py



Run Streamlit App:



python -m streamlit run app.py



\---



\## Author



Mayuri Gulave

AI Engineer | Data Scientist



