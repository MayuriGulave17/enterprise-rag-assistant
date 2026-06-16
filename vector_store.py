import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

# Load Documents

documents = []

for file in os.listdir("knowledge_base"):

    if file.endswith(".txt"):

        loader = TextLoader(
            os.path.join("knowledge_base", file)
        )

        documents.extend(loader.load())

# Chunk Documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# Embedding Model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Vector Database

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vector_db"
)

print("Vector database created successfully!")