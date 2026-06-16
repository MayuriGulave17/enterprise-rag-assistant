from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

# Load Embedding Model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Vector DB

vectordb = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

# User Question

query = input("Ask a question: ")
# Retrieve Relevant Chunks

results = vectordb.similarity_search(
    query,
    k=3
)

context = "\n".join(
    [doc.page_content for doc in results]
)

# Prompt

prompt = f"""
Answer the question only from the provided context.

Context:
{context}

Question:
{query}

If the answer is not present in the context,
say:
'I don't have enough information in the knowledge base.'
"""

# Call Llama

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAnswer:\n")

print(response["message"]["content"])

print("\nSources:\n")

sources = set()

for doc in results:
    sources.add(doc.metadata["source"])

for source in sources:
    print(source)