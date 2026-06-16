from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

query = "How many annual leave days do employees get?"

results = vectordb.similarity_search(
    query,
    k=3
)

print("\nRetrieved Documents:\n")

for doc in results:

    print("=" * 50)

    print(doc.metadata["source"])

    print(doc.page_content)