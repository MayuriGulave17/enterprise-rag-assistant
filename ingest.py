import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = []

for file in os.listdir("knowledge_base"):

    if file.endswith(".txt"):

        loader = TextLoader(
            os.path.join("knowledge_base", file)
        )

        documents.extend(loader.load())

print(f"\nLoaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

print(f"\nCreated {len(chunks)} chunks")

for i, chunk in enumerate(chunks):

    print("\n" + "="*50)

    print(f"Chunk {i+1}")

    print(chunk.metadata)

    print(chunk.page_content)