import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# 1. Load Environment Variables
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")


# -------------------------------
# 2. Load Documents
# -------------------------------
def load_documents(data_path="data"):
    documents = []

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif file.endswith(".md") or file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    return documents


# -------------------------------
# 3. Split Documents into Chunks
# -------------------------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return text_splitter.split_documents(documents)


# -------------------------------
# 4. Create Embeddings + Vector DB
# -------------------------------
def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# -------------------------------
# 5. Save Vector Store
# -------------------------------
def save_vectorstore(vectorstore, path="vectorstore"):
    if not os.path.exists(path):
        os.makedirs(path)

    vectorstore.save_local(path)
    print(f"✅ Vector store saved at: {path}")


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    print("📄 Loading documents...")
    docs = load_documents()

    print(f"✅ Loaded {len(docs)} documents")

    print("✂️ Splitting documents...")
    chunks = split_documents(docs)

    print(f"✅ Created {len(chunks)} chunks")

    print("🔍 Creating embeddings and vector store...")
    vectorstore = create_vectorstore(chunks)

    print("💾 Saving vector store...")
    save_vectorstore(vectorstore)

    print("🎉 Ingestion completed successfully!")


if __name__ == "__main__":
    main()