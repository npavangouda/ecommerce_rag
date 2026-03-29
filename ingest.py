import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------------------
# ENV SETUP
# -----------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

# -----------------------------------------
# PATHS
# -----------------------------------------
DATA_FOLDER = "data"
VECTOR_DB_PATH = "vectorstore/faiss_index"

# -----------------------------------------
# LOAD DOCUMENTS (.md / .txt)
# -----------------------------------------
def load_documents():
    print("📂 Loading documents...")

    documents = []

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".md") or file.endswith(".txt"):
            file_path = os.path.join(DATA_FOLDER, file)

            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file

            documents.extend(docs)

    if not documents:
        raise ValueError("❌ No documents found in the data folder")

    print(f"✅ Loaded {len(documents)} documents")
    return documents


# -----------------------------------------
# SPLIT DOCUMENTS INTO CHUNKS
# -----------------------------------------
def split_documents(documents):
    print("✂️ Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# -----------------------------------------
# CREATE EMBEDDINGS
# -----------------------------------------
def create_embeddings():
    print("🔍 Creating embeddings...")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )


# -----------------------------------------
# STORE IN FAISS
# -----------------------------------------
def store_vector_db(chunks, embeddings):
    print("📦 Building vector store...")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)

    print(f"💾 Vector store saved at: {VECTOR_DB_PATH}")


# -----------------------------------------
# MAIN PIPELINE
# -----------------------------------------
def main():
    print("\n🚀 Starting ingestion pipeline...\n")

    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    store_vector_db(chunks, embeddings)

    print("\n🎉 Ingestion completed successfully!\n")


if __name__ == "__main__":
    main()