import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

# -----------------------------------------
# ENV SETUP
# -----------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

VECTOR_DB_PATH = "vectorstore/faiss_index"

# -----------------------------------------
# LOAD VECTOR DB
# -----------------------------------------
print("📦 Loading vector store...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("✅ Vector store loaded successfully!\n")

# -----------------------------------------
# LLM SETUP
# -----------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# -----------------------------------------
# PROMPT TEMPLATE (STRICT GROUNDING)
# -----------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an E-Commerce Support Assistant.

Answer ONLY using the provided context.

Rules:
- If the answer is NOT in the context, say:
  "I don’t have enough information in the provided documents."
- Do NOT make up information.
- Keep answers clear and helpful.

Context:
{context}

Conversation History:
{history}

User Question:
{question}
""")

# -----------------------------------------
# CHAT LOOP WITH MEMORY
# -----------------------------------------
def chat():
    print("🛍️ E-Commerce Support Chatbot")
    print("Type 'exit' to quit\n")

    history = []

    while True:
        question = input("👤 You: ")

        if question.lower() == "exit":
            print("👋 Goodbye!")
            break

        print("🔎 Searching knowledge base...")

        docs = retriever.invoke(question)

        # Safety: no documents retrieved
        if not docs:
            print("🤖 Bot: I don’t have enough information in the provided documents.")
            print("-" * 60)
            continue

        context = "\n\n".join(doc.page_content for doc in docs)

        # Format conversation history
        history_text = "\n".join(history)

        print("💡 Generating answer...")

        response = llm.invoke(
            prompt.format_messages(
                context=context,
                history=history_text,
                question=question
            )
        )

        answer = response.content.strip()

        print(f"\n🤖 Bot: {answer}")
        print("-" * 60)

        # Update memory
        history.append(f"User: {question}")
        history.append(f"Bot: {answer}")


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    chat()