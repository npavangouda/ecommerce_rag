
# E-Commerce RAG Chatbot

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot for E-Commerce support. 
It answers user queries using product manuals, return policies, and shipping policies.

## Features
- Uses OpenAI embeddings and GPT model
- FAISS vector database for retrieval
- Context-aware conversation (supports follow-up questions)
- Strict grounding (no hallucination)
- CLI-based chatbot

## Architecture
1. Document Ingestion
   - Load documents (.md files)
   - Split into chunks
   - Generate embeddings
   - Store in FAISS

2. Chatbot
   - Retrieve top-k chunks
   - Inject into LLM prompt
   - Maintain conversation history
   - Generate grounded responses

## Setup

### Install dependencies
```
pip install -r requirements.txt
```

### Add environment variables
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key
```

### Run ingestion
```
python ingest.py
```

### Run chatbot
```
python chatbot.py
```

## Example Queries
- What is the return period?
- What if my product is damaged?
- What is delivery time?

## Refusal Behavior
If information is not found:
"I don’t have enough information in the provided documents."

## Tech Stack
- Python
- LangChain
- OpenAI API
- FAISS
