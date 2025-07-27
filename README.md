# sci-lit-rag

_AI-powered Literature Review using RAG and LLMs for Scientific Research_

## Description
sci-lit-rag is a Retrieval-Augmented Generation (RAG) pipeline designed for scientific literature review. It combines document retrieval, embedding-based similarity search (using Qdrant), and open-source LLM-powered response generation to assist researchers in efficiently reviewing scientific literature.

## Features
- Document retrieval from open-source archives like arXiv.
- Embedding generation and similarity search using Qdrant and `sentence-transformers/all-MiniLM-L6-v2`.
- Query-based response generation with `google/flan-t5-large` (Hugging Face Transformers).
- Web app interface for interaction.

## Setup Instructions
1. **Prerequisites**:
   - Python 3.8 or higher.
   - Required libraries: Flask, qdrant-client, python-dotenv, sentence-transformers, transformers, torch, etc.
   - Qdrant instance (local or Qdrant Cloud).

2. **Steps**:
   - Clone the repository.
   - Install dependencies using `pip install -r requirements.txt`.
   - Create a `.env` file in the `backend` directory with the following content:
     ```
     QDRANT_HOST=your-qdrant-host
     QDRANT_PORT=6333
     QDRANT_COLLECTION=scientific_literature
     ```
   - **Do not commit this file to version control. It is already included in `.gitignore`.**
   - Run the backend and frontend servers.

## Usage
- Input queries through the web app interface.
- Retrieve summaries or detailed explanations based on the query.
- Example query: "Summarize recent advancements in quantum computing."

## Architecture
- **Document Retrieval**: Fetch documents from open-source archives.
- **Embedding Generation**: Generate vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Database**: Store embeddings in Qdrant for similarity search.
- **Answer Generation**: Use `google/flan-t5-large` (Hugging Face Transformers) to generate responses based on retrieved documents and user queries.

## Deployment
- Designed for deployment on Render (free hosting).
- All configuration is managed via environment variables for portability.
- Qdrant can be used via Qdrant Cloud or a managed instance.
