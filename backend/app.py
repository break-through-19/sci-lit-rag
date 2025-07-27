import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import qdrant_client
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()
app = Flask(__name__)

# Qdrant configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "scientific_literature")

client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load LLM model and tokenizer
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")

    # Generate query embedding
    query_embedding = embedder.encode(user_query).tolist()

    # Search in Qdrant
    search_result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=5
    )

    # Retrieve top documents
    top_docs = [hit.payload.get("content", "") for hit in search_result]
    context = " ".join(top_docs)

    # Generate response using Flan-T5
    prompt = f"Query: {user_query}\nRelevant Documents: {context}\nAnswer:"
    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=256)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
