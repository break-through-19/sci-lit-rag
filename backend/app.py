import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import qdrant_client
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
import tempfile
import os
import pdfplumber

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

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"message": "No PDF file provided."}), 400
    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"message": "No selected file."}), 400

    # Save PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_file.save(tmp.name)
        tmp_path = tmp.name

    # Extract text from PDF
    text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    os.remove(tmp_path)

    # Clean text: remove references, acknowledgments (simple regex-based)
    import re
    cleaned_text = re.split(r"(references|acknowledg(e)?ments?)", text, flags=re.IGNORECASE)[0]

    # Tokenize (split into sentences)
    sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
    if not sentences:
        return jsonify({"message": "No valid text extracted from PDF."}), 400

    # Generate embeddings for each sentence (or chunk)
    embeddings = embedder.encode(sentences)

    # Store each embedding in Qdrant
    for idx, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=int(torch.randint(0, 1e9, (1,)).item()),
                    vector=embedding.tolist(),
                    payload={"content": sentence}
                )
            ]
        )

    return jsonify({"message": "PDF processed and embeddings stored."})

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
