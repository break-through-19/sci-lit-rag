import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import tempfile
import pdfplumber
import re

from models import EmbeddingAndLLM
from storage import upsert_embeddings, search_similar

load_dotenv()
app = Flask(__name__)

# Initialize models
models = EmbeddingAndLLM()

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
    cleaned_text = re.split(r"(references|acknowledg(e)?ments?)", text, flags=re.IGNORECASE)[0]

    # Tokenize (split into sentences)
    sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
    if not sentences:
        return jsonify({"message": "No valid text extracted from PDF."}), 400

    # Generate embeddings for each sentence (or chunk)
    embeddings = models.embed(sentences)

    # Store each embedding in Qdrant
    upsert_embeddings(sentences, embeddings)

    return jsonify({"message": "PDF processed and embeddings stored."})

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")

    # Generate query embedding
    query_embedding = models.embed([user_query])[0]

    # Search in Qdrant
    top_docs = search_similar(query_embedding, limit=5)
    context = " ".join(top_docs)

    # Generate response using Flan-T5
    prompt = f"Query: {user_query}\nRelevant Documents: {context}\nAnswer:"
    answer = models.generate(prompt)

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
