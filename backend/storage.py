import os
import torch
import qdrant_client
from qdrant_client.http.models import PointStruct

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "scientific_literature")

client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def upsert_embeddings(sentences, embeddings):
    for sentence, embedding in zip(sentences, embeddings):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=int(torch.randint(0, 1e9, (1,)).item()),
                    vector=embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
                    payload={"content": sentence}
                )
            ]
        )

def search_similar(query_embedding, limit=5):
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=limit
    )
    return [hit.payload.get("content", "") for hit in results]
