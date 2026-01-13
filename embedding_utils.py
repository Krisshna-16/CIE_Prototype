from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_patterns():
    with open("patterns.json", "r") as f:
        return json.load(f)

def create_vector_db(patterns):
    texts = [p["description"] for p in patterns]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings
