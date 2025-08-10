import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# === Paths ===
BASE_DIR = os.path.dirname(__file__)  # scripts folder

KB1_JSON_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/knowledge_bases/kb1_data.json"))
VECTOR_DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/vector_dbs/kb1_faiss.index"))
METADATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/vector_dbs/kb1_metadata.json"))

os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)

# === Loading KB1 data ===
with open(KB1_JSON_PATH, "r", encoding="utf-8") as f:
    kb1_data = json.load(f)
print(f"Loaded {len(kb1_data)} KB1 entries.")

# === Model for embeddings ===
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims

# === Preparing text blobs for embedding ===
texts = []
for entry in kb1_data:
    text_blob = (
        " ".join(entry.get("doc_types", [])) + " " +
        entry.get("legal_process", "") + " " +
        entry.get("description", "") + " " +
        " ".join(entry.get("required_documents", []))
    ).strip()
    texts.append(text_blob)

# === Creating normalized embeddings ===
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
embeddings = normalize(embeddings, axis=1).astype('float32')

# === Building FAISS index (cosine similarity via normalized L2) ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} KB1 vectors.")

# === Save index & metadata ===
faiss.write_index(index, VECTOR_DB_PATH)
print(f"Saved FAISS index to {VECTOR_DB_PATH}")

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(kb1_data, f, indent=2, ensure_ascii=False)
print(f"Saved KB1 metadata to {METADATA_PATH}")
