import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# === Paths ===
BASE_DIR = os.path.dirname(__file__)  # scripts folder

KB2_JSON_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/knowledge_bases/kb2_data.json"))
VECTOR_DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/vector_dbs/kb2_faiss.index"))
METADATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/vector_dbs/kb2_metadata.json"))

os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)

# === Loading KB2 data ===
with open(KB2_JSON_PATH, "r", encoding="utf-8") as f:
    kb2_data = json.load(f)
print(f"Loaded {len(kb2_data)} KB2 entries.")

# === Model for embeddings ===
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims

# === Preparing text blobs for embedding ===
texts = []
for entry in kb2_data:
    text_blob = (
        entry.get("red_flag_category", "") + " " +
        entry.get("description", "") + " " +
        " ".join(entry.get("examples", [])) + " " +
        " ".join(entry.get("detection_keywords", []))
    ).strip()
    texts.append(text_blob)

# === Creating normalized embeddings ===
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
embeddings = normalize(embeddings, axis=1).astype('float32')

# === Building FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} KB2 vectors.")

# === Save index & metadata ===
faiss.write_index(index, VECTOR_DB_PATH)
print(f"Saved FAISS index to {VECTOR_DB_PATH}")

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(kb2_data, f, indent=2, ensure_ascii=False)
print(f"Saved KB2 metadata to {METADATA_PATH}")
