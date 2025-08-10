import os
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np

# Loading model globally
model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_docx(docx_path: str) -> str:
    """Extracting and clean all text from a .docx file."""
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"File not found: {docx_path}")
    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def embed_text(text: str) -> np.ndarray:
    """Creating a normalized embedding vector for the given text."""
    emb = model.encode([text], convert_to_numpy=True)
    emb = normalize(emb, axis=1).astype('float32')
    return emb[0]

def parse_and_embed(docx_path: str):
    """
    Full pipeline: parse .docx and return (cleaned_text, embedding_vector)
    """
    text = parse_docx(docx_path)
    embedding_vector = embed_text(text)
    return text, embedding_vector
