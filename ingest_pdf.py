import os
import json
import faiss
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader

nltk.download("punkt", quiet=True)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=5, min_words=20):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if len(chunk.split()) > min_words:
            chunks.append(chunk.strip())
    return [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]

def save_chunks(chunks, out_path="data/chunks.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {out_path}")


def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2", out_dir="data/faiss_index"):
    model = SentenceTransformer(model_name)
    texts = [entry["text"] for entry in chunks]

    # Compute and normalize embeddings
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    # FAISS index with cosine similarity (L2-normalized dot product)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "procyon.index"))

    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"FAISS index built and saved to: {out_dir}")

if __name__ == "__main__":
    pdf_path = "data/procyon_guide.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    build_faiss_index(chunks)
