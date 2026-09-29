"""
Step 2: Split filings into chunks, embed them, store in FAISS.
Output: faiss_index.bin + chunks.json (needed by the Teacher in Step 3)
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 250      # words per chunk
CHUNK_OVERLAP = 50    # words shared between consecutive chunks (keeps context continuous)


def chunk_text(text: str, ticker: str):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunks.append({"ticker": ticker, "text": " ".join(chunk_words)})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main():
    all_chunks = []
    for ticker in ["AAPL", "TSLA", "MSFT"]:
        with open(f"raw_filings/{ticker}.txt", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, ticker)
        all_chunks.extend(chunks)
        print(f"[{ticker}] {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    print("Loading embedding model (first run downloads ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in all_chunks]
    print("Embedding all chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize so inner product = cosine similarity (standard trick for FAISS)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    print(f"\nSaved faiss_index.bin ({index.ntotal} vectors) and chunks.json")


if __name__ == "__main__":
    main()