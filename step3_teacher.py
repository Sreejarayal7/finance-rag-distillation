"""
Step 3: Teacher model = Flan-T5-Base + retrieval (RAG).
Given a question, retrieves top-k relevant chunks, then generates a grounded answer.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

TOP_K = 3

print("Loading retriever...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading Teacher model (Flan-T5-Base)...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
teacher = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


def retrieve(question: str, k: int = TOP_K):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0]]


def ask_teacher(question: str):
    retrieved = retrieve(question)
    context = " ".join(c["text"] for c in retrieved)

    prompt = f"Context: {context}\n\nBased only on the context above, answer this question in a full sentence: {question}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = teacher.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "sources": [c["ticker"] for c in retrieved],
    }


if __name__ == "__main__":
    result = ask_teacher("What is Apple's main business risk mentioned in the filing?")
    print("\nQ:", result["question"])
    print("A:", result["answer"])
    print("Sources:", result["sources"])