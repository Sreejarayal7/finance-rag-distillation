"""
Step 8: Deploy the Student model as a simple API.
Run locally, then test in browser at http://127.0.0.1:8000/docs
"""

import json
import faiss
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizerFast, TFDistilBertForQuestionAnswering

TOP_K = 3

print("Loading retriever...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading Student model...")
tokenizer = DistilBertTokenizerFast.from_pretrained("student_model")
student = TFDistilBertForQuestionAnswering.from_pretrained("student_model")

app = FastAPI(title="Finance Q&A - Distilled Student Model")


class Question(BaseModel):
    question: str


def retrieve(question, k=TOP_K):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0]]


@app.post("/ask")
def ask(q: Question):
    retrieved = retrieve(q.question)
    context = " ".join(c["text"] for c in retrieved)

    inputs = tokenizer(q.question, context, return_tensors="tf",
                        truncation="only_second", max_length=384, padding="max_length")
    outputs = student(**inputs)
    start_idx = int(tf.argmax(outputs.start_logits, axis=1)[0])
    end_idx = int(tf.argmax(outputs.end_logits, axis=1)[0])
    if end_idx < start_idx:
        end_idx = start_idx
    tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)

    return {
        "question": q.question,
        "answer": answer,
        "sources": [c["ticker"] for c in retrieved],
    }


@app.get("/")
def home():
    return {"status": "ok", "message": "POST a question to /ask"}