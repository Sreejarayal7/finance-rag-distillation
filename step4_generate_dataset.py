"""
Step 4: Run a batch of finance questions through the Teacher, save Q&A dataset.
Output: qa_dataset.json (used by the Student for distillation in Step 6)
"""

import json
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

TOP_K = 3

QUESTIONS = [
    "What is Apple's main business risk mentioned in the filing?",
    "What was Apple's revenue driver mentioned in the filing?",
    "What does Apple say about its supply chain?",
    "What is Tesla's main business risk mentioned in the filing?",
    "What does Tesla say about its production capacity?",
    "What does Tesla say about competition?",
    "What is Microsoft's main business risk mentioned in the filing?",
    "What does Microsoft say about its cloud business?",
    "What does Microsoft say about artificial intelligence?",
    "What does Apple say about foreign currency risk?",
    "What does Tesla say about battery supply?",
    "What does Microsoft say about cybersecurity risk?",
    "What does Apple say about litigation risk?",
    "What does Tesla say about regulatory risk?",
    "What does Microsoft say about competition?",
    "What does Apple say about research and development spending?",
    "What does Tesla say about research and development spending?",
    "What does Microsoft say about research and development spending?",
    "What does Apple say about its retail stores?",
    "What does Tesla say about its Gigafactories?",
    "What does Microsoft say about its data centers?",
    "What does Apple say about intellectual property?",
    "What does Tesla say about autonomous driving?",
    "What does Microsoft say about acquisitions?",
    "What does Apple say about economic conditions?",
    "What does Tesla say about raw material costs?",
    "What does Microsoft say about workforce and talent?",
    "What does Apple say about product innovation?",
    "What does Tesla say about energy storage business?",
    "What does Microsoft say about gaming business?",
]

print("Loading retriever...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading Teacher model...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
teacher = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


"""
Step 4 (expanded): Run more finance questions through the Teacher, save Q&A dataset.
"""

import json
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

TOP_K = 3

QUESTIONS = [
    "What is Apple's main business risk mentioned in the filing?",
    "What was Apple's revenue driver mentioned in the filing?",
    "What does Apple say about its supply chain?",
    "What is Tesla's main business risk mentioned in the filing?",
    "What does Tesla say about its production capacity?",
    "What does Tesla say about competition?",
    "What is Microsoft's main business risk mentioned in the filing?",
    "What does Microsoft say about its cloud business?",
    "What does Microsoft say about artificial intelligence?",
    "What does Apple say about foreign currency risk?",
    "What does Tesla say about battery supply?",
    "What does Microsoft say about cybersecurity risk?",
    "What does Apple say about litigation risk?",
    "What does Tesla say about regulatory risk?",
    "What does Microsoft say about competition?",
    "What does Apple say about research and development spending?",
    "What does Tesla say about research and development spending?",
    "What does Microsoft say about research and development spending?",
    "What does Apple say about its retail stores?",
    "What does Tesla say about its Gigafactories?",
    "What does Microsoft say about its data centers?",
    "What does Apple say about intellectual property?",
    "What does Tesla say about autonomous driving?",
    "What does Microsoft say about acquisitions?",
    "What does Apple say about economic conditions?",
    "What does Tesla say about raw material costs?",
    "What does Microsoft say about workforce and talent?",
    "What does Apple say about product innovation?",
    "What does Tesla say about energy storage business?",
    "What does Microsoft say about gaming business?",
    "What does Apple say about its Services segment?",
    "What does Apple say about manufacturing risk?",
    "What does Apple say about climate change?",
    "What does Apple say about data privacy?",
    "What does Apple say about its App Store?",
    "What does Apple say about semiconductor supply?",
    "What does Apple say about employee headcount?",
    "What does Apple say about dividends?",
    "What does Apple say about share repurchases?",
    "What does Apple say about tax matters?",
    "What does Tesla say about Full Self-Driving?",
    "What does Tesla say about vehicle deliveries?",
    "What does Tesla say about solar energy products?",
    "What does Tesla say about its supercharger network?",
    "What does Tesla say about labor relations?",
    "What does Tesla say about government incentives?",
    "What does Tesla say about international operations?",
    "What does Tesla say about cash flow?",
    "What does Tesla say about capital expenditures?",
    "What does Tesla say about manufacturing costs?",
    "What does Microsoft say about Azure?",
    "What does Microsoft say about LinkedIn?",
    "What does Microsoft say about Windows revenue?",
    "What does Microsoft say about regulatory matters?",
    "What does Microsoft say about intellectual property?",
    "What does Microsoft say about data privacy?",
    "What does Microsoft say about climate commitments?",
    "What does Microsoft say about employee benefits?",
    "What does Microsoft say about capital expenditures?",
    "What does Microsoft say about dividends?",
]

print("Loading retriever...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading Teacher model...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
teacher = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


def retrieve(question, k=TOP_K):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0]]


def ask_teacher(question):
    retrieved = retrieve(question)
    context = " ".join(c["text"] for c in retrieved)
    prompt = f"Context: {context}\n\nBased only on the context above, answer this question by copying the most relevant sentence or phrase from the context: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = teacher.generate(
        **inputs,
        max_new_tokens=150,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return context, answer


def main():
    dataset = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q}")
        context, answer = ask_teacher(q)
        dataset.append({"question": q, "context": context, "answer": answer})
        print(f"   -> {answer}\n")

    with open("qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {len(dataset)} Q&A pairs -> qa_dataset.json")


if __name__ == "__main__":
    main()