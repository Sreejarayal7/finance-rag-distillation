"""
Step 7: Evaluate Student vs Teacher.
Metrics: model size, inference latency, and answer accuracy (exact match + F1)
on a held-out set of questions.
"""

import json
import os
import time
import re
import tensorflow as tf
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    DistilBertTokenizerFast, TFDistilBertForQuestionAnswering,
)

with open("qa_dataset.json", encoding="utf-8") as f:
    qa_data = json.load(f)

test_set = qa_data[-5:]

print("Loading Teacher...")
t_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
teacher = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

print("Loading Student...")
s_tokenizer = DistilBertTokenizerFast.from_pretrained("student_model")
student = TFDistilBertForQuestionAnswering.from_pretrained("student_model")


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def ask_teacher(question, context):
    prompt = f"Context: {context}\n\nBased only on the context above, answer this question by copying the most relevant sentence or phrase from the context: {question}"
    inputs = t_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    start = time.time()
    outputs = teacher.generate(**inputs, max_new_tokens=150, no_repeat_ngram_size=3, repetition_penalty=1.3)
    latency = time.time() - start
    answer = t_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, latency


def ask_student(question, context):
    inputs = s_tokenizer(question, context, return_tensors="tf", truncation="only_second", max_length=384, padding="max_length")
    start = time.time()
    outputs = student(**inputs)
    latency = time.time() - start
    start_idx = int(tf.argmax(outputs.start_logits, axis=1)[0])
    end_idx = int(tf.argmax(outputs.end_logits, axis=1)[0])
    if end_idx < start_idx:
        end_idx = start_idx
    tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
    answer = s_tokenizer.decode(tokens, skip_special_tokens=True)
    return answer, latency


print("\n" + "=" * 60)
print("MODEL SIZE COMPARISON")
print("=" * 60)
teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = student.num_parameters()
print(f"Teacher (Flan-T5-Base):  {teacher_params:,} params")
print(f"Student (DistilBERT-QA): {student_params:,} params")
print(f"Size reduction: {teacher_params / student_params:.1f}x fewer parameters")

print("\n" + "=" * 60)
print("PER-QUESTION COMPARISON (held-out test set)")
print("=" * 60)

teacher_latencies, student_latencies = [], []
em_scores, f1_scores = [], []

for item in test_set:
    q, context = item["question"], item["context"]
    teacher_answer, t_lat = ask_teacher(q, context)
    student_answer, s_lat = ask_student(q, context)

    teacher_latencies.append(t_lat)
    student_latencies.append(s_lat)

    em = 1.0 if normalize(student_answer) == normalize(teacher_answer) else 0.0
    f1_score = f1(student_answer, teacher_answer)
    em_scores.append(em)
    f1_scores.append(f1_score)

    print(f"\nQ: {q}")
    print(f"  Teacher : {teacher_answer}  ({t_lat*1000:.0f}ms)")
    print(f"  Student : {student_answer}  ({s_lat*1000:.0f}ms)")
    print(f"  F1 vs Teacher: {f1_score:.2f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Avg Teacher latency: {sum(teacher_latencies)/len(teacher_latencies)*1000:.0f} ms")
print(f"Avg Student latency: {sum(student_latencies)/len(student_latencies)*1000:.0f} ms")
print(f"Speedup: {(sum(teacher_latencies)/len(teacher_latencies)) / (sum(student_latencies)/len(student_latencies)):.1f}x faster")
print(f"Exact Match (vs Teacher): {sum(em_scores)/len(em_scores)*100:.1f}%")
print(f"Avg F1 (vs Teacher): {sum(f1_scores)/len(f1_scores)*100:.1f}%")