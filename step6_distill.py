"""
Step 6: Distillation training.
Student (extractive QA) learns to point to the Teacher's answer inside the context.
This is the Teacher-generated qa_dataset.json being used as training labels = distillation.
"""

import json
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForQuestionAnswering

MAX_LEN = 384
EPOCHS = 15
BATCH_SIZE = 4
LR = 3e-5

print("Loading Teacher-generated dataset...")
with open("qa_dataset.json", encoding="utf-8") as f:
    qa_data = json.load(f)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
student = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")


def build_training_examples(qa_data):
    examples = []
    skipped = 0

    for item in qa_data:
        context = item["context"]
        question = item["question"]
        answer = item["answer"].strip()

        start_char = context.lower().find(answer.lower())
        if start_char == -1 or len(answer) == 0:
            skipped += 1
            continue

        end_char = start_char + len(answer)

        encoding = tokenizer(
            question, context,
            max_length=MAX_LEN, truncation="only_second",
            padding="max_length", return_offsets_mapping=True,
        )
        offsets = encoding.pop("offset_mapping")

        start_token = end_token = None
        for i, (s, e) in enumerate(offsets):
            if s <= start_char < e:
                start_token = i
            if s < end_char <= e:
                end_token = i
        if start_token is None or end_token is None:
            skipped += 1
            continue

        examples.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "start_position": start_token,
            "end_position": end_token,
        })

    print(f"Usable training examples: {len(examples)}  (skipped {skipped} — Teacher paraphrased, no exact span match)")
    return examples


examples = build_training_examples(qa_data)

input_ids = np.array([e["input_ids"] for e in examples])
attention_mask = np.array([e["attention_mask"] for e in examples])
start_positions = np.array([e["start_position"] for e in examples])
end_positions = np.array([e["end_position"] for e in examples])

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

dataset = tf.data.Dataset.from_tensor_slices(
    (input_ids, attention_mask, start_positions, end_positions)
).shuffle(100).batch(BATCH_SIZE)

print(f"\nTraining Student for {EPOCHS} epochs on {len(examples)} examples...")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    steps = 0
    for ids, mask, s_pos, e_pos in dataset:
        with tf.GradientTape() as tape:
            outputs = student(
                input_ids=ids, attention_mask=mask,
                start_positions=s_pos, end_positions=e_pos,
                training=True,
            )
            loss = outputs.loss
        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        epoch_loss += float(tf.reduce_mean(loss))
        steps += 1

    print(f"Epoch {epoch+1}/{EPOCHS} — avg loss: {epoch_loss/steps:.4f}")

student.save_pretrained("student_model")
tokenizer.save_pretrained("student_model")
print("\nSaved trained Student -> student_model/")