"""
Step 5: Set up the Student model in TensorFlow (different framework from the PyTorch Teacher).
Just confirms it loads correctly — training happens in Step 6.
"""

from transformers import DistilBertTokenizerFast, TFDistilBertForQuestionAnswering

print("Loading Student tokenizer + model (TensorFlow, ~250MB download)...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
student = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

print("\nStudent model loaded successfully.")
print("Framework:", student.framework)
print("Total parameters:", student.num_parameters())