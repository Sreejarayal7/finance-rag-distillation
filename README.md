Cross-Framework RAG-Distillation for Finance Q&A
A Teacher (PyTorch) generates grounded answers from SEC 10-K filings using
Retrieval-Augmented Generation. A Student (TensorFlow) is distilled from the
Teacher's outputs to answer the same questions ~5.6x faster with 3.7x fewer
parameters.
Architecture
```
10-K Filings (AAPL, TSLA, MSFT)
        |
   Chunk + Embed (MiniLM) -> FAISS index
        |
   Question -> Retriever -> Top-3 chunks
        |
   Teacher (Flan-T5-Base, PyTorch) -> grounded answer
        |
   Teacher output = training label
        |
   Student (DistilBERT-QA, TensorFlow) -> distilled, trained
        |
   Deployed as FastAPI service
```
Why cross-framework distillation
Most KD techniques assume Teacher and Student share a framework (shared
weights/gradients). Here they don't — so we use response-based (black-box)
distillation: the Student learns only from the Teacher's output text, used
as an answer-span label. This is the same style Hinton et al. (2015)
describe as knowledge transfer without internal access.
Results
Metric	Teacher (Flan-T5-Base)	Student (DistilBERT-QA)
Parameters	247.6M	66.4M (3.7x smaller)
Avg latency	6044 ms	1075 ms (5.6x faster)
Exact Match vs Teacher	—	20.0%
F1 vs Teacher	—	41.9%
Known limitations
Only 19 of 60 Teacher answers were usable as exact-span training labels
(Teacher paraphrased rather than quoted for the rest) — small training set.
Final training loss (0.23) on 19 examples suggests partial memorization
rather than full generalization — scaling the dataset is the clear next step.
Retrieval mismatch observed on a few questions (retriever pulled a
topically-close but incorrect chunk) — a known RAG failure mode.
How to run
```
pip install -r requirements.txt
python step1_get_data.py        # download filings
python step2_embed.py           # build FAISS index
python step4_generate_dataset.py  # generate Teacher Q&A data
python step6_distill.py         # train Student
python step7_evaluate.py        # compare Teacher vs Student
uvicorn step8_app:app --reload  # deploy
```
Stack
Python, PyTorch, TensorFlow, HuggingFace Transformers, Sentence-Transformers,
FAISS, FastAPI.
