from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class EmbeddingAndLLM:
    def __init__(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    def embed(self, texts):
        return self.embedder.encode(texts)

    def generate(self, prompt, max_new_tokens=256):
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.llm_model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
