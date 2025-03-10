from transformers import AutoTokenizer
import onnxruntime as ort

tokenizer = AutoTokenizer.from_pretrained("paraphrase-MiniLM-L6-v2")
tokenizer.save_pretrained("tokenizer")