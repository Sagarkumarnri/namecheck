import torch
from transformers import AutoModel, AutoTokenizer

# Load the model
model_name = "paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Dummy input (tokenized)
dummy_text = ["Hello world!"]
inputs = tokenizer(dummy_text, return_tensors="pt")["input_ids"]

# Convert the model to TorchScript using tracing
traced_model = torch.jit.trace(model, inputs)

# Save the model as a TorchScript .pt file
torch.jit.save(traced_model, "paraphrase-MiniLM-L6-v2.pt")

print("✅ TorchScript model saved successfully!")
