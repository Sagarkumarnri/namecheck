from onnx2torch import convert
import torch

# Load the ONNX model
onnx_model_path = "paraphrase-MiniLM-l6-v2.onnx"
torch_model = convert(onnx_model_path)

# Save as PyTorch model
torch.save(torch_model.state_dict(), "paraphrase-MiniLM-l6-v2.pt")
