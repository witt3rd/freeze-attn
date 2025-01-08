import torch

model_name = "cognitivecomputations/Dolphin3.0-Llama3.1-8B"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.bfloat16)
