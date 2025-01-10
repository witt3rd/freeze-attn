from pprint import pprint

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.pipelines import pipeline

# Specify the model name
model_name = "cognitivecomputations/Dolphin3.0-Llama3.1-8B"

# Load configuration and tokenizer
config = LlamaConfig.from_pretrained(model_name)
config._attn_implementation = "sdpa"
pprint(config)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with explicit config
model = LlamaForCausalLM.from_pretrained(
  model_name,
  config=config,
  torch_dtype=torch.bfloat16,
  device_map="auto",
)

text_generation = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
)

# Generate text
output = text_generation(
  "Here's a story about a boy and his dog:", max_length=50, num_return_sequences=1
)
print(output)
