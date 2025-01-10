"""Implements a caching mechanism for LLaMA model attention states."""

from collections.abc import Callable

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer

MODEL_NAME = "cognitivecomputations/Dolphin3.0-Llama3.1-8B"


def save_cache_to_disk(
  cache: DynamicCache,
  path: str,
  *,
  input_ids: torch.Tensor | None = None,
  start_pos: int | None = None,
  attention_mask: torch.Tensor | None = None,
) -> None:
  """Save a DynamicCache instance to disk along with additional state information.

  Args:
  ----
      cache: The DynamicCache instance containing key/value states
      path: Path to save the state to
      input_ids: The tokenized input that generated this cache state
      start_pos: The position counter for where processing ended
      attention_mask: The attention mask used for this sequence

  """
  # Extract the tensors we need to save
  save_data = {
    "key_cache": cache.key_cache,
    "value_cache": cache.value_cache,
    "_seen_tokens": cache._seen_tokens,
    # Additional state information
    "input_ids": input_ids,
    "start_pos": start_pos,
    "attention_mask": attention_mask,
  }
  torch.save(save_data, path)


def load_cache_from_disk(path: str) -> tuple[DynamicCache, dict]:
  """Load a DynamicCache instance and additional state from disk."""
  save_data = torch.load(path)

  cache = DynamicCache()
  # We need to update the cache layer by layer to properly track positions
  for layer_idx, (key_states, value_states) in enumerate(
    zip(save_data["key_cache"], save_data["value_cache"], strict=False)
  ):
    cache.update(key_states, value_states, layer_idx)

  # print("\nDEBUG Cache State:")
  # print(f"- _seen_tokens: {cache._seen_tokens}")
  # print(f"- key_cache shape: {cache.key_cache[0].shape}")
  # print(f"- value_cache shape: {cache.value_cache[0].shape}")

  # Extract additional state information
  additional_state = {
    "input_ids": save_data.get("input_ids"),
    "start_pos": save_data.get("start_pos"),
    "attention_mask": save_data.get("attention_mask"),
  }
  # print("\nDEBUG Additional State:")
  # print(
  #   f"- input_ids shape: {additional_state['input_ids'].shape if additional_state['input_ids'] is not None else None}"
  # )
  # print(f"- start_pos: {additional_state['start_pos']}")
  # print(
  #   f"- attention_mask shape: {additional_state['attention_mask'].shape if additional_state['attention_mask'] is not None else None}"
  # )

  return cache, additional_state


def process_prefix(
  model: LlamaForCausalLM,
  tokenizer: LlamaTokenizer,
  prefix_text: str,
  cache_path: str,
) -> None:
  """Process a prefix text and save its attention state."""
  print(f"\nProcessing prefix: {prefix_text}")

  # Tokenize and process the prefix
  inputs = tokenizer(prefix_text, return_tensors="pt", padding=True).to(model.device)
  start_pos = 0  # Initial position
  seq_len = inputs.input_ids.shape[1]

  # Run forward pass to build cache
  outputs = model(**inputs, use_cache=True)

  # Convert tuple past_key_values to DynamicCache
  if outputs.past_key_values is not None:
    # Always convert to DynamicCache
    cache = DynamicCache.from_legacy_cache(outputs.past_key_values)
    save_cache_to_disk(
      cache,
      cache_path,
      input_ids=inputs.input_ids,
      start_pos=start_pos + seq_len,  # Next position after this sequence
      attention_mask=inputs.attention_mask,
    )
    print(f"Saved cache to {cache_path}")
  else:
    print("No cache was generated!")


def generate_with_prefix(
  model: LlamaForCausalLM,
  tokenizer: LlamaTokenizer,
  continuation_text: str,
  cache_path: str,
  max_new_tokens: int = 50,
) -> str:
  """Generate text using a saved prefix state."""
  print(f"\nGenerating with continuation: {continuation_text}")

  # Load the cached state
  cache, state = load_cache_from_disk(cache_path)

  # Debug: Print cache stats before generation
  print("\nCache Statistics:")
  print(f"- Cache sequence length: {cache.get_seq_length()}")
  print(f"- Cache seen tokens: {cache._seen_tokens}")
  print(f"- Number of layers in cache: {len(cache.key_cache)}")
  print(f"- Shape of first layer cache: {cache.key_cache[0].shape}")

  # Process the continuation
  new_inputs = tokenizer(continuation_text, return_tensors="pt", padding=True).to(
    model.device
  )
  input_ids = new_inputs.input_ids

  # Debug: Print input stats
  print("\nInput Statistics:")
  print(f"- New input length: {input_ids.shape[1]}")
  print(f"- Starting position: {state['start_pos']}")

  # Move cache to device
  for i in range(len(cache.key_cache)):
    cache.key_cache[i] = cache.key_cache[i].to(model.device)
    cache.value_cache[i] = cache.value_cache[i].to(model.device)

  # Create attention mask that includes both prefix and new tokens
  prefix_len = state["attention_mask"].shape[1]

  # Pad the input tokens to extend beyond cache length
  total_length = cache._seen_tokens + input_ids.shape[1]
  pad_length = total_length - input_ids.shape[1]
  if pad_length > 0:
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padding = torch.full(
      (1, pad_length), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
    )
    input_ids = torch.cat([padding, input_ids], dim=1)
    print("\nPadding Statistics:")
    print(f"- Pad length: {pad_length}")
    print(f"- Final input shape: {input_ids.shape}")

  full_attention_mask = torch.ones(
    (1, prefix_len + input_ids.shape[1]),
    dtype=torch.bool,
    device=model.device,
  )

  # Create position IDs that continue from where the prefix left off
  position_ids = torch.arange(
    state["start_pos"],
    state["start_pos"] + input_ids.shape[1],
    dtype=torch.long,
    device=model.device,
  ).unsqueeze(0)

  print("\nPosition Statistics:")
  print(f"- Position IDs shape: {position_ids.shape}")
  print(f"- Position IDs range: {position_ids[0][0]} to {position_ids[0][-1]}")

  # Generate with the loaded cache
  outputs = model.generate(
    input_ids,
    attention_mask=full_attention_mask,
    position_ids=position_ids,
    past_key_values=cache,
    max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    use_cache=True,
  )

  # Debug: Print final stats
  print("\nGeneration Complete")
  print(f"- Output length: {outputs.shape[1]}")

  return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_generation_test(
  model: LlamaForCausalLM,
  tokenizer: LlamaTokenizer,
  prefix_text: str,
  prompts: list[str],
  cache_path: str,
  *,
  max_new_tokens: int = 50,
  max_prefix_tokens: int | None = None,
  process_result: Callable[[str], str] = lambda x: x,
) -> None:
  """Run a generation test with and without cache, measuring performance.

  Args:
  ----
      model: The LLaMA model to use
      tokenizer: The tokenizer to use
      prefix_text: The prefix text to cache
      prompts: List of prompts to test with
      cache_path: Path to save/load the cache
      max_new_tokens: Maximum number of new tokens to generate
      max_prefix_tokens: Optional limit on prefix token length
      process_result: Optional function to process the generated result

  """
  # Process and cache the prefix
  if max_prefix_tokens:
    prefix_inputs = tokenizer(
      prefix_text, return_tensors="pt", truncation=True, max_length=max_prefix_tokens
    )
    prefix_text = tokenizer.decode(prefix_inputs.input_ids[0], skip_special_tokens=True)
    print(f"\nProcessing prefix (first {max_prefix_tokens} tokens)")
    print(f"Prefix length: {len(prefix_text)} characters")

  process_prefix(model, tokenizer, prefix_text, cache_path)

  print("\n=== Testing with Cache ===")
  import time

  start_time = time.time()
  for prompt in prompts:
    result = generate_with_prefix(
      model,
      tokenizer,
      prompt,
      cache_path,
      max_new_tokens=max_new_tokens,
    )
    print(f"\nPrompt: {prompt}")
    print(f"Result: {process_result(result)}")
  cached_time = time.time() - start_time
  print(f"\nTime with cache: {cached_time:.2f} seconds")

  print("\n=== Testing without Cache ===")
  start_time = time.time()
  for prompt in prompts:
    full_prompt = f"{prefix_text}\n\n{prompt}"
    inputs = tokenizer(
      full_prompt,
      return_tensors="pt",
      truncation=True,
      max_length=(max_prefix_tokens + 100 if max_prefix_tokens else None),
    ).to(model.device)
    outputs = model.generate(
      inputs.input_ids,
      attention_mask=inputs.attention_mask,
      max_new_tokens=max_new_tokens,
      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Result: {process_result(result)}")
  uncached_time = time.time() - start_time
  print(f"\nTime without cache: {uncached_time:.2f} seconds")
  print(f"Speed improvement: {uncached_time/cached_time:.1f}x faster with cache")


def test_simple_story(
  model: LlamaForCausalLM,
  tokenizer: LlamaTokenizer,
  cache_path: str,
) -> None:
  """Test simple story generation with and without cache."""
  prefix = "Here's a story about a brave knight named Sir Roland. He lived in a castle high up in the mountains."
  continuations = [
    "The knight was known for",
    "One day, while riding his horse",
    "In the castle, there was",
  ]

  run_generation_test(
    model,
    tokenizer,
    prefix,
    continuations,
    cache_path,
    max_new_tokens=50,
  )


def test_book_qa(
  model: LlamaForCausalLM,
  tokenizer: LlamaTokenizer,
  book_path: str,
  cache_path: str,
  max_prefix_tokens: int = 2000,  # Limit prefix length to avoid memory issues
) -> None:
  """Test question answering using a book as prefix context."""
  # Read the book
  with open(book_path, encoding="utf-8") as f:
    book_text = f.read()

  questions = [
    "What happens in the first paragraph?",
    "Who are the main characters mentioned so far?",
    "What is the setting of this story?",
    "Summarize the key events described in this text.",
  ]

  # Format questions as prompts
  prompts = [f"Based on the text, please answer: {q}\nAnswer:" for q in questions]

  def extract_answer(result: str) -> str:
    # Extract just the answer part for book QA results
    return result.split("Answer:")[-1].strip()

  run_generation_test(
    model,
    tokenizer,
    book_text,
    prompts,
    cache_path,
    max_new_tokens=100,
    max_prefix_tokens=max_prefix_tokens,
    process_result=extract_answer,
  )


def main():
  # Initialize model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
  )

  # Run both tests
  print("\n=== Running Simple Story Test ===")
  test_simple_story(model, tokenizer, "prefix_cache.pt")

  print("\n=== Running Book QA Test ===")
  test_book_qa(
    model,
    tokenizer,
    "book.txt",  # You'll need to provide this file
    "book_cache.pt",
    max_prefix_tokens=2000,
  )


if __name__ == "__main__":
  main()
