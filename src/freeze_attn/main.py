"""Implements explicit client-side control over LLaMA model attention states.

This module provides functionality to externalize and reuse LLaMA model attention states,
enabling several unique capabilities beyond standard prefix caching:

- Persistent storage of attention states for cross-session reuse
- Sharing cached states between different clients/applications
- Explicit versioning and management of prefix states
- Fine-grained control over state reuse
- Reduced bandwidth needs in distributed systems by avoiding prefix retransmission

The implementation focuses on the LLaMA architecture but the concepts could be extended
to other transformer-based models that expose their attention mechanisms.
"""

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
) -> None:
  """Persist attention state and context for later reuse.

  Parameters
  ----------
  cache : DynamicCache
      The DynamicCache instance containing key/value states
  path : str
      Path where the state will be saved
  input_ids : torch.Tensor or None, optional
      Tokenized input that generated this cache state

  Notes
  -----
  The saved state can be used across sessions or shared between different clients,
  enabling distributed systems to avoid retransmitting large prefixes.

  """
  save_data = {
    "key_cache": cache.key_cache,
    "value_cache": cache.value_cache,
    "_seen_tokens": cache._seen_tokens,
    "input_ids": input_ids,
  }
  torch.save(save_data, path)


def load_cache_from_disk(path: str) -> tuple[DynamicCache, dict]:
  """Restore a previously saved attention state and its context.

  Parameters
  ----------
  path : str
      Path to the saved cache file

  Returns
  -------
  tuple[DynamicCache, dict]
      - DynamicCache
          The restored DynamicCache instance
      - dict
          Additional state information needed for generation

  Notes
  -----
  This function reconstructs both the attention cache and the associated metadata
  needed for proper state continuation. The cache must be restored layer by layer
  to maintain correct position tracking.

  """
  save_data = torch.load(path, weights_only=True)

  cache = DynamicCache()
  # Reconstruct cache layer by layer to maintain position tracking
  for layer_idx, (key_states, value_states) in enumerate(
    zip(save_data["key_cache"], save_data["value_cache"], strict=False)
  ):
    cache.update(key_states, value_states, layer_idx)

  additional_state = {
    "input_ids": save_data.get("input_ids"),
  }

  return cache, additional_state


def process_prefix(
  model: LlamaForCausalLM,
  tokenizer: LlamaTokenizer,
  prefix_text: str,
  cache_path: str,
) -> None:
  """Process and cache the attention state for a given prefix text.

  Parameters
  ----------
  model : LlamaForCausalLM
      The LLaMA model instance
  tokenizer : LlamaTokenizer
      Associated tokenizer
  prefix_text : str
      Text to process and cache
  cache_path : str
      Path where to save the cached state

  Notes
  -----
  This function represents the first phase of the two-phase generation process:
  1. Process the prefix once and cache its attention state
  2. Reuse that state for multiple different continuations

  This approach is particularly valuable when:
  - The same prefix needs to be reused multiple times
  - The prefix processing is computationally expensive
  - Network bandwidth for transmitting prefixes is limited

  """
  print(f"\nProcessing prefix: {prefix_text}")

  inputs = tokenizer(prefix_text, return_tensors="pt", padding=True).to(model.device)
  outputs = model(**inputs, use_cache=True)

  if outputs.past_key_values is not None:
    cache = DynamicCache.from_legacy_cache(outputs.past_key_values)
    save_cache_to_disk(
      cache,
      cache_path,
      input_ids=inputs.input_ids,
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
  """Generate text using a previously cached attention state.

  Parameters
  ----------
  model : LlamaForCausalLM
      The LLaMA model instance
  tokenizer : LlamaTokenizer
      Associated tokenizer
  continuation_text : str
      Text to continue from the cached state
  cache_path : str
      Path to the saved attention state
  max_new_tokens : int, optional
      Maximum number of tokens to generate, by default 50

  Returns
  -------
  str
      The generated text incorporating both prefix and continuation

  Notes
  -----
  This function implements the second phase of the two-phase generation process,
  where a cached prefix state is reused for generation. The key challenge is
  ensuring proper alignment between the cached state and the new input, including:
  - Position IDs that continue from where the prefix left off
  - Attention masks that span both cached and new tokens
  - Proper padding for alignment with cache dimensions

  """
  print(f"\nGenerating with continuation: {continuation_text}")

  import time

  load_start = time.time()
  cache, state = load_cache_from_disk(cache_path)
  load_time = time.time() - load_start

  print("\nCache Statistics:")
  print(f"- Load time: {load_time:.4f} seconds")
  print(f"- Cache sequence length: {cache.get_seq_length()}")
  print(f"- Cache seen tokens: {cache._seen_tokens}")
  print(f"- Number of layers in cache: {len(cache.key_cache)}")
  print(f"- Shape of first layer cache: {cache.key_cache[0].shape}")

  device_start = time.time()
  # Ensure cache is on the correct device
  for i in range(len(cache.key_cache)):
    cache.key_cache[i] = cache.key_cache[i].to(model.device)
    cache.value_cache[i] = cache.value_cache[i].to(model.device)
  device_time = time.time() - device_start
  print(f"- Device transfer time: {device_time:.4f} seconds")
  print(f"- Total cache prep time: {(load_time + device_time):.4f} seconds")

  new_inputs = tokenizer(continuation_text, return_tensors="pt", padding=True).to(
    model.device
  )
  input_ids = new_inputs.input_ids

  print("\nInput Statistics:")
  print(f"- New input length: {input_ids.shape[1]}")
  print(f"- Starting position: {cache._seen_tokens}")

  # Handle alignment between cache and new input
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

  # Create attention mask spanning both prefix and new tokens
  full_attention_mask = torch.ones(
    (1, cache._seen_tokens + input_ids.shape[1]),
    dtype=torch.bool,
    device=model.device,
  )

  # Continue position IDs from where prefix ended
  position_ids = torch.arange(
    cache._seen_tokens,  # Start from number of tokens seen
    cache._seen_tokens + input_ids.shape[1],  # Continue for new input length
    dtype=torch.long,
    device=model.device,
  ).unsqueeze(0)

  print("\nPosition Statistics:")
  print(f"- Position IDs shape: {position_ids.shape}")
  print(f"- Position IDs range: {position_ids[0][0]} to {position_ids[0][-1]}")

  outputs = model.generate(
    input_ids,
    attention_mask=full_attention_mask,
    position_ids=position_ids,
    past_key_values=cache,
    max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    use_cache=True,
  )

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
  """Compare performance between cached and uncached text generation.

  This function provides empirical validation of the performance benefits
  of attention state caching. It runs the same generation tasks both with
  and without caching, measuring the time difference.

  The test helps demonstrate the value proposition of caching in scenarios
  where the same prefix is used repeatedly, as the time savings can be
  substantial.

  Args:
  ----
      model: The LLaMA model instance
      tokenizer: Associated tokenizer
      prefix_text: The common prefix to cache
      prompts: Different continuations to test
      cache_path: Where to save the cached state
      max_new_tokens: Maximum tokens to generate per continuation
      max_prefix_tokens: Optional limit on prefix length
      process_result: Optional post-processing of generated text

  """
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

  # Calculate and report the speed difference
  if cached_time < uncached_time:
    speedup = uncached_time / cached_time
    print(f"Speed improvement: {speedup:.4f}x faster with cache")
  else:
    slowdown = cached_time / uncached_time
    print(f"Speed regression: {slowdown:.4f}x slower with cache")


def test_simple_story(
  model: LlamaForCausalLM,
  tokenizer: LlamaTokenizer,
  cache_path: str,
) -> None:
  """Demonstrate attention caching with a simple story generation example.

  This test shows how the same story opening can be efficiently reused to
  generate multiple different continuations, illustrating the basic concept
  of attention state reuse.
  """
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
  max_prefix_tokens: int = 2000,
) -> None:
  """Demonstrate attention caching for document question-answering.

  This test shows a practical application: processing a long document once
  and caching its state, then efficiently answering multiple questions about
  the document without having to reprocess it each time.

  The max_prefix_tokens parameter helps manage memory usage with long documents
  while still maintaining enough context for meaningful QA.
  """
  with open(book_path, encoding="utf-8") as f:
    book_text = f.read()

  questions = [
    "What happens in the first paragraph?",
    "Who are the main characters mentioned so far?",
    "What is the setting of this story?",
    "Summarize the key events described in this text.",
  ]

  prompts = [f"Based on the text, please answer: {q}\nAnswer:" for q in questions]

  def extract_answer(result: str) -> str:
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
  """Run demonstration tests of attention state caching capabilities."""
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
  )

  print("\n=== Running Simple Story Test ===")
  test_simple_story(model, tokenizer, "prefix_cache.pt")

  print("\n=== Running Book QA Test ===")
  test_book_qa(
    model,
    tokenizer,
    "book.txt",
    "book_cache.pt",
    max_prefix_tokens=2000,
  )


if __name__ == "__main__":
  main()
