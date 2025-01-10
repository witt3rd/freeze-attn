# Freeze-Attn: LLaMA 3.x Attention State Caching

A Python library that enables explicit client-side control over LLaMA 3.x model attention states. Unlike internal prefix caching optimizations found in systems like vLLM, sglang, or commercial APIs, Freeze-Attn allows clients to explicitly identify, save, and reuse attention states from specific prefixes.

This approach provides several unique capabilities:

- Save attention states to persistent storage for later reuse across sessions
- Share cached states between different clients or applications
- Explicitly version and manage different prefix states
- Control exactly when and how prefix states are reused
- Avoid transmitting large prefix texts repeatedly in distributed systems

## Core Functions

- `save_cache_to_disk`: Saves attention state (DynamicCache) with additional context like input IDs and position information
- `load_cache_from_disk`: Loads saved attention state and context for reuse
- `process_prefix`: Processes initial text through the model and saves its attention state
- `generate_with_prefix`: Generates continuations using a loaded attention state
- `run_generation_test`: Compares performance between cached and uncached text generation

## Example Usage

```python
# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/Dolphin3.0-Llama3.1-8B")
model = LlamaForCausalLM.from_pretrained(
    "cognitivecomputations/Dolphin3.0-Llama3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Simple story generation test
prefix = "Here's a story about a brave knight named Sir Roland. He lived in a castle high up in the mountains."
continuations = [
    "The knight was known for",
    "One day, while riding his horse",
    "In the castle, there was"
]
test_simple_story(model, tokenizer, "prefix_cache.pt")

# Book Q&A test with token limit
test_book_qa(
    model,
    tokenizer,
    "book.txt",
    "book_cache.pt",
    max_prefix_tokens=2000
)
```

## Use Cases

### Distributed Systems

- Save prefix states on a central server, allowing thin clients to generate completions without transmitting or processing the full prefix
- Share common prefix states (e.g., system prompts, context) across multiple services
- Version control different prefix states for A/B testing or gradual rollout

### Content Generation

- Cache the attention state for a story's beginning or character description, then generate multiple alternative continuations
- Save states at key plot points to explore different narrative branches
- Maintain consistent context across multiple generation sessions

### Document Analysis

- Process and cache a large document's attention state once, then run multiple queries against it
- Save states for different sections of a document to enable focused analysis
- Maintain separate cached states for different documents in a corpus

### Interactive Applications

- Cache the conversation history state in chat applications
- Save game world descriptions in text-based games
- Maintain persistent context in long-running interactive sessions

### Mental Frame Priming

- Cache attention states that represent specific "mental dispositions" of the model after careful dialog-based priming
- Create a library of different "personalities" or "expert modes" through saved attention states
- Achieve a form of lightweight, reversible "fine-tuning" without modifying model weights
- Experiment with different prompting strategies and save the most effective ones as reusable states
- Switch between different cognitive frames instantly by loading different cached states

This approach offers an alternative to traditional fine-tuning:

- No modification of model weights required
- Instantly reversible - just load a different state
- Can maintain multiple different "tunings" simultaneously
- Experimental and iterative - easily try different priming approaches
- More transparent - the priming dialog is explicit and reviewable

### Summary

These use cases particularly shine in scenarios where:

- The same prefix needs to be reused across different sessions or clients
- Prefix processing is computationally expensive or time-consuming
- Network bandwidth for transmitting large prefixes is constrained
- Fine-grained control over state management is required

## Performance

The library includes built-in performance comparison, showing speed improvements when using cached attention states versus processing the full prefix each time.

## Technical Details

- Uses PyTorch for tensor operations
- Integrates with Hugging Face's DynamicCache for attention state management
- Handles position IDs and attention masks for proper continuation
- Supports truncation for long texts via `max_prefix_tokens`
- Includes detailed debug logging for cache statistics and generation process

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- LLaMA model (tested with Dolphin3.0-Llama3.1-8B)

## License

MIT License. See [LICENSE](LICENSE) for details.
