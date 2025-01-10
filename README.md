# Freeze-Attn: LLaMA Attention State Caching

A Python library that implements an efficient caching mechanism for LLaMA model attention states, optimizing the processing of long prefixes like book text or documents.

## Purpose

This library optimizes the processing of long text sequences by:

- Running prefix text through the model once to build attention state
- Capturing and saving that state to disk
- Loading the saved state later to continue processing from that point
- Supporting multiple loads of the same state for different continuations

## Key Components

- `save_cache_to_disk`: Saves a DynamicCache instance (containing key/value attention states) to disk
- `load_cache_from_disk`: Loads a saved DynamicCache from disk
- `process_prefix`: Takes a prefix text, processes it through the model, and saves its attention state
- `generate_with_prefix`: Loads a saved prefix state and generates continuations from it

## Benefits

- Avoids reprocessing the same prefix text repeatedly
- Improves performance for applications like question-answering over long documents
- Allows multiple different continuations from the same prefix point
- Reduces computational overhead for repetitive tasks

## Example Usage

```python
# Process and cache a story prefix
prefix = "Here's a story about a brave knight named Sir Roland..."
process_prefix(model, tokenizer, prefix, "prefix_cache.pt")

# Generate multiple different continuations from the same prefix
continuations = [
    "The knight was known for",
    "One day, while riding his horse",
    "In the castle, there was"
]
```

## Use Cases

This library is particularly useful for applications where you need to:

- Process long documents efficiently
- Generate multiple different continuations from the same starting point
- Avoid redundant computation of the same prefix text
- Implement efficient question-answering systems over long documents

## Technical Details

The library uses the Hugging Face Transformers library and specifically works with LLaMA models, leveraging their attention caching mechanisms. It ensures complete state storage by capturing:

- `key_cache` and `value_cache`: The attention KV states for each layer
- `_seen_tokens`: Number of tokens processed
- Position information
- Attention mask state

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- LLaMA model (tested with Dolphin3.0-Llama3.1-8B)

## License

MIT License. See [LICENSE](LICENSE) for details.
