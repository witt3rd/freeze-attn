import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

model_name = "cognitivecomputations/Dolphin3.0-Llama3.1-8B"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.bfloat16)


def generate_text(
    prompt: str, model, tokenizer, max_length: int = 200, temperature: float = 0.7
):
    """Generate text from a prompt using the model.

    Args:
    ----
        prompt (str): The input prompt
        model: The language model
        tokenizer: The tokenizer
        max_length (int): Maximum length of generated text
        temperature (float): Controls randomness in generation. Higher values (e.g., 1.0) make output more random,
                           lower values (e.g., 0.1) make it more deterministic
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        device
    )

    # Generate text
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        top_p=0.95,
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    # Load the model configuration first
    print(f"Loading model configuration from {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    print("Model configuration:", config)

    # Load the model and tokenizer
    print(f"Loading model {model_name}...")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        config=config,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    print("Model and tokenizer loaded successfully!")

    # Example prompt
    prompt = "Write a short poem about artificial intelligence:"
    print("\nPrompt:", prompt)
    print("\nGenerating response...\n")

    response = generate_text(prompt, model, tokenizer)
    print("Generated text:", response)

    return model, tokenizer, config


if __name__ == "__main__":
    main()
