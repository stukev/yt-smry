import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Model info
SELECTED_MODEL = "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit"

def gpu_summarize(text, summary_level, verbose=False):
    """Summarizes text on GPU with Mistral 7B (4-bit)."""
    if verbose:
        print(f"📊 Summarizing text on GPU (Level: {summary_level})...")

    print(f"🔹 Loading {SELECTED_MODEL} for GPU summarization...")
    model = AutoModelForCausalLM.from_pretrained(
        SELECTED_MODEL,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(SELECTED_MODEL)

    summary_lengths = {"short": 500, "medium": 1000, "long": 2000}
    max_new_tokens = summary_lengths.get(summary_level, 1000)

    tokenized_input = tokenizer(text, return_tensors="pt")
    input_length = tokenized_input["input_ids"].shape[1]

    max_input_tokens = 3500
    print(f"📏 Original Transcript Size: {input_length} tokens")

    if input_length > max_input_tokens:
        print(f"⚠ Warning: Input too long, truncating to {max_input_tokens} tokens...")
        text = tokenizer.decode(tokenized_input["input_ids"][0, :max_input_tokens], skip_special_tokens=True)

    prompt = f"""You are a professional summarizer.
- Summarize this in a third-person perspective.
- Do not copy phrases from the transcript.
- Keep it concise and only include key points.
- The summary must be significantly shorter than the original text.
- Return the summary in a natural paragraph format with no bullet points or lists.

Transcript:
{text}

Generate a clear and concise summary:
"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return summary.split("Generate a clear and concise summary:")[-1].strip()
