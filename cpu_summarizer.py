import nltk
import torch
import psutil
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# BART model info
MODEL_NAME = "facebook/bart-large-cnn"

# Lazy-loaded globals
bart_tokenizer = None
bart_model = None

def ensure_nltk():
    """Checks for NLTK resources and downloads if missing."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("🔍 Missing NLTK 'punkt'. Downloading now...")
        nltk.download('punkt')

def check_available_ram():
    """Returns available system RAM in GB."""
    return round(psutil.virtual_memory().available / 1e9, 2)

def get_sentence_limit(verbose=False):
    """Returns a TextRank sentence limit and a BART max token limit based on RAM."""
    available_ram = check_available_ram()

    if available_ram < 8.0:
        sentence_limit = 5
        bart_max_tokens = 512
    elif available_ram < 16.0:
        sentence_limit = 7
        bart_max_tokens = 768
    else:
        sentence_limit = 10
        bart_max_tokens = 1024

    if verbose:
        print(f"🧠 Available RAM: {available_ram}GB. sentence_limit={sentence_limit}, bart_max_tokens={bart_max_tokens}")
    return sentence_limit, bart_max_tokens

def load_bart():
    """Initializes BART model/tokenizer if not already loaded."""
    global bart_tokenizer, bart_model
    if bart_tokenizer is None or bart_model is None:
        print("🔹 Loading facebook/bart-large-cnn for CPU summarization...")
        bart_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cpu")
        bart_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def extractive_summary(text, num_sentences, verbose=False):
    """Uses TextRank to extract key sentences."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    extracted_text = " ".join(str(sentence) for sentence in summary)

    if verbose:
        token_count = len(extracted_text.split())
        print(f"\n🔎 Extracted TextRank Summary: {token_count} tokens.")
    return extracted_text

def abstractive_summary(text, min_len=30, max_len=60, length_penalty=1.0, no_repeat_ngram_size=3, verbose=False):
    """Runs BART with specified constraints for min/max length."""
    load_bart()

    if verbose:
        print(f"\n🔎 Abstractive Summary: min_len={min_len}, max_len={max_len}, length_penalty={length_penalty}")

    inputs = bart_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cpu")

    with torch.no_grad():
        output = bart_model.generate(
            **inputs,
            min_length=min_len,
            max_length=max_len,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=4,
            early_stopping=True
        )

    summary = bart_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    if verbose:
        out_tokens = len(bart_tokenizer(summary, return_tensors="pt")["input_ids"][0])
        print(f"🔎 Final summary has {out_tokens} tokens.")
    return summary

def chunk_text(text, chunk_size, verbose=False):
    """Splits text into chunks of up to chunk_size tokens using the BART tokenizer."""
    load_bart()
    tokens = bart_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        subset = tokens[i:i + chunk_size]
        chunk_str = bart_tokenizer.decode(subset, skip_special_tokens=True)
        chunks.append(chunk_str)

    if verbose:
        print(f"Split into {len(chunks)} chunks, max {chunk_size} tokens each.")
    return chunks

def chunked_bart_summarize(text, max_tokens=1024, verbose=False):
    """Summarizes large text by chunking, partial summarizing, and merging."""
    chunks = chunk_text(text, chunk_size=max_tokens, verbose=verbose)
    partial_summaries = []

    for idx, chunk in enumerate(chunks, start=1):
        if verbose:
            print(f"📝 Summarizing chunk {idx}/{len(chunks)}")
        partial = abstractive_summary(chunk, min_len=30, max_len=300, verbose=verbose)
        partial_summaries.append(partial)

    combined = " ".join(partial_summaries)

    load_bart()
    total_tokens = len(bart_tokenizer.encode(combined, add_special_tokens=False))
    if total_tokens > max_tokens:
        if verbose:
            print("Combined partial summary exceeds max_tokens, second pass to reduce size.")
        combined = abstractive_summary(combined, min_len=50, max_len=500, verbose=verbose)

    return combined

def cpu_summarize(text, summary_level="medium", verbose=False, chunking=False):
    """
    CPU pipeline:
    1) If chunking=True, do chunk-based summarization, then a final pass.
    2) If chunking=False, do TextRank -> BART.
    """
    if verbose:
        print(f"🖥️ CPU Summarization: chunking={chunking}, level={summary_level}")

    ensure_nltk()
    load_bart()

    sys_limit, sys_bart_max = get_sentence_limit(verbose)

    # Define final pass constraints (short/medium/long)
    summary_params = {
        "short":  {"min_len": 30,  "max_len": 80,  "length_penalty": 0.9},
        "medium": {"min_len": 80,  "max_len": 160, "length_penalty": 1.0},
        "long":   {"min_len": 160, "max_len": 240, "length_penalty": 1.1},
    }
    user_choice = summary_params.get(summary_level, summary_params["medium"])

    if chunking:
        if verbose:
            print(f"Chunk-based pass with {sys_bart_max} tokens per chunk.")
        chunked_result = chunked_bart_summarize(text, max_tokens=sys_bart_max, verbose=verbose)

        if verbose:
            print("Final pass with user summary preferences.")
        final_summary = abstractive_summary(
            chunked_result,
            min_len=user_choice["min_len"],
            max_len=user_choice["max_len"],
            length_penalty=user_choice["length_penalty"],
            verbose=verbose
        )
        return final_summary
    else:
        if verbose:
            print("TextRank -> Abstractive summarization.")
        extracted = extractive_summary(text, num_sentences=sys_limit, verbose=verbose)
        if not extracted:
            return "ERROR: No valid text available for summarization."

        final_summary = abstractive_summary(
            extracted,
            min_len=user_choice["min_len"],
            max_len=user_choice["max_len"],
            length_penalty=user_choice["length_penalty"],
            verbose=verbose
        )
        return final_summary
