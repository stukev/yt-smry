import os
import re
import torch
import yt_dlp
import psutil
import argparse
from transformers import AutoTokenizer
from pycaption import WebVTTReader, SRTWriter

# Minimum resource requirements
MIN_VRAM_GB = 5.0
MIN_RAM_GB = 16.0

def check_vram():
    """Returns (available_vram_GB, total_vram_GB)."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return round(free / 1e9, 2), round(total / 1e9, 2)
    return 0.0, 0.0

def check_ram():
    """Returns (available_ram_GB, total_ram_GB)."""
    info = psutil.virtual_memory()
    return round(info.available / 1e9, 2), round(info.total / 1e9, 2)

def download_subtitles(youtube_url, verbose):
    """Downloads YouTube subtitles and caches them."""
    cache_dir = "/tmp/youtube_subtitles"
    os.makedirs(cache_dir, exist_ok=True)

    video_id = youtube_url.split("v=")[-1]
    subtitle_cache = os.path.join(cache_dir, f"{video_id}.en.txt")

    if os.path.exists(subtitle_cache):
        if verbose:
            print("♻️ Using cached subtitles.")
        with open(subtitle_cache, "r", encoding="utf-8") as f:
            return f.read()

    if verbose:
        print("🔍 Downloading subtitles...")

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "outtmpl": f"{video_id}.%(ext)s",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    subtitle_file = f"{video_id}.en.vtt"
    if os.path.exists(subtitle_file):
        with open(subtitle_file, "r", encoding="utf-8") as f:
            subtitles = f.read()
        os.remove(subtitle_file)

        with open(subtitle_cache, "w", encoding="utf-8") as f:
            f.write(subtitles)

        return subtitles

    print("❌ No subtitles found.")
    return None

def clean_vtt(srt_content):
    """Removes timestamps, duplicates, and header from VTT subtitles."""
    if not srt_content:
        return ""

    lines = srt_content.split("\n")
    cleaned_lines = []
    seen_sentences = set()

    for line in lines:
        line = line.strip()
        if "-->" in line:
            continue
        if line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\s+", " ", line)

        if line and line.lower() not in seen_sentences:
            seen_sentences.add(line.lower())
            cleaned_lines.append(line)

    original_text = " ".join(cleaned_lines)
    cleaned_text = original_text.replace("WEBVTT Kind: captions Language: en", "").strip()
    return cleaned_text

def summarize_text(text, level, device_map, verbose, chunking=False):
    """Dispatches CPU or GPU summarizer."""
    if device_map == "cpu":
        from cpu_summarizer import cpu_summarize
        return cpu_summarize(text, summary_level=level, verbose=verbose, chunking=chunking)
    else:
        from gpu_summarizer import gpu_summarize
        return gpu_summarize(text, level, verbose=verbose)

def parse_arguments():
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description="YouTube Transcript Summarizer")

    parser.add_argument("url", nargs="?", help="YouTube video URL (Required in CLI mode)")
    parser.add_argument("--level", choices=["short", "medium", "long"], default="medium", help="Summary length")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--truncate", action="store_true", help="Use TextRank-based truncation")

    args = parser.parse_args()

    if (args.url is None) and (not args.truncate) and (not args.verbose):
        return None  # Interactive mode

    return args

def main():
    args = parse_arguments()

    if args is None:
        # Interactive mode
        print("\nℹ️ No arguments supplied. Entering interactive mode...\n")

        youtube_url = None
        while not youtube_url:
            youtube_url = input("🎥 Enter YouTube URL: ").strip()
            if not youtube_url:
                print("❌ URL cannot be empty. Please enter a valid YouTube URL.")

        summary_level = input("📜 Choose summary level (short, medium, long) [default: medium]: ").strip().lower()
        if summary_level not in ["short", "medium", "long"]:
            print("ℹ️ No valid level selected, defaulting to 'medium'.")
            summary_level = "medium"

        verbose_input = input("🔍 Enable verbose mode? (y/n) [default: n]: ").strip().lower()
        verbose = (verbose_input == "y")

        chunk_input = input("🔀 Use chunking for summarization? (c=chunking / t=truncate) [default: c]: ").strip().lower()
        if chunk_input == "t":
            chunking = False
        else:
            chunking = True

    else:
        # CLI mode
        youtube_url = args.url
        summary_level = args.level
        verbose = args.verbose
        chunking = not args.truncate

    vram_available, vram_total = check_vram()
    ram_available, ram_total = check_ram()

    if verbose:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        print(f"🖥️ GPU Detected: {gpu_name}")
        print(f"💾 VRAM Available: {vram_available}GB / {vram_total}GB")
        print(f"🧠 RAM Available: {ram_available}GB / {ram_total}GB")

    if vram_available < MIN_VRAM_GB:
        if verbose:
            print(f"⚠ Warning: Less than {MIN_VRAM_GB}GB VRAM detected.")
        if ram_available >= MIN_RAM_GB:
            if verbose:
                print("🖥️ Switching to CPU mode.")
            device_map = "cpu"
        else:
            print(f"❌ Insufficient RAM for CPU execution. Need at least {MIN_RAM_GB}GB.")
            exit()
    else:
        device_map = "auto"

    subtitles = download_subtitles(youtube_url, verbose)
    cleaned_text = clean_vtt(subtitles)

    summary = summarize_text(cleaned_text, summary_level, device_map, verbose, chunking)
    print("\n📢 **SUMMARY:**\n")
    print(summary)

if __name__ == "__main__":
    main()
