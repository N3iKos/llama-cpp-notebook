import os
import re
import sys
import json
import time
import shutil
import urllib.request
from typing import Optional, Dict, Any, Callable
from .config import RuntimeProfile, save_state
from .ui import live_print, clear
from . import shell

def parse_aria2_line(line: str, filename: str) -> None:
    """Parses raw aria2c progress line and renders a beautiful single-panel progress card."""
    # Pattern matching: [#3a55ad 1.2GiB/4.3GiB(27%) CN:16 DL:12.5MiB ETA:4m12s]
    # We match the ID, downloaded size, total size, percent, connections, speed, and optional ETA
    match = re.search(
        r'\[#([0-9a-fA-F]+)\s+([0-9.]+[a-zA-Z]+)/([0-9.]+[a-zA-Z]+)\(([0-9]+)%\)\s+CN:(\d+)\s+DL:([0-9.]+[a-zA-Z]+)(?:\s+ETA:([0-9a-zA-Z]+))?\]',
        line
    )
    if match:
        gid, downloaded, total, percent, cn, speed, eta = match.groups()
        eta_str = f" ETA:{eta}" if eta else ""
        
        # Build clean progress lines
        progress_card = [
            f"Downloading: {filename}",
            f"#{gid} {downloaded}/{total}({percent}%) CN:{cn} DL:{speed}{eta_str}"
        ]
        # Refresh the single panel
        live_print(progress_card, title="GGUF Downloader", force=False, min_interval=0.25)
    elif "Download complete" in line or "is already fully downloaded" in line:
        live_print([f"Download Completed: {filename}"], title="GGUF Downloader", force=True)


def download_file_aria2(
    url: str,
    output_dir: str,
    filename: str,
    connections: int = 16
) -> str:
    """Downloads a single file using aria2c with live progress updates."""
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, filename)
    
    # If already exists and fully completed (aria2 handles resume automatically, but we can double check)
    if os.path.exists(target_path) and not os.path.exists(target_path + ".aria2"):
        live_print([f"File already downloaded: {filename}"], title="GGUF Downloader", force=True)
        return target_path

    # Construct aria2c command
    cmd = [
        "aria2c",
        "-c",                       # Resume partial downloads
        f"-x{connections}",         # Max connections per server
        f"-s{connections}",         # Segmented download count
        "-k1M",                     # 1MB split size
        "--summary-interval=1",     # Print summary every 1 second
        "--console-log-level=warn", # Keep logs clean
        "-d", output_dir,           # Output directory
        "-o", filename,             # Output filename
        url
    ]
    
    # Custom parser function that binds the current filename
    def line_parser(line: str) -> None:
        parse_aria2_line(line, filename)
        
    live_print([f"Starting download of {filename} via aria2c..."], title="GGUF Downloader", force=True)
    
    ret_code = shell.stream(cmd, parser=line_parser)
    
    if ret_code != 0:
        # Check if the file got downloaded regardless, or raise
        if not os.path.exists(target_path) or os.path.exists(target_path + ".aria2"):
            raise RuntimeError(f"aria2c failed downloading {filename} with exit code {ret_code}.")
            
    return target_path


def download_file_fallback(url: str, output_dir: str, filename: str) -> str:
    """Fallback segmented/buffer downloader using urllib standard library when aria2c is not available."""
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, filename)
    
    if os.path.exists(target_path):
        live_print([f"File already downloaded: {filename}"], title="GGUF Downloader", force=True)
        return target_path
        
    live_print([f"aria2c unavailable, using fallback downloader for {filename}..."], title="GGUF Downloader", force=True)
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB blocks
        downloaded = 0
        
        start_time = time.time()
        
        with open(target_path, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                downloaded += len(buffer)
                
                # Format progress
                percent = int(downloaded * 100 / total_size) if total_size else 0
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                
                elapsed = time.time() - start_time
                speed = downloaded_mb / elapsed if elapsed > 0 else 0
                
                progress_card = [
                    f"Downloading: {filename} (Fallback Mode)",
                    f"Progress: {downloaded_mb:.1f}MB/{total_mb:.1f}MB ({percent}%)",
                    f"Speed: {speed:.1f} MB/s"
                ]
                live_print(progress_card, title="GGUF Downloader", force=False, min_interval=0.25)
                
    live_print([f"Download Completed: {filename}"], title="GGUF Downloader", force=True)
    return target_path


def download_model(
    model_url: str,
    mmproj_url: Optional[str] = None,
    output_dir: str = "",
    connections: int = 16,
    hf_token: Optional[str] = None,
    profile: Optional[RuntimeProfile] = None
) -> Dict[str, Any]:
    """Downloads model.gguf and optional mmproj.gguf using parallel segmented downloading.
    
    Saves model_config.json inside the target directory and records state.
    """
    if profile is None:
        profile_name = "kaggle_t4x2" if "kaggle" in output_dir.lower() else "colab_t4x1"
        profile = RuntimeProfile.from_name(profile_name)
    
    # Use profile default model path if not specified
    if not output_dir:
        output_dir = profile.model_dir
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Resolve names from URLs
    model_filename = model_url.split("?")[0].split("/")[-1]
    if not model_filename.endswith(".gguf"):
        model_filename = "model.gguf"
        
    # Append auth header if token is provided
    # Standard Hugging Face download format supports auth headers
    resolved_model_url = model_url
    if hf_token and "huggingface.co" in model_url:
        # Append token parameter or use environment
        if "?" in model_url:
            resolved_model_url = f"{model_url}&token={hf_token}"
        else:
            resolved_model_url = f"{model_url}?token={hf_token}"
            
    # 1. Download model file
    use_aria2 = shutil.which("aria2c") is not None
    if use_aria2:
        model_path = download_file_aria2(resolved_model_url, output_dir, model_filename, connections)
    else:
        model_path = download_file_fallback(resolved_model_url, output_dir, model_filename)
        
    # 2. Download mmproj file if provided
    mmproj_path = None
    if mmproj_url and mmproj_url.strip():
        mmproj_filename = mmproj_url.split("?")[0].split("/")[-1]
        if not mmproj_filename.endswith(".gguf"):
            mmproj_filename = "mmproj.gguf"
            
        resolved_mmproj_url = mmproj_url
        if hf_token and "huggingface.co" in mmproj_url:
            if "?" in mmproj_url:
                resolved_mmproj_url = f"{mmproj_url}&token={hf_token}"
            else:
                resolved_mmproj_url = f"{mmproj_url}?token={hf_token}"
                
        if use_aria2:
            mmproj_path = download_file_aria2(resolved_mmproj_url, output_dir, mmproj_filename, connections)
        else:
            mmproj_path = download_file_fallback(resolved_mmproj_url, output_dir, mmproj_filename)

    # 3. Create model_config.json metadata
    config_data = {
        "model_url": model_url,
        "model_path": os.path.abspath(model_path),
        "model_filename": model_filename,
        "mmproj_url": mmproj_url,
        "mmproj_path": os.path.abspath(mmproj_path) if mmproj_path else None,
        "mmproj_filename": mmproj_filename if mmproj_path else None,
        "download_timestamp": time.time()
    }
    
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
        
    # Update runtime state
    state_update = {
        "model_path": os.path.abspath(model_path),
        "mmproj_path": os.path.abspath(mmproj_path) if mmproj_path else None,
        "model_config_file": os.path.abspath(config_path)
    }
    save_state(profile, state_update)
    
    live_print([
        "All downloads completed successfully!",
        f"Model: {model_path}",
        f"Multimodal Projector: {mmproj_path if mmproj_path else 'None'}",
        f"Metadata file: {config_path}"
    ], title="Downloads Finished", force=True)
    
    return config_data
