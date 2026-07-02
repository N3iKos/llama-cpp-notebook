import os
import sys
import re
import time
import argparse
import subprocess

# 1. Coba impor atau pasang dependensi yang diperlukan secara otomatis
try:
    import huggingface_hub
except ImportError:
    print("Installing huggingface_hub...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    import huggingface_hub

try:
    import hf_transfer
except ImportError:
    print("Installing hf_transfer for high-speed downloads...", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer"], check=True)
        import hf_transfer
    except Exception as e:
        print(f"Warning: Failed to install hf_transfer ({e}). Falling back to default download method.", flush=True)
        hf_transfer = None

# Aktifkan hf_transfer jika pustaka tersedia, serta HF_XET_HIGH_PERFORMANCE untuk kompatibilitas modern
if hf_transfer is not None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from tqdm import tqdm
from huggingface_hub import hf_hub_download

# 2. Custom Tqdm untuk memformat output agar menyerupai parser progress aria2c
class AriaStyleTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["file"] = sys.stdout  # Alihkan output progress bar ke stdout agar terbaca subprocess
        super().__init__(*args, **kwargs)
        self.last_print_time = 0.0

    def display(self, msg=None, pos=None):
        now = time.time()
        # Batasi output kemajuan agar tidak terlalu membebani log (maksimal 4 kali per detik)
        if now - self.last_print_time < 0.25 and self.n < (self.total or 0):
            return

        self.last_print_time = now
        completed = self.n
        total = self.total or 0

        def format_size(bytes_val):
            if bytes_val is None:
                return "0B"
            for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
                if bytes_val < 1024.0:
                    return f"{bytes_val:.1f}{unit}"
                bytes_val /= 1024.0
            return f"{bytes_val:.1f}PiB"

        pct = (completed / total) * 100 if total > 0 else 0.0
        elapsed = now - self.start_t

        if elapsed > 0 and completed > 0:
            speed = completed / elapsed
            speed_str = format_size(speed) + "/s"
            if total > completed:
                eta = (total - completed) / speed
                if eta >= 60:
                    eta_str = f"{int(eta // 60)}m{int(eta % 60)}s"
                else:
                    eta_str = f"{int(eta)}s"
            else:
                eta_str = "0s"
        else:
            speed_str = "0B/s"
            eta_str = "unknown"

        completed_str = format_size(completed)
        total_str = format_size(total) if total > 0 else "unknown"

        # Format aria2c progress: [#baa800 11GiB/15GiB(74%) CN:1 DL:49MiB ETA:1m20s]
        # Kita gunakan CUID fiktif [#000000] agar dibaca secara tepat oleh parser regex
        progress_line = f"[#000000 {completed_str}/{total_str}({pct:.0f}%) CN:1 DL:{speed_str.replace('/s', '')} ETA:{eta_str}]"
        
        sys.stdout.write(progress_line + "\n")
        sys.stdout.flush()

# Lakukan monkey patch pada tqdm agar dipakai oleh huggingface_hub
tqdm.tqdm = AriaStyleTqdm
try:
    import tqdm.auto
    tqdm.auto.tqdm = AriaStyleTqdm
except ImportError:
    pass


def parse_hf_url(url: str):
    # Hilangkan query string jika ada (?download=true, dll.)
    if "?" in url:
        url = url.split("?")[0]
        
    # Regex yang mendukung repo ID tanpa namespace (bert-base-uncased) dan dengan namespace (user/repo)
    pattern_resolve = r"https?://huggingface\.co/([^/]+(?:/[^/]+)?)/resolve/([^/]+)/(.+)"
    match = re.match(pattern_resolve, url)
    if match:
        return match.group(1), match.group(2), match.group(3)
        
    pattern_download = r"https?://huggingface\.co/([^/]+(?:/[^/]+)?)/download/([^/]+)/(.+)"
    match = re.match(pattern_download, url)
    if match:
        return match.group(1), match.group(2), match.group(3)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Hugging Face model file URL")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--token", default="", help="Hugging Face token")
    args = parser.parse_args()

    parsed = parse_hf_url(args.url)
    if not parsed:
        print(f"ERROR: URL is not a valid Hugging Face download URL: {args.url}", file=sys.stderr)
        sys.exit(1)

    repo_id, revision, filename = parsed
    print(f"Parsed HF URL: Repo={repo_id}, Revision={revision}, File={filename}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=args.out_dir,
            local_dir_use_symlinks=False,
            token=args.token if args.token else None,
        )
        print(f"SUCCESS: Downloaded to {path}", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
