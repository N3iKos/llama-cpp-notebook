"""Helper script: download a single file from Hugging Face using huggingface_hub.

Dijalankan sebagai subprocess oleh downloader.py.  Mencetak progress dalam
format yang kompatibel dengan parser aria2c di panel.py sehingga widget
notebook tetap menampilkan live progress bar.
"""

import os
import sys
import re
import time
import argparse
import subprocess
import warnings

# Suppress deprecation warnings agar tidak mengotori output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 1. Auto-install dependencies
# ---------------------------------------------------------------------------
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    print("Installing huggingface_hub...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
        check=True,
    )
    import huggingface_hub  # noqa: F401

try:
    import hf_transfer  # noqa: F401
except ImportError:
    print("Installing hf_transfer for high-speed downloads...", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "hf_transfer"],
            check=True,
        )
        import hf_transfer  # noqa: F401
    except Exception as exc:
        print(
            f"Warning: hf_transfer not available ({exc}). Using default method.",
            flush=True,
        )
        hf_transfer = None  # type: ignore[assignment]

# Aktifkan akselerator yang tersedia
if hf_transfer is not None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from tqdm import tqdm  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402


# ---------------------------------------------------------------------------
# 2. Custom tqdm: cetak progress dalam format aria2c
# ---------------------------------------------------------------------------
class AriaStyleTqdm(tqdm):
    """tqdm subclass yang mencetak satu baris progress per update ke stdout."""

    def __init__(self, *args, **kwargs):
        kwargs["file"] = sys.stdout
        super().__init__(*args, **kwargs)
        self._last_print: float = 0.0

    def display(self, msg=None, pos=None):
        now = time.time()
        if now - self._last_print < 0.25 and self.n < (self.total or 0):
            return
        self._last_print = now

        completed = self.n
        total = self.total or 0

        def _fmt(b):
            if b is None:
                return "0B"
            for u in ("B", "KiB", "MiB", "GiB", "TiB"):
                if b < 1024.0:
                    return f"{b:.1f}{u}"
                b /= 1024.0
            return f"{b:.1f}PiB"

        pct = (completed / total * 100) if total > 0 else 0.0
        elapsed = now - self.start_t

        if elapsed > 0 and completed > 0:
            speed = completed / elapsed
            speed_s = _fmt(speed)
            if total > completed:
                eta = (total - completed) / speed
                eta_s = f"{int(eta // 60)}m{int(eta % 60)}s" if eta >= 60 else f"{int(eta)}s"
            else:
                eta_s = "0s"
        else:
            speed_s = "0B"
            eta_s = "unknown"

        line = (
            f"[#000000 {_fmt(completed)}/{_fmt(total)}({pct:.0f}%) "
            f"CN:1 DL:{speed_s} ETA:{eta_s}]"
        )
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


# Monkey-patch tqdm agar huggingface_hub memakai kelas kita
tqdm.tqdm = AriaStyleTqdm  # type: ignore[attr-defined]
try:
    import tqdm.auto  # noqa: E402
    tqdm.auto.tqdm = AriaStyleTqdm  # type: ignore[attr-defined]
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 3. URL parser
# ---------------------------------------------------------------------------
def parse_hf_url(url: str):
    """Parse URL Hugging Face menjadi (repo_id, revision, filename).

    Mendukung format:
      https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>
      https://huggingface.co/<repo>/resolve/<rev>/<path>   (tanpa namespace)
    Query string (?download=true dll.) otomatis di-strip.
    """
    if "?" in url:
        url = url.split("?")[0]

    for pattern in (
        r"https?://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)",
        r"https?://huggingface\.co/([^/]+)/resolve/([^/]+)/(.+)",
        r"https?://huggingface\.co/([^/]+/[^/]+)/download/([^/]+)/(.+)",
        r"https?://huggingface\.co/([^/]+)/download/([^/]+)/(.+)",
    ):
        m = re.match(pattern, url)
        if m:
            return m.group(1), m.group(2), m.group(3)
    return None


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Download a file from Hugging Face Hub")
    ap.add_argument("url", help="Hugging Face file URL")
    ap.add_argument("out_dir", help="Output directory")
    ap.add_argument("--token", default="", help="HF token (optional)")
    args = ap.parse_args()

    parsed = parse_hf_url(args.url)
    if not parsed:
        # Cetak ke stdout (bukan stderr) agar terbaca oleh panel notebook
        print(f"ERROR: Cannot parse Hugging Face URL: {args.url}", flush=True)
        sys.exit(1)

    repo_id, revision, filename = parsed
    print(f"Parsed HF URL: repo={repo_id}  rev={revision}  file={filename}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=args.out_dir,
            token=args.token or None,
        )
        print(f"SUCCESS: {path}", flush=True)
    except Exception as exc:
        # Cetak ke stdout agar error terbaca oleh panel notebook
        print(f"ERROR: Download failed — {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
