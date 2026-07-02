"""Helper script: download a single file from Hugging Face using huggingface_hub.

Dijalankan sebagai subprocess oleh downloader.py.  Mencetak progress dalam
format yang kompatibel dengan parser aria2c di panel.py sehingga widget
notebook tetap menampilkan live progress bar.

Pendekatan: jalankan hf_hub_download di thread latar belakang, lalu
monitor ukuran file di disk dari thread utama dan cetak progress.
Ini bekerja 100% terlepas dari backend transfer (hf_transfer / xet / default).
"""

import os
import sys
import re
import glob
import time
import argparse
import subprocess
import threading
import warnings
import urllib.request

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

from huggingface_hub import hf_hub_download  # noqa: E402


# ---------------------------------------------------------------------------
# 2. Utilitas
# ---------------------------------------------------------------------------
def _fmt_size(b):
    """Format bytes ke string yang mudah dibaca (KiB, MiB, GiB, dst.)."""
    if b is None or b <= 0:
        return "0B"
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if b < 1024.0:
            return f"{b:.1f}{u}"
        b /= 1024.0
    return f"{b:.1f}PiB"


def _get_remote_size(url, token=""):
    """Dapatkan ukuran file dari server via HEAD request."""
    headers = {"User-Agent": "llama-cpp-notebook"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def _scan_download_bytes(out_dir, filename):
    """Cari ukuran file yang sedang diunduh — termasuk file .incomplete di cache HF."""
    # 1. Cek file final
    final = os.path.join(out_dir, filename)
    if os.path.isfile(final):
        try:
            return os.path.getsize(final)
        except OSError:
            pass

    # 2. Cek file .incomplete di cache HF lokal (huggingface_hub >= 0.25)
    #    Pattern: <out_dir>/.cache/huggingface/download/*.incomplete
    for pattern in (
        os.path.join(out_dir, ".cache", "huggingface", "download", "*.incomplete"),
        os.path.join(out_dir, ".cache", "**", "*.incomplete"),
    ):
        for fp in glob.glob(pattern, recursive=True):
            try:
                return os.path.getsize(fp)
            except OSError:
                pass

    # 3. Cek cache global HF (~/.cache/huggingface/hub/...)
    hf_cache = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )
    if os.path.isdir(hf_cache):
        for fp in glob.glob(os.path.join(hf_cache, "**", "*.incomplete"), recursive=True):
            try:
                sz = os.path.getsize(fp)
                if sz > 0:
                    return sz
            except OSError:
                pass

    return 0


def _print_progress(downloaded, total, start_time):
    """Cetak satu baris progress bergaya aria2c ke stdout."""
    now = time.time()
    elapsed = now - start_time
    pct = (downloaded / total * 100) if total > 0 else 0.0

    if elapsed > 0 and downloaded > 0:
        speed = downloaded / elapsed
        speed_s = _fmt_size(speed)
        remaining = total - downloaded
        if remaining > 0 and speed > 0:
            eta = remaining / speed
            eta_s = f"{int(eta // 60)}m{int(eta % 60)}s" if eta >= 60 else f"{int(eta)}s"
        else:
            eta_s = "0s"
    else:
        speed_s = "0B"
        eta_s = "unknown"

    line = (
        f"[#000000 {_fmt_size(downloaded)}/{_fmt_size(total)}({pct:.0f}%) "
        f"CN:1 DL:{speed_s} ETA:{eta_s}]"
    )
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# 3. URL parser
# ---------------------------------------------------------------------------
def parse_hf_url(url: str):
    """Parse URL Hugging Face menjadi (repo_id, revision, filename)."""
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
        print(f"ERROR: Cannot parse Hugging Face URL: {args.url}", flush=True)
        sys.exit(1)

    repo_id, revision, filename = parsed
    print(f"Parsed HF URL: repo={repo_id}  rev={revision}  file={filename}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # Dapatkan ukuran total file untuk progress tracking
    total_size = _get_remote_size(args.url, token=args.token)
    if total_size:
        print(f"Remote size: {_fmt_size(total_size)}", flush=True)
    else:
        print("Remote size: unknown (progress will be approximate)", flush=True)

    # Jalankan download di thread latar belakang
    result = {"path": None, "error": None}

    def _download_thread():
        try:
            result["path"] = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=args.out_dir,
                token=args.token or None,
            )
        except Exception as exc:
            result["error"] = exc

    dl_thread = threading.Thread(target=_download_thread, daemon=True)
    dl_thread.start()

    # Monitor progress dari thread utama
    start_time = time.time()
    last_print = 0.0
    effective_total = total_size or 0

    while dl_thread.is_alive():
        now = time.time()
        if now - last_print >= 1.0:  # Update setiap 1 detik
            downloaded = _scan_download_bytes(args.out_dir, filename)
            if downloaded > 0:
                # Jika total_size belum diketahui, estimasi dari progress
                if effective_total <= 0 and downloaded > 0:
                    effective_total = downloaded  # akan terus update
                _print_progress(downloaded, effective_total, start_time)
            last_print = now
        time.sleep(0.25)

    # Cek hasil akhir
    dl_thread.join()

    if result["error"] is not None:
        exc = result["error"]
        print(f"ERROR: Download failed — {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)

    # Cetak progress akhir 100%
    if effective_total > 0:
        _print_progress(effective_total, effective_total, start_time)

    print(f"SUCCESS: {result['path']}", flush=True)


if __name__ == "__main__":
    main()
