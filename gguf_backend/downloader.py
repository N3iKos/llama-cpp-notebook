import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .ui import live_print


def filename_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = Path(urllib.parse.unquote(path)).name
    if not name or "." not in name:
        raise ValueError(f"Cannot derive filename from URL: {url}")
    return name


def get_remote_size(url: str, token: str = ""):
    headers = {"User-Agent": "llama-cpp-notebook"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            size = r.headers.get("Content-Length")
            return int(size) if size else None
    except Exception:
        return None


def local_size(path):
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def download_with_aria_live(url, out_dir, out_name=None, *, token="", connections=16, clear=True):
    if not url:
        return ""

    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c not found.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = out_name or filename_from_url(url)
    out_path = out_dir / out_name
    remote_size = get_remote_size(url, token=token)

    if remote_size and local_size(out_path) >= remote_size:
        live_print(f"download complete: {out_name}\nsize: {local_size(out_path) / 1024**3:.2f} GiB", clear=clear)
        return str(out_path)

    headers = []
    if token:
        headers += ["--header", f"Authorization: Bearer {token}"]

    cmd = [
        "aria2c", "-c", "-x", str(connections), "-s", str(connections), "-k", "1M",
        "--file-allocation=none", "--allow-overwrite=true", "--auto-file-renaming=false",
        "--summary-interval=1", "--console-log-level=notice",
        "-d", str(out_dir), "-o", out_name, *headers, url,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    start = time.time()
    last_line = ""
    last_update = 0.0
    history = []

    try:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            now = time.time()

            if line:
                line = line.strip()
                if line:
                    history.append(line)
                    history = history[-10:]
                    if line.startswith("[#") or "ETA:" in line or "DL:" in line:
                        last_line = line

            current = local_size(out_path)
            if now - last_update >= 0.5:
                if remote_size:
                    pct = min(100.0, current / remote_size * 100)
                    size_line = f"{current / 1024**2:.0f}MiB/{remote_size / 1024**3:.2f}GiB ({pct:.1f}%)"
                else:
                    size_line = f"{current / 1024**2:.0f}MiB"

                live_print("\n".join([f"download: {out_name}", f"elapsed: {int(now - start)}s", last_line or size_line]), clear=clear)
                last_update = now

            if proc.poll() is not None:
                rest = proc.stdout.read() if proc.stdout else ""
                if rest:
                    history.extend([x for x in rest.splitlines() if x.strip()])
                break

            if not line:
                time.sleep(0.1)

        if proc.returncode != 0:
            live_print("\n".join([f"download failed: {out_name}", *history[-20:]]), clear=clear)
            raise RuntimeError(f"aria2c failed for {out_name} with code {proc.returncode}")

        final_size = local_size(out_path)
        if final_size <= 0:
            raise RuntimeError(f"Downloaded file is empty: {out_path}")

        live_print(f"download complete: {out_name}\nsize: {final_size / 1024**3:.2f} GiB", clear=clear)
        return str(out_path)

    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


def download_model_pair(model_url, mmproj_url, model_dir, *, hf_token="", connections=16):
    model_dir = Path(model_dir)
    model_path = download_with_aria_live(model_url, model_dir, token=hf_token, connections=connections)

    mmproj_path = ""
    if mmproj_url:
        mmproj_path = download_with_aria_live(mmproj_url, model_dir, token=hf_token, connections=connections)

    cfg = {
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "model_dir": str(model_dir),
        "model_url": model_url,
        "mmproj_url": mmproj_url,
    }

    if str(model_dir).startswith("/kaggle/working"):
        cfg_path = Path("/kaggle/working/model_config.json")
    elif str(model_dir).startswith("/content"):
        cfg_path = Path("/content/model_config.json")
    else:
        cfg_path = model_dir / "model_config.json"

    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg
