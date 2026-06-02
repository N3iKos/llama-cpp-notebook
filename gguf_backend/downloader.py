import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from .panel import run_command, show_summary


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


def _size_gib(num):
    return f"{num / 1024**3:.2f} GiB"


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
        show_summary(
            "Download skipped",
            lines=[f"file: {out_path}", f"size: {_size_gib(local_size(out_path))}", "state: already complete"],
        )
        return str(out_path)

    headers = []
    if token:
        headers += ["--header", f"Authorization: Bearer {token}"]

    cmd = [
        "aria2c", "-c", "-x", str(connections), "-s", str(connections), "-k", "1M",
        "--file-allocation=none", "--allow-overwrite=true", "--auto-file-renaming=false",
        "--summary-interval=1", "--console-log-level=warn", "--show-console-readout=true",
        "--retry-wait=3", "--max-tries=5",
        "-d", str(out_dir), "-o", out_name, *headers, url,
    ]

    result = run_command(
        cmd,
        label=f"download: {out_name}",
        mode="download",
        check=True,
        log_name=f"download_{Path(out_name).name}.log".replace("/", "_"),
        tail_lines=120,
        refresh_interval=0.10,
        hide_redirects=True,
    )

    final_size = local_size(out_path)
    if final_size <= 0:
        raise RuntimeError(f"Downloaded file is empty: {out_path}")

    show_summary(
        "Download result",
        lines=[f"file: {out_path}", f"size: {_size_gib(final_size)}", "state: complete"],
        log_path=getattr(result, "log_path", None),
    )
    return str(out_path)


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
    show_summary("Model config", lines=[f"model_path: {model_path}", f"mmproj_path: {mmproj_path or '-'}", f"config: {cfg_path}"])
    return cfg
