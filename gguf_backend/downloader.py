import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from .terminal_panel import NotebookTerminalPanel, TerminalPanelConfig, run_terminal_panel
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
    log_path = out_dir / f"{out_name}.aria2.log"

    if remote_size and local_size(out_path) >= remote_size:
        if clear:
            panel = NotebookTerminalPanel(enabled=True)
            panel.update(
                config=TerminalPanelConfig(
                    label=f"download skipped: {out_name}",
                    command_display="cached file already matches remote size",
                    log_path=log_path,
                ),
                status="DONE",
                elapsed_seconds=0.0,
                exit_code=0,
                lines=[f"size: {local_size(out_path) / 1024**3:.2f} GiB", str(out_path)],
            )
        else:
            live_print(f"download complete: {out_name}\nsize: {local_size(out_path) / 1024**3:.2f} GiB", clear=False)
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

    run_terminal_panel(
        cmd,
        TerminalPanelConfig(
            label=f"download: {out_name}",
            log_path=log_path,
            tail_lines=20,
            failure_tail_lines=30,
        ),
        display=clear,
        check=True,
    )

    final_size = local_size(out_path)
    if final_size <= 0:
        raise RuntimeError(f"Downloaded file is empty: {out_path}")

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
    return cfg
