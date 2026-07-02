import json
import shutil
import sys
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


def download_with_aria_live(url, out_dir, out_name=None, *, token="", connections=16, clear=True, downloader="auto", panel=None):
    if not url:
        return ""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = out_name or filename_from_url(url)
    out_path = out_dir / out_name
    remote_size = get_remote_size(url, token=token)

    if remote_size and local_size(out_path) >= remote_size:
        lines = [f"file: {out_path}", f"size: {_size_gib(local_size(out_path))}", "state: already complete"]
        if panel:
            panel.append(f"download skipped: {out_path}")
            panel.set_summary("download skipped", lines=lines)
            panel.render()
        else:
            show_summary("download skipped", lines=lines)
        return str(out_path)

    # Logika deteksi downloader
    is_hf_url = "huggingface.co" in url
    use_hf = False
    if downloader == "huggingface":
        use_hf = True
    elif downloader == "auto" and is_hf_url:
        use_hf = True

    if use_hf:
        helper_script = Path(__file__).parent / "hf_download_helper.py"
        cmd = [
            sys.executable, str(helper_script), url, str(out_dir),
        ]
        if token:
            cmd += ["--token", token]
        
        result = run_command(
            cmd,
            label=f"download (hf): {out_name}",
            mode="download",
            check=True,
            log_name=f"download_hf_{Path(out_name).name}.log".replace("/", "_"),
            tail_lines=160,
            refresh_interval=0.10,
            hide_redirects=True,
            panel=panel,
            finalize=False if panel else True,
        )
        
        final_size = local_size(out_path)
        if final_size <= 0:
            raise RuntimeError(f"Downloaded file is empty: {out_path}")

        lines = [f"file: {out_path}", f"size: {_size_gib(final_size)}", "state: complete"]
        if panel:
            panel.set_summary("download result", lines=lines)
            panel.append(f"download complete: {out_path}")
            panel.render()
        else:
            show_summary("download result", lines=lines, log_path=getattr(result, "log_path", None))
        return str(out_path)

    # Fallback ke aria2c
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c not found.")

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
        tail_lines=160,
        refresh_interval=0.10,
        hide_redirects=True,
        panel=panel,
        finalize=False if panel else True,
    )

    final_size = local_size(out_path)
    if final_size <= 0:
        raise RuntimeError(f"Downloaded file is empty: {out_path}")

    lines = [f"file: {out_path}", f"size: {_size_gib(final_size)}", "state: complete"]
    if panel:
        panel.set_summary("download result", lines=lines)
        panel.append(f"download complete: {out_path}")
        panel.render()
    else:
        show_summary("download result", lines=lines, log_path=getattr(result, "log_path", None))
    return str(out_path)


def download_model_pair(model_url, mmproj_url, model_dir, *, hf_token="", connections=16, downloader="auto", panel=None):
    model_dir = Path(model_dir)

    if panel:
        panel.section("download model")
    model_path = download_with_aria_live(model_url, model_dir, token=hf_token, connections=connections, downloader=downloader, panel=panel)

    mmproj_path = ""
    if mmproj_url:
        if panel:
            panel.section("download mmproj")
        mmproj_path = download_with_aria_live(mmproj_url, model_dir, token=hf_token, connections=connections, downloader=downloader, panel=panel)

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
    cfg["config_path"] = str(cfg_path)

    if panel:
        panel.set_summary("model config", data=cfg)
        panel.render()
    else:
        show_summary("model config", cfg)
    return cfg
