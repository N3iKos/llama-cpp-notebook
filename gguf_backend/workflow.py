"""High-level Kaggle/Colab notebook workflows.

Notebook cells should stay short and call these functions. Each function renders a
single structured ipywidgets output cell.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .downloader import download_model_pair
from .installer import ensure_apt_tools, install_llama_cpp_prebuilt, read_env_file
from .panel import NotebookPanel, run_command
from .server import ServerConfig, start_server
from .tunnel import start_tunnels


def _root(default=None):
    if default:
        return str(default)
    if Path("/kaggle/working").exists():
        return "/kaggle/working"
    if Path("/content").exists():
        return "/content"
    return str(Path.cwd())


def get_kaggle_secret(name: str, default: str = "") -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret(name)
        return value or default
    except Exception:
        return default


def setup_runtime(
    *,
    root_dir=None,
    cuda_preference="12.8",
    force_llama=False,
    include_diagnostics=True,
    install_pyngrok=True,
):
    root_dir = _root(root_dir)
    panel = NotebookPanel("Cell 1 — setup / diagnostics / llama.cpp", show_progress=False, height=390)
    panel.display_panel()
    panel.set_status("starting setup")
    panel.set_summary("runtime", lines=[f"root: {root_dir}", f"python: {sys.version.split()[0]}"])

    try:
        panel.section("apt tools")
        ensure_apt_tools(panel=panel)

        if install_pyngrok:
            panel.section("python dependencies")
            run_command(
                [sys.executable, "-m", "pip", "install", "-q", "pyngrok"],
                label="install pyngrok",
                panel=panel,
                finalize=False,
                check=False,
                tail_lines=120,
            )

        if include_diagnostics:
            panel.section("diagnostics")
            diag_cmds = [
                ("gpu summary", "nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader,nounits || true"),
                ("gpu topology", "nvidia-smi topo -m || true"),
                ("cuda compiler", "nvcc --version || true"),
                ("disk", f"df -h {root_dir} /tmp || true"),
            ]
            for label, cmd in diag_cmds:
                run_command(cmd, label=label, panel=panel, finalize=False, check=False, tail_lines=180)

        info = install_llama_cpp_prebuilt(root_dir, cuda_preference=cuda_preference, force=force_llama, panel=panel)

        env = read_env_file(Path(root_dir) / "llama_env.sh")
        panel.section("llama.cpp checks")
        run_command([info["server"], "--version"], label="llama-server version", env=env, panel=panel, finalize=False, check=False)
        run_command([info["server"], "--list-devices"], label="llama-server devices", env=env, panel=panel, finalize=False, check=False, tail_lines=180)

        panel.set_summary("setup complete", data=info)
        panel.finish(True, "setup complete")
        return info
    except Exception as e:
        panel.append("")
        panel.append(f"ERROR: {type(e).__name__}: {e}")
        panel.finish(False, "setup failed")
        raise


def download_assets(
    *,
    model_url: str,
    mmproj_url: str = "",
    model_dir=None,
    hf_token: str = "",
    connections: int = 16,
):
    root = _root(None)
    model_dir = model_dir or str(Path(root) / "models" / "current")
    hf_token = hf_token or os.environ.get("HF_TOKEN", "")

    panel = NotebookPanel("Cell 2 — model / mmproj downloader", show_progress=True, height=390)
    panel.display_panel()
    panel.set_status("starting downloader")
    panel.set_summary("download config", lines=[f"model_dir: {model_dir}", f"hf_token: {'set' if hf_token else 'empty'}", f"connections: {connections}"])

    try:
        cfg = download_model_pair(
            model_url,
            mmproj_url,
            model_dir,
            hf_token=hf_token,
            connections=connections,
            panel=panel,
        )
        panel.set_progress_visible(False)
        panel.set_summary("download complete", data=cfg)
        panel.finish(True, "download complete")
        return cfg
    except Exception as e:
        panel.append("")
        panel.append(f"ERROR: {type(e).__name__}: {e}")
        panel.finish(False, "download failed")
        raise


def launch_backend(
    *,
    root_dir=None,
    model_config_path=None,
    port=8080,
    alias="local-vl",
    ctx_size=8192,
    split_mode="row",
    fallback_split_mode="layer",
    tensor_split="1,1",
    threads=4,
    threads_batch=4,
    parallel=1,
    batch_size=2048,
    ubatch_size=512,
    flash_attn=True,
    cache_type_k="f16",
    cache_type_v="f16",
    image_min_tokens=None,
    image_max_tokens=None,
    chat_template_kwargs=None,
    mmproj_offload=True,
    cuda_visible_devices="0,1",
    warmup=True,
    tunnel_mode="both",
    ngrok_token="",
    fallback_cloudflare=True,
):
    root_dir = _root(root_dir)
    model_config_path = Path(model_config_path or (Path(root_dir) / "model_config.json"))

    panel = NotebookPanel("Cell 3 — server / warmup / tunnel", show_progress=False, height=420)
    panel.display_panel()
    panel.set_status("loading config")

    try:
        cfg_json = json.loads(model_config_path.read_text())
        cfg = ServerConfig(
            root_dir=root_dir,
            model_path=cfg_json["model_path"],
            mmproj_path=cfg_json.get("mmproj_path", ""),
            port=port,
            alias=alias,
            ctx_size=ctx_size,
            split_mode=split_mode,
            fallback_split_mode=fallback_split_mode,
            tensor_split=tensor_split,
            threads=threads,
            threads_batch=threads_batch,
            parallel=parallel,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            flash_attn=flash_attn,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            image_min_tokens=image_min_tokens,
            image_max_tokens=image_max_tokens,
            chat_template_kwargs=chat_template_kwargs,
            mmproj_offload=mmproj_offload,
            cuda_visible_devices=cuda_visible_devices,
        )

        panel.set_summary(
            "launch config",
            lines=[
                f"model: {cfg.model_path}",
                f"mmproj: {cfg.mmproj_path or '-'}",
                f"port: {port}",
                f"ctx_size: {ctx_size}",
                f"split_mode: {split_mode}",
                f"tunnel_mode: {tunnel_mode}",
                f"ngrok_token: {'set' if ngrok_token else 'empty'}",
            ],
        )

        panel.section("start llama-server")
        server_info = start_server(cfg, warmup=warmup, panel=panel)

        panel.section("tunnel")
        tunnel_urls = start_tunnels(
            port,
            root_dir,
            mode=tunnel_mode,
            ngrok_token=ngrok_token,
            fallback_cloudflare=fallback_cloudflare,
            panel=panel,
            finalize=False,
        )

        links = {
            "local chat": server_info["chat_endpoint"],
            "local models": server_info["models_endpoint"],
        }
        for name, url in tunnel_urls.items():
            if not name.endswith("_error"):
                base = url.rstrip("/")
                links[f"{name} chat"] = base + "/v1/chat/completions"
                links[f"{name} models"] = base + "/v1/models"

        panel.set_links("ready endpoints", links, note="Copy the chat endpoint for OpenAI-compatible clients.")
        panel.append("")
        panel.append("backend ready")
        panel.render()
        panel.finish(True, "backend ready")

        return {"server": server_info, "tunnels": tunnel_urls, "links": links}
    except Exception as e:
        panel.append("")
        panel.append(f"ERROR: {type(e).__name__}: {e}")
        panel.finish(False, "launch failed")
        raise
