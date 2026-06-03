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
from .panel import NotebookPanel, run_command, tail_text
from .server import ServerConfig, start_server, shutdown_all
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


def _is_non_empty(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def launch_backend(
    *,
    root_dir=None,
    model_config_path=None,
    warmup=True,
    tunnel_mode="both",
    ngrok_token="",
    fallback_cloudflare=True,
    thinking_config=None,
    **server_options,
):
    """Start llama-server, warm it up, and expose tunnels.

    All llama-server options are passed as keyword arguments matching
    ServerConfig fields. Empty string values are accepted and skipped by
    the command builder, so llama.cpp defaults remain active.

    Parameters
    ----------
    thinking_config : dict, optional
        Model-aware reasoning/thinking toggle config.  Keys: ``family``,
        ``mode``, ``budget``, ``format``, ``soft_prompt``.
    """
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
            thinking_config=thinking_config,
            **server_options,
        )

        selected = []
        for key in [
            "host", "port", "alias", "ctx_size", "gpu_layers", "cuda_visible_devices",
            "split_mode", "tensor_split", "main_gpu", "parallel", "batch_size",
            "ubatch_size", "flash_attn", "cache_type_k", "cache_type_v",
            "reasoning", "reasoning_budget", "api_key", "ui", "metrics", "slots",
        ]:
            value = getattr(cfg, key, "")
            if _is_non_empty(value):
                if key == "api_key":
                    value = "set"
                selected.append(f"{key}: {value}")

        panel.set_summary(
            "launch config",
            lines=[
                f"model: {cfg.model_path}",
                f"mmproj: {cfg.mmproj_path or '-'}",
                f"tunnel_mode: {tunnel_mode}",
                f"ngrok_token: {'set' if ngrok_token else 'empty'}",
                "",
                *selected,
            ],
        )

        panel.section("start llama-server")
        server_info = start_server(cfg, warmup=warmup, panel=panel)

        # Display thinking mapper summary (if thinking_config was used).
        thinking_result = getattr(cfg, "_thinking_result", None)
        if thinking_result and thinking_result.summary:
            panel.section("thinking mapper")
            for k, v in thinking_result.summary.items():
                panel.append(f"  {k}: {v}")
            if thinking_result.warnings:
                panel.append("")
                for w in thinking_result.warnings:
                    panel.append(f"  ⚠ {w}")
            panel.render()

        panel.section("tunnel")
        port_for_tunnel = int(cfg.port) if _is_non_empty(cfg.port) else 8080
        tunnel_urls = start_tunnels(
            port_for_tunnel,
            root_dir,
            mode=tunnel_mode,
            ngrok_token=ngrok_token,
            fallback_cloudflare=fallback_cloudflare,
            panel=panel,
            finalize=False,
        )

        links = {
            "local base": {"url": server_info["base_url"], "copy": True, "open": False},
            "local chat": {"url": server_info["chat_endpoint"], "copy": True, "open": False},
            "local models": {"url": server_info["models_endpoint"], "copy": True, "open": False},
        }
        for name, url in tunnel_urls.items():
            if not name.endswith("_error"):
                base = url.rstrip("/")
                links[f"{name} public"] = {"url": base, "copy": True, "open": True}
                links[f"{name} chat"] = {"url": base + "/v1/chat/completions", "copy": True, "open": False}
                links[f"{name} models"] = {"url": base + "/v1/models", "copy": True, "open": False}

        panel.set_links(
            "ready endpoints",
            links,
            note="Open the public URL for the llama.cpp playground, or copy /v1 endpoints for OpenAI-compatible clients.",
        )
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


def launch_backend_live(
    *,
    root_dir=None,
    model_config_path=None,
    warmup=True,
    tunnel_mode="both",
    ngrok_token="",
    fallback_cloudflare=True,
    thinking_config=None,
    health_interval=5,
    log_refresh=1.0,
    **server_options,
):
    """Start llama-server with tunnels, then enter a live monitoring loop.

    This function wraps ``launch_backend()`` and then continuously streams
    the server log file and performs periodic health checks. A prominent
    **Shutdown** button is displayed; clicking it kills the llama-server,
    cloudflared, and ngrok processes and cleanly exits the monitoring loop.

    The cell that calls this function will keep running until either:
    - The user clicks the Shutdown button, or
    - The llama-server process dies unexpectedly.

    Parameters
    ----------
    root_dir : str, optional
        Working directory. Auto-detected for Kaggle/Colab if omitted.
    model_config_path : str, optional
        Path to ``model_config.json``. Defaults to ``<root>/model_config.json``.
    warmup : bool
        Send a warmup chat request before declaring ready.
    tunnel_mode : str
        ``"both"``, ``"ngrok"``, ``"cloudflare"``, or ``"none"``.
    ngrok_token : str
        ngrok auth token. Empty = skip ngrok.
    fallback_cloudflare : bool
        Start cloudflare if ngrok fails.
    health_interval : int
        Seconds between ``/health`` checks (default 5).
    log_refresh : float
        Seconds between log tail refreshes (default 1.0).
    **server_options
        Keyword arguments forwarded to ``ServerConfig``.

    Returns
    -------
    dict
        Same as ``launch_backend()`` plus ``{"shutdown_status": ...}``.
    """
    import asyncio
    import signal
    import time

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        try:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nest_asyncio"], check=False)
            import nest_asyncio
            nest_asyncio.apply()
        except Exception:
            pass

    from .client import get_json

    root_dir = _root(root_dir)
    shutdown_flag_path = Path(root_dir) / "shutdown.flag"

    # Clean up any stale flag from a previous run.
    if shutdown_flag_path.exists():
        shutdown_flag_path.unlink(missing_ok=True)

    # ── Phase 1: delegate to launch_backend() for the initial setup ──
    result = launch_backend(
        root_dir=root_dir,
        model_config_path=model_config_path,
        warmup=warmup,
        tunnel_mode=tunnel_mode,
        ngrok_token=ngrok_token,
        fallback_cloudflare=fallback_cloudflare,
        thinking_config=thinking_config,
        **server_options,
    )

    base_url = result["server"]["base_url"]
    log_path = Path(result["server"]["log_path"])
    server_pid = result["server"]["pid"]

    # ── Phase 2: create the live monitoring panel ──
    panel = NotebookPanel(
        "Cell 3 — live server monitor",
        show_progress=False,
        height=380,
    )
    panel.display_panel()
    panel.set_footer(f"log: {log_path} | pid: {server_pid}")

    # Re-display the endpoint links in the live panel dashboard.
    if "links" in result:
        panel.set_links(
            "ready endpoints",
            result["links"],
            note="Open the public URL for the llama.cpp playground, or copy /v1 endpoints for OpenAI-compatible clients.",
        )

    # ── Phase 3: wire up the shutdown button ──
    #
    # In Jupyter/Kaggle/Colab, ipywidgets button callbacks run on the main thread's
    # Tornado IOLoop. To keep the loop responsive during cell execution:
    #
    #   1. We apply ``nest_asyncio`` to allow nested event loops in Jupyter.
    #   2. We run the monitoring loop as an asynchronous coroutine.
    #   3. The click callback calls ``shutdown_all`` and sets ``shutdown_event``.
    #
    # This guarantees the shutdown button functions instantly and terminates the loop.

    shutdown_event = asyncio.Event()

    def _do_shutdown():
        """Callback invoked by the shutdown button click."""
        panel.append("")
        panel.append("═" * 80)
        panel.append("SHUTDOWN REQUESTED")
        panel.append("═" * 80)
        panel.render()

        # 1. Kill all server/tunnel processes.
        status = shutdown_all(root_dir)
        for component, state in status.items():
            panel.append(f"  {component}: {state}")
        panel.render()

        result["shutdown_status"] = status

        # 2. Write file-based shutdown flag (just in case).
        try:
            shutdown_flag_path.write_text("shutdown")
        except OSError:
            pass

        # 3. Set the event safely from the event loop thread
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(shutdown_event.set)
            else:
                shutdown_event.set()
        except Exception:
            shutdown_event.set()

    panel.add_shutdown_button(_do_shutdown)
    panel.set_live_status("server running", "#4ade80")
    panel.section("live log")
    panel.set_status("live monitoring | server running")
    panel.render()

    # ── Phase 4: continuous monitoring loop ──
    async def _monitoring_loop():
        last_health_check = 0.0
        health_ok = True
        uptime_start = time.time()

        while not shutdown_event.is_set():
            if shutdown_flag_path.exists():
                break

            now = time.time()
            uptime_secs = int(now - uptime_start)
            uptime_str = _format_uptime(uptime_secs)

            # --- Stream log tail ---
            log_tail = tail_text(log_path, 60)
            panel.lines = log_tail.splitlines() if log_tail else ["(no log output yet)"]

            # --- Periodic health check ---
            if now - last_health_check >= health_interval:
                try:
                    status_code, _ = get_json(base_url + "/health", timeout=5)
                    if status_code == 200:
                        if not health_ok:
                            panel.append("")
                            panel.append(f"[{_ts()}] server recovered")
                        health_ok = True
                        panel.set_live_status(
                            f"server healthy | uptime {uptime_str} | pid {server_pid}",
                            "#4ade80",
                        )
                        panel.set_status(f"live monitoring | healthy | uptime {uptime_str}")
                    else:
                        health_ok = False
                        panel.set_live_status(
                            f"server unhealthy (HTTP {status_code}) | uptime {uptime_str}",
                            "#facc15",
                        )
                        panel.set_status(f"live monitoring | unhealthy | uptime {uptime_str}")
                except Exception:
                    health_ok = False
                    panel.set_live_status(
                        f"server unreachable | uptime {uptime_str}",
                        "#f87171",
                    )
                    panel.set_status(f"live monitoring | unreachable | uptime {uptime_str}")
                last_health_check = now

            # --- Check if the server process is still alive ---
            try:
                os.kill(server_pid, 0)  # signal 0 = check existence
            except OSError:
                panel.append("")
                panel.append(f"[{_ts()}] server process (pid {server_pid}) is no longer running")
                panel.set_live_status("server process died", "#f87171")
                panel.set_status("server process died")
                panel.render()
                break

            panel.render()

            # Sleep asynchronously to yield control back to the event loop.
            await asyncio.sleep(log_refresh)

    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_monitoring_loop())
    except KeyboardInterrupt:
        # Manual interrupt (not from our button) — kill processes too.
        if not shutdown_event.is_set() and not shutdown_flag_path.exists():
            panel.append("")
            panel.append(f"[{_ts()}] keyboard interrupt — shutting down")
            panel.render()
            shutdown_all(root_dir)

    # ── Phase 5: finalize ──
    panel.set_shutdown_complete()
    panel.append("")
    panel.append(f"[{_ts()}] monitoring stopped — you can re-run this cell to restart the server")
    panel.render()

    # Clean up the flag file.
    try:
        shutdown_flag_path.unlink(missing_ok=True)
    except OSError:
        pass

    return result


def _format_uptime(seconds: int) -> str:
    """Convert seconds to a human-readable uptime string."""
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


def _ts() -> str:
    """Short HH:MM:SS timestamp for log lines."""
    import time as _time
    return _time.strftime("%H:%M:%S")
