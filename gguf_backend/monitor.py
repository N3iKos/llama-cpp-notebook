from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from .client import chat, get_json
from .installer import ensure_cloudflared, read_env_file
from .panel import NotebookPanel, sanitize_log_line, tail_text
from .server import ServerConfig, build_cmd, get_help, kill_pid, port_is_open, start_once, stop_server
from .thinking import build_thinking_args
from .shell import parse_trycloudflare_url


def _kill_file_pid(path):
    path = Path(path)
    if not path.exists():
        return
    try:
        kill_pid(int(path.read_text().strip()))
    except Exception:
        pass


def stop_runtime(root_dir):
    root = Path(root_dir)
    stop_server(root)
    _kill_file_pid(root / "cloudflared.pid")
    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass


def _read_new_lines(path, state, *, max_lines=80):
    path = Path(path)
    if not path.exists():
        return []
    pos = state.get(str(path), 0)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            text = f.read()
            state[str(path)] = f.tell()
    except Exception:
        return []
    lines = []
    for line in text.splitlines():
        clean = sanitize_log_line(line, hide_redirects=True, max_len=1000)
        if clean:
            lines.append(clean)
    return lines[-max_lines:]


def _make_controls(panel, root_dir, stop_event):
    import ipywidgets as widgets
    from IPython.display import display

    shutdown = widgets.Button(
        description="Shutdown server + tunnel",
        button_style="danger",
        tooltip="Stop llama-server, cloudflared, and ngrok",
        layout=widgets.Layout(width="220px"),
    )
    stop_tail = widgets.Button(
        description="Stop log monitor",
        button_style="warning",
        tooltip="Stop only realtime log monitoring; server remains running",
        layout=widgets.Layout(width="160px"),
    )
    state = widgets.HTML(
        value='<span style="font-family:ui-monospace,monospace;color:#aaa;">controls ready</span>'
    )

    def on_shutdown(_):
        state.value = '<span style="font-family:ui-monospace,monospace;color:#fca5a5;">shutdown requested...</span>'
        panel.append("shutdown requested: stopping llama-server + tunnels")
        panel.render()
        stop_event.set()
        stop_runtime(root_dir)
        panel.finish(False, "shutdown complete")
        state.value = '<span style="font-family:ui-monospace,monospace;color:#86efac;">shutdown complete</span>'

    def on_stop_tail(_):
        state.value = '<span style="font-family:ui-monospace,monospace;color:#fde68a;">log monitor stopped; server still running</span>'
        panel.append("log monitor stopped by user; server remains running")
        panel.render()
        stop_event.set()

    shutdown.on_click(on_shutdown)
    stop_tail.on_click(on_stop_tail)
    display(widgets.HBox([shutdown, stop_tail, state]))
    return shutdown, stop_tail, state


def _wait_ready_panel(proc, cfg: ServerConfig, log_path, panel, *, timeout=240, stop_event=None):
    base = f"http://127.0.0.1:{cfg.port}"
    start = time.time()
    last_render = 0.0
    state = {}

    while time.time() - start < timeout:
        if stop_event and stop_event.is_set():
            return False
        if proc.poll() is not None:
            panel.append("llama-server exited during startup")
            for line in tail_text(log_path, 120).splitlines():
                clean = sanitize_log_line(line, hide_redirects=True)
                if clean:
                    panel.append(clean)
            panel.render()
            return False
        if port_is_open("127.0.0.1", cfg.port):
            try:
                status, _ = get_json(base + "/health", timeout=5)
                if status == 200:
                    panel.append(f"llama-server health OK: {base}")
                    panel.render()
                    return True
            except Exception:
                pass

        now = time.time()
        if now - last_render >= 0.7:
            panel.set_status(f"llama-server starting | elapsed {int(now-start)}s | port {cfg.port}")
            for line in _read_new_lines(log_path, state, max_lines=40):
                panel.append(line)
            panel.render()
            last_render = now
        time.sleep(0.25)

    panel.append("llama-server startup timeout")
    for line in tail_text(log_path, 120).splitlines():
        clean = sanitize_log_line(line, hide_redirects=True)
        if clean:
            panel.append(clean)
    panel.render()
    return False


def _start_server_panel(cfg: ServerConfig, panel, *, stop_event=None):
    root = Path(cfg.root_dir)
    env = read_env_file(root / "llama_env.sh")
    server = Path(env.get("LLAMA_SERVER", ""))
    bin_dir = Path(env.get("LLAMA_BIN_DIR", ""))

    if not server.exists():
        raise RuntimeError(f"LLAMA_SERVER not found: {server}")

    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    panel.section("start llama-server")
    help_text = get_help(server, env)
    if cfg.thinking_config:
        _, thinking_summary, _ = build_thinking_args(
            cfg.thinking_config,
            help_text=help_text,
            model_path=cfg.model_path,
            alias=cfg.alias,
            existing_chat_template_kwargs=cfg.chat_template_kwargs,
        )
        panel.append("thinking mapper:")
        for line in thinking_summary:
            panel.append("  " + line)
        panel.render()
    stop_server(cfg.root_dir)

    proc, cmd, log_path = start_once(cfg, env, server, help_text)
    panel.set_footer(f"server log: {log_path}")
    panel.append("$ " + " ".join(str(x) for x in cmd))
    panel.render()

    ok = _wait_ready_panel(proc, cfg, log_path, panel, stop_event=stop_event)
    if not ok and cfg.fallback_split_mode and cfg.fallback_split_mode != cfg.split_mode:
        kill_pid(proc.pid)
        panel.append(f"fallback split mode: {cfg.split_mode} -> {cfg.fallback_split_mode}")
        cfg.split_mode = cfg.fallback_split_mode
        proc, cmd, log_path = start_once(cfg, env, server, help_text)
        ok = _wait_ready_panel(proc, cfg, log_path, panel, stop_event=stop_event)

    if not ok:
        raise RuntimeError("llama-server failed to start.")

    base = f"http://127.0.0.1:{cfg.port}"
    return {
        "base_url": base,
        "chat_endpoint": base + "/v1/chat/completions",
        "models_endpoint": base + "/v1/models",
        "pid": proc.pid,
        "cmd": cmd,
        "log_path": str(log_path),
    }


def _start_cloudflare_panel(port, root_dir, panel, *, stop_event=None, timeout=75):
    root_dir = Path(root_dir)
    cloudflared = ensure_cloudflared(target_dir=root_dir)
    log_path = root_dir / "cloudflared.log"
    pid_path = root_dir / "cloudflared.pid"

    _kill_file_pid(pid_path)
    cmd = [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(root_dir), bufsize=1)
    pid_path.write_text(str(proc.pid))

    panel.section("cloudflare tunnel")
    panel.append("$ " + " ".join(cmd))
    panel.render()

    start = time.time()
    url = None
    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        while time.time() - start < timeout:
            if stop_event and stop_event.is_set():
                return None
            if proc.poll() is not None:
                panel.append("cloudflared exited before URL was generated")
                panel.render()
                return None
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                logf.write(line)
                logf.flush()
                clean = sanitize_log_line(line, hide_redirects=False, max_len=1000)
                if clean:
                    panel.append(clean)
                    found = parse_trycloudflare_url(clean)
                    if found:
                        url = found.rstrip("/")
                        break
            panel.set_status(f"cloudflare starting | elapsed {int(time.time()-start)}s")
            panel.render()
            time.sleep(0.2)
    if url:
        panel.append(f"cloudflare public: {url}")
        panel.render()
    return url


def _start_ngrok_panel(port, token, panel):
    if not token:
        return None
    panel.section("ngrok tunnel")
    try:
        subprocess.run(["python3", "-m", "pip", "install", "-q", "pyngrok"], check=False)
        from pyngrok import ngrok
        ngrok.set_auth_token(token)
        ngrok.kill()
        tunnel = ngrok.connect(port, "http", bind_tls=True)
        url = tunnel.public_url.rstrip("/")
        panel.append(f"ngrok public: {url}")
        panel.render()
        return url
    except Exception as e:
        panel.append(f"ngrok error: {e}")
        panel.render()
        return None


def _tail_monitor(panel, root_dir, server_log, stop_event):
    root = Path(root_dir)
    files = [Path(server_log), root / "cloudflared.log"]
    state = {str(Path(server_log)): Path(server_log).stat().st_size if Path(server_log).exists() else 0}
    panel.append("realtime log monitor: running")
    panel.render()
    while not stop_event.is_set():
        any_new = False
        for path in files:
            for line in _read_new_lines(path, state, max_lines=120):
                panel.append(line)
                any_new = True
        if any_new:
            panel.render()
        time.sleep(1.0)
    panel.append("realtime log monitor: stopped")
    panel.render()


def launch_backend_dashboard(
    cfg: ServerConfig,
    *,
    tunnel_mode="both",
    ngrok_token="",
    fallback_cloudflare=True,
    warmup=True,
    keep_monitor=True,
):
    """Start server+tunnel, show endpoints, and keep realtime log monitor alive in a background thread.

    The cell returns after startup, but the widget keeps updating while the kernel is alive.
    Use the shutdown button to stop llama-server and tunnel processes.
    """
    panel = NotebookPanel("Cell 3 — server / warmup / tunnel", show_progress=False, height=430, max_lines=900)
    panel.display_panel()
    stop_event = threading.Event()
    _make_controls(panel, cfg.root_dir, stop_event)

    try:
        server_info = _start_server_panel(cfg, panel, stop_event=stop_event)

        warmup_state = {"done": False, "error": None}
        def warmup_fn():
            try:
                if warmup and not stop_event.is_set():
                    panel.section("warmup")
                    panel.set_status("warmup running")
                    panel.render()
                    chat(server_info["base_url"], cfg.alias, "ping", max_tokens=16)
                    warmup_state["done"] = True
                    panel.append("warmup: ok")
                    panel.render()
            except Exception as e:
                warmup_state["error"] = str(e)
                panel.append(f"warmup error: {e}")
                panel.render()
        warmup_thread = threading.Thread(target=warmup_fn, daemon=True)
        warmup_thread.start()

        urls = {}
        if tunnel_mode not in ("none", ""):
            panel.section("tunnel")
            if tunnel_mode in ("both", "ngrok") and ngrok_token:
                url = _start_ngrok_panel(cfg.port, ngrok_token, panel)
                if url:
                    urls["ngrok"] = url
            if tunnel_mode in ("both", "cloudflare") or (tunnel_mode == "ngrok" and fallback_cloudflare and "ngrok" not in urls):
                url = _start_cloudflare_panel(cfg.port, cfg.root_dir, panel, stop_event=stop_event)
                if url:
                    urls["cloudflare"] = url

        links = {
            "local base": {"url": server_info["base_url"], "copy": True, "open": False},
            "local chat": {"url": server_info["chat_endpoint"], "copy": True, "open": False},
            "local models": {"url": server_info["models_endpoint"], "copy": True, "open": False},
        }
        for name, url in urls.items():
            base = url.rstrip("/")
            links[f"{name} public"] = {"url": base, "copy": True, "open": True}
            links[f"{name} chat"] = {"url": base + "/v1/chat/completions", "copy": True, "open": False}
            links[f"{name} models"] = {"url": base + "/v1/models", "copy": True, "open": False}

        panel.set_links("ready endpoints", links, note="Open public URL for llama.cpp playground; copy /v1 endpoints for API clients.")
        panel.set_status("backend ready | realtime log monitor active")
        panel.append("backend ready")
        panel.render()

        monitor_thread = None
        if keep_monitor:
            monitor_thread = threading.Thread(target=_tail_monitor, args=(panel, cfg.root_dir, server_info["log_path"], stop_event), daemon=True)
            monitor_thread.start()

        return {"server": server_info, "tunnels": urls, "links": links, "panel": panel, "stop_event": stop_event, "monitor_thread": monitor_thread}
    except Exception as e:
        panel.append(f"ERROR: {type(e).__name__}: {e}")
        panel.finish(False, "launch failed")
        stop_runtime(cfg.root_dir)
        raise
