import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .client import chat, get_json
from .installer import read_env_file
from .panel import NotebookPanel, show_summary, tail_text


@dataclass
class ServerConfig:
    root_dir: str
    model_path: str
    mmproj_path: str = ""

    # General config. These are intentionally explicit in the notebook.
    host: str = "0.0.0.0"
    port: int | str = 8080
    alias: str = "local-vl"
    ctx_size: int | str = 8192
    gpu_layers: int | str = 999
    cuda_visible_devices: str = "0,1"
    split_mode: str = "row"
    fallback_split_mode: str = "layer"
    tensor_split: str = "1,1"

    # Advanced config. Empty string means: do not pass the flag; use llama.cpp default.
    main_gpu: int | str = ""
    threads: int | str = ""
    threads_batch: int | str = ""
    threads_http: int | str = ""
    parallel: int | str = ""
    batch_size: int | str = ""
    ubatch_size: int | str = ""
    flash_attn: bool | str = ""
    cache_type_k: str = ""
    cache_type_v: str = ""
    kv_offload: bool | str = ""
    cont_batching: bool | str = ""
    cache_prompt: bool | str = ""
    cache_reuse: int | str = ""
    mmap: bool | str = ""
    mlock: bool | str = ""
    no_perf: bool | str = ""
    log_verbosity: int | str = ""

    # Multimodal / template / reasoning. Empty string keeps llama.cpp defaults.
    mmproj_offload: bool | str = ""
    image_min_tokens: int | str = ""
    image_max_tokens: int | str = ""
    chat_template_kwargs: str = ""
    chat_template: str = ""
    chat_template_file: str = ""
    jinja: bool | str = ""
    reasoning: str = ""
    reasoning_format: str = ""
    reasoning_budget: int | str = ""
    reasoning_budget_message: str = ""

    # Server/API features. Empty string keeps llama.cpp defaults.
    timeout: int | str = ""
    api_key: str = ""
    api_key_file: str = ""
    api_prefix: str = ""
    ui: bool | str = ""
    metrics: bool | str = ""
    slots: bool | str = ""
    props: bool | str = ""
    embedding: bool | str = ""
    reranking: bool | str = ""
    slot_save_path: str = ""
    media_path: str = ""

    # Escape hatch for new llama.cpp flags not wrapped yet.
    extra_server_args: list[str] | tuple[str, ...] | str | None = None

    # Model-aware reasoning/thinking toggle.
    # When set, the thinking mapper auto-detects the model family and generates
    # the correct --reasoning, --chat-template-kwargs, and --jinja flags.
    # Manual overrides (reasoning, reasoning_budget, etc.) still win.
    thinking_config: dict | None = None


def port_is_open(host="127.0.0.1", port=8080, timeout=1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def get_help(server, env):
    p = subprocess.Popen([str(server), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        out, _ = p.communicate(timeout=40)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
    return out or ""


def has_flag(help_text, flag):
    return flag in help_text


def kill_pid(pid):
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def stop_server(root_dir):
    root = Path(root_dir)
    pid_path = root / "llama_server.pid"

    if pid_path.exists():
        try:
            kill_pid(int(pid_path.read_text().strip()))
        except Exception:
            pass

    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmdline = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if "llama-server" in cmdline and str(root) in cmdline:
            kill_pid(int(p.name))


def shutdown_all(root_dir):
    """Kill llama-server, cloudflared, and ngrok processes for a full shutdown.

    This is the single entry point for the shutdown button. It stops all
    components that ``launch_backend`` / ``launch_backend_live`` started.

    Returns:
        dict with shutdown status for each component.
    """
    root = Path(root_dir)
    status = {}

    # 1. Kill llama-server
    try:
        stop_server(root_dir)
        status["llama_server"] = "stopped"
    except Exception as e:
        status["llama_server"] = f"error: {e}"

    # 2. Kill cloudflared (PID file written by tunnel.py)
    cf_pid_path = root / "cloudflared.pid"
    if cf_pid_path.exists():
        try:
            kill_pid(int(cf_pid_path.read_text().strip()))
            status["cloudflared"] = "stopped"
        except Exception as e:
            status["cloudflared"] = f"error: {e}"
    else:
        status["cloudflared"] = "not running"

    # 3. Kill ngrok (via pyngrok if available)
    try:
        from pyngrok import ngrok
        ngrok.kill()
        status["ngrok"] = "stopped"
    except ImportError:
        status["ngrok"] = "pyngrok not installed"
    except Exception as e:
        status["ngrok"] = f"error: {e}"

    return status


def _is_set(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _boolish(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if v in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
    return bool(value)


def _onoff(value):
    return "on" if _boolish(value) else "off"


def _has_any(help_text, *flags):
    return any(flag in help_text for flag in flags)


def _add_value(cmd, help_text, flag, value, *aliases):
    if _is_set(value) and _has_any(help_text, flag, *aliases):
        cmd += [flag, str(value)]


def _add_onoff(cmd, help_text, flag, value, *aliases):
    if _is_set(value) and _has_any(help_text, flag, *aliases):
        cmd += [flag, _onoff(value)]


def _add_bool_flag(cmd, help_text, flag, value, *, no_flag=None):
    if not _is_set(value):
        return
    enabled = _boolish(value)
    if enabled and has_flag(help_text, flag):
        cmd.append(flag)
    elif not enabled and no_flag and has_flag(help_text, no_flag):
        cmd.append(no_flag)


def _extend_extra_args(cmd, extra_args):
    if not _is_set(extra_args):
        return
    if isinstance(extra_args, str):
        cmd.extend(extra_args.split())
    else:
        cmd.extend(str(x) for x in extra_args)


def build_cmd(cfg: ServerConfig, help_text, server):
    cmd = [str(server), "-m", cfg.model_path]

    if cfg.mmproj_path and Path(cfg.mmproj_path).exists():
        cmd += ["--mmproj", cfg.mmproj_path]
        if _is_set(cfg.mmproj_offload):
            _add_bool_flag(cmd, help_text, "--mmproj-offload", cfg.mmproj_offload)

    # General config. If user sets an empty string, skip the flag and use llama.cpp default.
    _add_value(cmd, help_text, "--host", cfg.host)
    _add_value(cmd, help_text, "--port", cfg.port)
    _add_value(cmd, help_text, "--alias", cfg.alias)
    _add_value(cmd, help_text, "-c", cfg.ctx_size, "--ctx-size")
    _add_value(cmd, help_text, "-ngl", cfg.gpu_layers, "--n-gpu-layers")
    _add_value(cmd, help_text, "--split-mode", cfg.split_mode)
    if _is_set(cfg.tensor_split) and (not _is_set(cfg.split_mode) or str(cfg.split_mode) != "none"):
        _add_value(cmd, help_text, "--tensor-split", cfg.tensor_split, "-ts")

    # Advanced performance / memory.
    _add_value(cmd, help_text, "--main-gpu", cfg.main_gpu, "-mg")
    _add_value(cmd, help_text, "-t", cfg.threads, "--threads")
    _add_value(cmd, help_text, "--threads-batch", cfg.threads_batch)
    _add_value(cmd, help_text, "--threads-http", cfg.threads_http)
    _add_value(cmd, help_text, "--parallel", cfg.parallel)
    _add_value(cmd, help_text, "--batch-size", cfg.batch_size)
    _add_value(cmd, help_text, "--ubatch-size", cfg.ubatch_size)
    _add_onoff(cmd, help_text, "--flash-attn", cfg.flash_attn)
    _add_value(cmd, help_text, "--cache-type-k", cfg.cache_type_k)
    _add_value(cmd, help_text, "--cache-type-v", cfg.cache_type_v)
    _add_onoff(cmd, help_text, "--kv-offload", cfg.kv_offload)
    _add_bool_flag(cmd, help_text, "--cont-batching", cfg.cont_batching, no_flag="--no-cont-batching")
    _add_bool_flag(cmd, help_text, "--cache-prompt", cfg.cache_prompt, no_flag="--no-cache-prompt")
    _add_value(cmd, help_text, "--cache-reuse", cfg.cache_reuse)
    _add_bool_flag(cmd, help_text, "--mmap", cfg.mmap, no_flag="--no-mmap")
    _add_bool_flag(cmd, help_text, "--mlock", cfg.mlock, no_flag="--no-mlock")
    _add_bool_flag(cmd, help_text, "--no-perf", cfg.no_perf)
    _add_value(cmd, help_text, "--verbosity", cfg.log_verbosity, "--log-verbosity")

    # Multimodal / template / reasoning.
    _add_value(cmd, help_text, "--image-min-tokens", cfg.image_min_tokens)
    _add_value(cmd, help_text, "--image-max-tokens", cfg.image_max_tokens)
    _add_value(cmd, help_text, "--chat-template-kwargs", cfg.chat_template_kwargs)
    _add_value(cmd, help_text, "--chat-template", cfg.chat_template)
    _add_value(cmd, help_text, "--chat-template-file", cfg.chat_template_file)
    _add_bool_flag(cmd, help_text, "--jinja", cfg.jinja, no_flag="--no-jinja")
    _add_value(cmd, help_text, "--reasoning", cfg.reasoning, "-rea")
    _add_value(cmd, help_text, "--reasoning-format", cfg.reasoning_format)
    _add_value(cmd, help_text, "--reasoning-budget", cfg.reasoning_budget)
    _add_value(cmd, help_text, "--reasoning-budget-message", cfg.reasoning_budget_message)

    # Server/API features.
    _add_value(cmd, help_text, "--timeout", cfg.timeout, "-to")
    _add_value(cmd, help_text, "--api-key", cfg.api_key)
    _add_value(cmd, help_text, "--api-key-file", cfg.api_key_file)
    _add_value(cmd, help_text, "--api-prefix", cfg.api_prefix)
    _add_bool_flag(cmd, help_text, "--ui", cfg.ui, no_flag="--no-ui")
    _add_bool_flag(cmd, help_text, "--metrics", cfg.metrics)
    _add_bool_flag(cmd, help_text, "--slots", cfg.slots, no_flag="--no-slots")
    _add_bool_flag(cmd, help_text, "--props", cfg.props)
    _add_bool_flag(cmd, help_text, "--embedding", cfg.embedding)
    _add_bool_flag(cmd, help_text, "--reranking", cfg.reranking, no_flag="--no-reranking")
    _add_value(cmd, help_text, "--slot-save-path", cfg.slot_save_path)
    _add_value(cmd, help_text, "--media-path", cfg.media_path)

    if has_flag(help_text, "--log-colors"):
        cmd += ["--log-colors", "off"]

    # Apply thinking mapper (if thinking_config is set).
    # This runs AFTER all manual user flags are in cmd, but BEFORE
    # extra_server_args, so the escape hatch can still override.
    if cfg.thinking_config:
        from .thinking import apply_thinking
        cfg._thinking_result = apply_thinking(cfg, help_text, cmd)

    _extend_extra_args(cmd, cfg.extra_server_args)
    return cmd


def start_once(cfg: ServerConfig, env, server, help_text):
    root = Path(cfg.root_dir)
    log_path = root / "llama_server.log"
    pid_path = root / "llama_server.pid"
    cmd = build_cmd(cfg, help_text, server)

    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(root), text=True)
    pid_path.write_text(str(proc.pid))

    return proc, cmd, log_path


def wait_ready(proc, cfg: ServerConfig, log_path, timeout=240, panel=None):
    base = f"http://127.0.0.1:{cfg.port}"
    start = time.time()
    last = 0.0
    own_panel = panel is None
    if panel is None:
        panel = NotebookPanel("llama-server", show_progress=False, height=330)
    panel.display_panel()
    panel.set_progress_visible(False)
    panel.set_footer(f"log: {log_path}")

    while time.time() - start < timeout:
        elapsed = int(time.time() - start)
        if proc.poll() is not None:
            panel.lines = ["llama-server exited during startup", "", *tail_text(log_path, 120).splitlines()]
            if own_panel:
                panel.finish(False, f"server exited | elapsed {elapsed}s")
            else:
                panel.set_status(f"server exited | elapsed {elapsed}s")
                panel.render()
            return False

        if port_is_open("127.0.0.1", cfg.port):
            try:
                status, _ = get_json(base + "/health", timeout=5)
                if status == 200:
                    panel.lines = [
                        f"ready: {base}/v1/chat/completions",
                        f"port: {cfg.port}",
                        f"elapsed: {elapsed}s",
                        "",
                        *tail_text(log_path, 20).splitlines(),
                    ]
                    if own_panel:
                        panel.finish(True, "llama-server ready")
                    else:
                        panel.set_status("llama-server ready")
                        panel.render()
                    return True
            except Exception:
                pass

        if time.time() - last >= 0.5:
            panel.set_status(f"starting | elapsed {elapsed}s | port {cfg.port}")
            panel.lines = [f"base: {base}", f"pid: {proc.pid}", "", *tail_text(log_path, 60).splitlines()]
            panel.render()
            last = time.time()

        time.sleep(0.5)

    panel.lines = ["llama-server startup timeout", "", *tail_text(log_path, 120).splitlines()]
    if own_panel:
        panel.finish(False, f"startup timeout after {timeout}s")
    else:
        panel.set_status(f"startup timeout after {timeout}s")
        panel.render()
    return False

def start_server(cfg: ServerConfig, *, warmup=True, panel=None):
    root = Path(cfg.root_dir)
    env = read_env_file(root / "llama_env.sh")

    server = Path(env.get("LLAMA_SERVER", ""))
    bin_dir = Path(env.get("LLAMA_BIN_DIR", ""))

    if not server.exists():
        raise RuntimeError(f"LLAMA_SERVER not found: {server}")

    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    if _is_set(cfg.cuda_visible_devices):
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    help_text = get_help(server, env)
    stop_server(cfg.root_dir)

    proc, cmd, log_path = start_once(cfg, env, server, help_text)
    ok = wait_ready(proc, cfg, log_path, panel=panel)

    if not ok and cfg.fallback_split_mode and cfg.fallback_split_mode != cfg.split_mode:
        kill_pid(proc.pid)
        cfg.split_mode = cfg.fallback_split_mode
        proc, cmd, log_path = start_once(cfg, env, server, help_text)
        ok = wait_ready(proc, cfg, log_path, panel=panel)

    if not ok:
        raise RuntimeError("llama-server failed to start.")

    base = f"http://127.0.0.1:{cfg.port}"

    if warmup:
        if panel:
            panel.section("warmup")
            panel.set_status("warmup request...")
        chat(base, cfg.alias, "ping", max_tokens=16)
        if panel:
            panel.append("warmup: ok")
            panel.render()

    ready_lines = [
        f"local: {base}/v1/chat/completions",
        f"model: {cfg.alias}",
        f"pid: {proc.pid}",
        f"log: {log_path}",
    ]
    if panel:
        panel.set_summary("llama-server ready", lines=ready_lines)
    else:
        show_summary("llama-server ready", lines=ready_lines, log_path=log_path)

    return {
        "base_url": base,
        "chat_endpoint": base + "/v1/chat/completions",
        "models_endpoint": base + "/v1/models",
        "pid": proc.pid,
        "cmd": cmd,
        "log_path": str(log_path),
    }
