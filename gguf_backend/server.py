import os
import re
import signal
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .client import chat, get_json
from .installer import read_env_file
from .ui import live_print, tail_text
from .thinking import build_thinking_args


@dataclass
class ServerConfig:
    root_dir: str
    model_path: str
    mmproj_path: str = ""
    host: str = "0.0.0.0"
    port: int = 8080
    alias: str = "local-vl"
    ctx_size: int = 8192
    gpu_layers: int = 999
    split_mode: str = "row"
    fallback_split_mode: str = "layer"
    tensor_split: str = "1,1"
    threads: int = 4
    threads_batch: int = 4
    parallel: int = 1
    batch_size: int = 2048
    ubatch_size: int = 512
    flash_attn: bool = True
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    image_min_tokens: int | None = None
    image_max_tokens: int | None = None
    chat_template_kwargs: str | None = None
    reasoning: str | None = None
    reasoning_budget: str | None = None
    reasoning_format: str | None = None
    jinja: bool = False
    mmproj_offload: bool = True
    cuda_visible_devices: str = "0,1"
    extra_server_args: list[str] | tuple[str, ...] | None = None
    thinking_config: dict | None = None



SECRET_FLAG_RE = re.compile(r"(token|key|secret|password|passwd|auth)", re.I)
BOOLEAN_WORDS = {"on", "off", "true", "false", "1", "0", "yes", "no"}
ON_WORDS = {"on", "true", "1", "yes"}
TOGGLE_FLAGS = {"--ui": "--no-ui", "--jinja": ""}
ENABLE_ONLY_FLAGS = {"--metrics", "--props", "--slots"}


def sanitize_cmd(cmd):
    """Return a display-safe command string with secrets redacted."""
    if isinstance(cmd, str):
        parts = shlex.split(cmd)
    else:
        parts = [str(x) for x in cmd]
    redacted = []
    hide_next = False
    for part in parts:
        if hide_next:
            redacted.append("***")
            hide_next = False
            continue
        if part.startswith("--") and "=" in part:
            key, _ = part.split("=", 1)
            if SECRET_FLAG_RE.search(key):
                redacted.append(key + "=***")
            else:
                redacted.append(part)
            continue
        redacted.append(part)
        if part.startswith("-") and SECRET_FLAG_RE.search(part):
            hide_next = True
    return " ".join(shlex.quote(x) for x in redacted)


def normalize_extra_server_args(args):
    """Keep manual extra args, but fix common invalid boolean forms for toggle flags."""
    if isinstance(args, str):
        raw_items = shlex.split(args)
    else:
        raw_items = []
        for item in (args or []):
            text = str(item)
            raw_items.extend(shlex.split(text) if any(ch.isspace() for ch in text) else [text])
    items = raw_items
    out = []
    i = 0
    while i < len(items):
        flag = items[i]
        low = flag.lower()
        if low in TOGGLE_FLAGS or low in ENABLE_ONLY_FLAGS:
            next_value = items[i + 1].strip().lower() if i + 1 < len(items) else ""
            if next_value in BOOLEAN_WORDS:
                if next_value in ON_WORDS:
                    out.append(flag)
                elif low in TOGGLE_FLAGS and TOGGLE_FLAGS[low]:
                    out.append(TOGGLE_FLAGS[low])
                i += 2
                continue
            out.append(flag)
            i += 1
            continue
        out.append(flag)
        i += 1
    return out


def port_is_open(host="127.0.0.1", port=8080, timeout=1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def get_help(server, env):
    p = subprocess.run([str(server), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=40, env=env)
    return p.stdout or ""


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


def build_cmd(cfg: ServerConfig, help_text, server):
    cmd = [str(server), "-m", cfg.model_path]

    if cfg.mmproj_path and Path(cfg.mmproj_path).exists():
        cmd += ["--mmproj", cfg.mmproj_path]
        if cfg.mmproj_offload and has_flag(help_text, "--mmproj-offload"):
            cmd += ["--mmproj-offload"]

    cmd += [
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--alias", cfg.alias,
        "-c", str(cfg.ctx_size),
        "-t", str(cfg.threads),
        "-ngl", str(cfg.gpu_layers),
    ]

    if has_flag(help_text, "--parallel"):
        cmd += ["--parallel", str(cfg.parallel)]
    if has_flag(help_text, "--threads-batch"):
        cmd += ["--threads-batch", str(cfg.threads_batch)]
    if has_flag(help_text, "--batch-size"):
        cmd += ["--batch-size", str(cfg.batch_size)]
    if has_flag(help_text, "--ubatch-size"):
        cmd += ["--ubatch-size", str(cfg.ubatch_size)]
    if has_flag(help_text, "--split-mode"):
        cmd += ["--split-mode", cfg.split_mode]
    if has_flag(help_text, "--tensor-split") and cfg.split_mode != "none":
        cmd += ["--tensor-split", cfg.tensor_split]
    if cfg.flash_attn not in (None, "") and has_flag(help_text, "--flash-attn"):
        if isinstance(cfg.flash_attn, bool):
            cmd += ["--flash-attn", "on" if cfg.flash_attn else "off"]
        else:
            fa = str(cfg.flash_attn).strip().lower()
            if fa in ("on", "off", "auto"):
                cmd += ["--flash-attn", fa]
    if cfg.cache_type_k and has_flag(help_text, "--cache-type-k"):
        cmd += ["--cache-type-k", cfg.cache_type_k]
    if cfg.cache_type_v and has_flag(help_text, "--cache-type-v"):
        cmd += ["--cache-type-v", cfg.cache_type_v]
    if cfg.image_min_tokens is not None and has_flag(help_text, "--image-min-tokens"):
        cmd += ["--image-min-tokens", str(cfg.image_min_tokens)]
    if cfg.image_max_tokens is not None and has_flag(help_text, "--image-max-tokens"):
        cmd += ["--image-max-tokens", str(cfg.image_max_tokens)]
    if cfg.thinking_config:
        thinking_args, _, _ = build_thinking_args(
            cfg.thinking_config,
            help_text=help_text,
            model_path=cfg.model_path,
            alias=cfg.alias,
            existing_chat_template_kwargs=cfg.chat_template_kwargs,
            manual_reasoning=cfg.reasoning,
            manual_reasoning_budget=cfg.reasoning_budget,
            manual_reasoning_format=cfg.reasoning_format,
            manual_jinja=cfg.jinja,
        )
        cmd += thinking_args
    else:
        if cfg.jinja and has_flag(help_text, "--jinja"):
            cmd += ["--jinja"]
        if cfg.chat_template_kwargs is not None and str(cfg.chat_template_kwargs).strip() and has_flag(help_text, "--chat-template-kwargs"):
            cmd += ["--chat-template-kwargs", str(cfg.chat_template_kwargs).strip()]
        if cfg.reasoning is not None and str(cfg.reasoning).strip() and has_flag(help_text, "--reasoning"):
            cmd += ["--reasoning", str(cfg.reasoning).strip()]
        if cfg.reasoning_budget is not None and str(cfg.reasoning_budget).strip() and has_flag(help_text, "--reasoning-budget"):
            cmd += ["--reasoning-budget", str(cfg.reasoning_budget).strip()]
        if cfg.reasoning_format is not None and str(cfg.reasoning_format).strip() and has_flag(help_text, "--reasoning-format"):
            cmd += ["--reasoning-format", str(cfg.reasoning_format).strip()]
    if has_flag(help_text, "--log-colors"):
        cmd += ["--log-colors", "off"]
    if cfg.extra_server_args:
        cmd += normalize_extra_server_args(cfg.extra_server_args)

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


def wait_ready(proc, cfg: ServerConfig, log_path, timeout=240):
    base = f"http://127.0.0.1:{cfg.port}"
    start = time.time()
    last = 0.0

    while time.time() - start < timeout:
        if proc.poll() is not None:
            live_print("llama-server exited during startup\n\n" + tail_text(log_path, 120), clear=True)
            return False

        if port_is_open("127.0.0.1", cfg.port):
            try:
                status, _ = get_json(base + "/health", timeout=5)
                if status == 200:
                    return True
            except Exception:
                pass

        if time.time() - last >= 1:
            live_print(
                "\n".join([
                    "llama-server starting",
                    f"elapsed: {int(time.time() - start)}s",
                    f"port: {cfg.port}",
                    "",
                    tail_text(log_path, 20),
                ]),
                clear=True,
            )
            last = time.time()

        time.sleep(0.5)

    live_print("llama-server startup timeout\n\n" + tail_text(log_path, 120), clear=True)
    return False


def start_server(cfg: ServerConfig, *, warmup=True):
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

    help_text = get_help(server, env)
    stop_server(cfg.root_dir)

    proc, cmd, log_path = start_once(cfg, env, server, help_text)
    ok = wait_ready(proc, cfg, log_path)

    if not ok and cfg.fallback_split_mode and cfg.fallback_split_mode != cfg.split_mode:
        kill_pid(proc.pid)
        cfg.split_mode = cfg.fallback_split_mode
        proc, cmd, log_path = start_once(cfg, env, server, help_text)
        ok = wait_ready(proc, cfg, log_path)

    if not ok:
        raise RuntimeError("llama-server failed to start.")

    base = f"http://127.0.0.1:{cfg.port}"

    if warmup:
        chat(base, cfg.alias, "ping", max_tokens=16)

    live_print(
        "\n".join([
            "llama-server ready",
            f"local: {base}/v1/chat/completions",
            f"model: {cfg.alias}",
            f"pid: {proc.pid}",
        ]),
        clear=True,
    )

    return {
        "base_url": base,
        "chat_endpoint": base + "/v1/chat/completions",
        "models_endpoint": base + "/v1/models",
        "pid": proc.pid,
        "cmd": cmd,
        "log_path": str(log_path),
    }
