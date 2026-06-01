import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .client import chat, get_json
from .installer import read_env_file
from .ui import live_print, tail_text


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
    mmproj_offload: bool = True
    cuda_visible_devices: str = "0,1"


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
    if cfg.flash_attn and has_flag(help_text, "--flash-attn"):
        cmd += ["--flash-attn", "on"]
    if cfg.cache_type_k and has_flag(help_text, "--cache-type-k"):
        cmd += ["--cache-type-k", cfg.cache_type_k]
    if cfg.cache_type_v and has_flag(help_text, "--cache-type-v"):
        cmd += ["--cache-type-v", cfg.cache_type_v]
    if cfg.image_min_tokens is not None and has_flag(help_text, "--image-min-tokens"):
        cmd += ["--image-min-tokens", str(cfg.image_min_tokens)]
    if cfg.image_max_tokens is not None and has_flag(help_text, "--image-max-tokens"):
        cmd += ["--image-max-tokens", str(cfg.image_max_tokens)]
    if cfg.chat_template_kwargs is not None and has_flag(help_text, "--chat-template-kwargs"):
        cmd += ["--chat-template-kwargs", cfg.chat_template_kwargs]
    if has_flag(help_text, "--log-colors"):
        cmd += ["--log-colors", "off"]

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
