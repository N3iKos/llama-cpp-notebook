import os
import sys
import time
import json
import signal
import socket
import subprocess
from typing import Optional, Dict, Any, List
import urllib.request
import urllib.error
from .config import ServerConfig, RuntimeProfile, save_state, load_state
from .ui import live_print
from . import shell

def is_port_in_use(port: int) -> bool:
    """Checks if a local port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def kill_process_on_port(port: int) -> None:
    """Kills any process currently listening on the target port (Linux/Windows)."""
    if is_port_in_use(port):
        live_print([f"Cleaning up port {port}..."], title="Server Lifecycle", force=True)
        if sys.platform.startswith("linux"):
            try:
                # Use fuser to kill anything on TCP port
                shell.run(["fuser", "-k", f"{port}/tcp"], check=False)
                time.sleep(1.0)
            except Exception:
                pass
            try:
                # Use lsof fallback
                pids_res = shell.run(["lsof", "-t", f"-i:{port}"], check=False)
                pids = pids_res.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        shell.run(["kill", "-9", pid], check=False)
                time.sleep(1.0)
            except Exception:
                pass
        elif os.name == "nt":
            try:
                # Windows port lookup and kill
                res = shell.run(["netstat", "-ano"], check=False)
                for line in res.stdout.split("\n"):
                    if f"127.0.0.1:{port}" in line or f"0.0.0.0:{port}" in line or f"[::]:{port}" in line:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            shell.run(["taskkill", "/F", "/PID", pid], check=False)
                time.sleep(1.0)
            except Exception:
                pass


def stop_server(profile_name: str = "colab_t4x1") -> None:
    """Stops the active llama-server process and associated tunnels."""
    prof = RuntimeProfile.from_name(profile_name)
    state = load_state(prof)
    
    # Kill using stored PIDs
    server_pid = state.get("server_pid")
    if server_pid:
        live_print([f"Stopping server process PID {server_pid}..."], title="Server Lifecycle", force=True)
        try:
            if os.name == "nt":
                shell.run(["taskkill", "/F", "/PID", str(server_pid)], check=False)
            else:
                os.kill(server_pid, signal.SIGKILL)
        except Exception:
            pass
            
    # Clean up standard port 8080 just in case
    kill_process_on_port(8080)
    
    # Save stopped status
    save_state(prof, {"server_pid": None, "server_status": "stopped"})


def check_server_health(port: int = 8080) -> bool:
    """Queries the /health endpoint of the llama-server to verify readiness."""
    try:
        url = f"http://127.0.0.1:{port}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                # Health check returns status: "ok" or slot information
                status_val = data.get("status", "")
                if status_val == "ok" or "slots" in data or "loading" not in status_val:
                    return True
    except Exception:
        pass
    return False


def get_log_tail(log_path: str, lines_count: int = 120) -> List[str]:
    """Reads the final lines of the server log file for troubleshooting."""
    if not os.path.exists(log_path):
        return ["Log file does not exist."]
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-lines_count:]]
    except Exception as e:
        return [f"Failed to read logs: {str(e)}"]


def diagnose_crash(log_lines: List[str]) -> str:
    """Analyzes log lines to output common, actionable crash diagnoses."""
    log_text = "\n".join(log_lines).lower()
    if "out of memory" in log_text or "cuda oom" in log_text or "alloc" in log_text and "failed" in log_text:
        return "CRASH DIAGNOSIS: Out of Memory (OOM). Model size or context length exceeds available VRAM. Try a lower context size or a higher quant model."
    if "incompatible" in log_text or "mmproj" in log_text and "bad" in log_text:
        return "CRASH DIAGNOSIS: Multimodal Projector incompatibility. The mmproj GGUF does not match the base LLM structure."
    if "not found" in log_text or "failed to open" in log_text:
        return "CRASH DIAGNOSIS: Model file not found or corrupted. Please verify the download completed successfully."
    if "cuda" in log_text and ("driver" in log_text or "no device" in log_text or "error" in log_text):
        return "CRASH DIAGNOSIS: CUDA initialization error. Mismatch between PyTorch/CUDA environments or drivers."
    return "CRASH DIAGNOSIS: Unknown failure. Please review the detailed log tail below."


def build_server_command(
    server_path: str,
    config: ServerConfig,
    model_path: str,
    mmproj_path: Optional[str] = None
) -> List[str]:
    """Constructs the exact llama-server command arguments based on config values."""
    cmd = [
        server_path,
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", "8080",
        "-c", str(config.ctx_size),
        "-b", str(config.batch_size),
        "--ubatch-size", str(config.ubatch_size),
        "--alias", config.alias,
        "--parallel", str(config.parallel)
    ]
    
    # 1. Multimodal support
    if mmproj_path and os.path.exists(mmproj_path):
        cmd.extend(["--mmproj", mmproj_path])
        
    # 2. Flash Attention
    if config.flash_attn:
        cmd.extend(["--flash-attn"])
        
    # 3. Cache KV Types
    cmd.extend(["--cache-type-k", config.cache_type_k])
    cmd.extend(["--cache-type-v", config.cache_type_v])
    
    # 4. GPU splitting parameters (only if multiple GPUs or dual profile is used)
    if config.profile == "kaggle_t4x2" and "0,1" in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"):
        # Split mode horizontal/layer splitting
        cmd.extend(["--split-mode", config.split_mode])
        if config.tensor_split and config.split_mode == "row":
            cmd.extend(["--tensor-split", config.tensor_split])
    else:
        # Single GPU mode
        cmd.extend(["--split-mode", "none"])

    # 5. Image overrides (when explicitly set by the user)
    if config.image_min_tokens is not None:
        cmd.extend(["--image-min-tokens", str(config.image_min_tokens)])
    if config.image_max_tokens is not None:
        cmd.extend(["--image-max-tokens", str(config.image_max_tokens)])
        
    # 6. Chat template keyword args
    if config.chat_template_kwargs:
        cmd.extend(["--chat-template-kwargs", config.chat_template_kwargs])
        
    # 7. Add extra args
    if config.extra_args:
        cmd.extend(config.extra_args)
        
    return cmd


def execute_warmup(port: int = 8080) -> Dict[str, Any]:
    """Sends a quick warmup completion prompt to the server to pre-allocate memory."""
    payload = {
        "model": "local-vl",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
    
    start_time = time.time()
    try:
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(req, data=json.dumps(payload).encode(), timeout=15.0) as response:
            latency = time.time() - start_time
            if response.status == 200:
                res_data = json.loads(response.read().decode())
                text = res_data["choices"][0]["message"]["content"]
                return {"success": True, "latency": latency, "response": text}
    except Exception as e:
        return {"success": False, "error": str(e)}
        
    return {"success": False, "error": "Unknown warmup failure"}


def start_and_warmup(config: ServerConfig) -> Dict[str, Any]:
    """Starts the llama-server with automatic fallback options and executes a warmup query."""
    prof = RuntimeProfile.from_name(config.profile)
    state = load_state(prof)
    
    model_path = state.get("model_path")
    if not model_path or not os.path.exists(model_path):
        # Scan models directory if state is lost
        model_dir = prof.model_dir
        if os.path.exists(model_dir):
            ggufs = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".gguf") and "mmproj" not in f]
            if ggufs:
                model_path = ggufs[0]
                
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Please run Task 2 downloader first.")
        
    mmproj_path = state.get("mmproj_path")
    
    # 1. Clean previous runs
    stop_server(config.profile)
    
    server_bin = os.path.join(prof.bin_dir, "llama-server")
    if not os.path.exists(server_bin):
        # Fallback search on path
        server_bin = shutil.which("llama-server") or "llama-server"
        
    # Construct base log paths
    os.makedirs(prof.log_dir, exist_ok=True)
    log_path = os.path.join(prof.log_dir, "server.log")
    
    # Clear old log files
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception:
            pass
            
    # Set visible devices
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = prof.cuda_visible_devices
    
    # Try startup cycle (with row -> layer fallback if kaggle row split mode fails)
    split_modes_to_try = [config.split_mode]
    if config.profile == "kaggle_t4x2" and config.split_mode == "row" and config.fallback_split_mode:
        split_modes_to_try.append(config.fallback_split_mode)
        
    process = None
    success = False
    active_cmd = []
    
    for current_split in split_modes_to_try:
        config.split_mode = current_split
        cmd = build_server_command(server_bin, config, model_path, mmproj_path)
        active_cmd = cmd
        
        live_print([
            f"Launching llama-server using split-mode: {current_split}...",
            f"Command: {' '.join(cmd)}"
        ], title="Server Launch", force=True)
        
        # Open server log file in write mode
        log_file = open(log_path, "w", encoding="utf-8")
        
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env_vars
        )
        
        # Monitor startup for up to 120 seconds
        start_wait = time.time()
        crashed = False
        
        while time.time() - start_wait < 120.0:
            # Check if background process exited prematurely
            if process.poll() is not None:
                crashed = True
                break
                
            # Perform health check query
            if check_server_health(8080):
                success = True
                break
                
            time.sleep(1.0)
            
        if success:
            log_file.close()
            break
            
        # Startup failed for this split mode - clean up
        process.terminate()
        process.wait()
        log_file.close()
        
        tail_lines = get_log_tail(log_path, 40)
        
        live_print([
            f"WARNING: Startup failed using split-mode {current_split}.",
            "Tail of log output:",
            *tail_lines,
            "-------------------"
        ], title="Server Fallback Manager", force=True)
        
    if not success:
        # Full crash diagnostic extraction
        tail_lines = get_log_tail(log_path, 120)
        diagnosis = diagnose_crash(tail_lines)
        
        error_report = [
            "CRITICAL: Failed to launch inference server.",
            f"Active Command: {' '.join(active_cmd)}",
            f"Log path: {log_path}",
            f"Selected GPU Profile: {config.profile}",
            "",
            diagnosis,
            "",
            "=== SERVER LOG TAIL (LAST 120 LINES) ===",
            *tail_lines
        ]
        live_print(error_report, title="Server Failure Report", force=True)
        raise RuntimeError("Inference server failed to startup successfully.")
        
    # Start was successful - save details and run warmup
    save_state(prof, {
        "server_pid": process.pid,
        "server_status": "running",
        "server_port": 8080,
        "model_alias": config.alias
    })
    
    live_print([
        "Server is loaded and HEALTHY!",
        "Executing warmup inference completion..."
    ], title="Server Ready", force=True)
    
    warmup_res = execute_warmup(8080)
    
    results = [
        "Server startup cycle finished successfully!",
        f"PID: {process.pid}",
        f"Port: 8080",
        f"Warmup Successful: {warmup_res.get('success')}",
    ]
    if warmup_res.get("success"):
        results.append(f"Warmup Latency: {warmup_res.get('latency'):.2f} seconds")
        results.append(f"Response: '{warmup_res.get('response')}'")
    else:
        results.append(f"Warmup Error: {warmup_res.get('error')}")
        
    live_print(results, title="Startup & Warmup Success", force=True)
    
    return {
        "pid": process.pid,
        "port": 8080,
        "warmup": warmup_res
    }


def status(profile_name: str = "colab_t4x1") -> Dict[str, Any]:
    """Retrieves current server operational status metrics."""
    prof = RuntimeProfile.from_name(profile_name)
    state = load_state(prof)
    
    pid = state.get("server_pid")
    running = False
    
    if pid:
        # Check if process actually exists
        try:
            if os.name == "nt":
                # Check process list on Windows
                res = shell.run(["tasklist", "/FI", f"PID eq {pid}"], check=False)
                running = str(pid) in res.stdout
            else:
                # Signal 0 checks process existence on POSIX
                os.kill(pid, 0)
                running = True
        except Exception:
            pass
            
    if running and check_server_health(8080):
        return {
            "status": "running",
            "pid": pid,
            "port": 8080,
            "model_alias": state.get("model_alias")
        }
        
    return {"status": "stopped", "pid": None}
