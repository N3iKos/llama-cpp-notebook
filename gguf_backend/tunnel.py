import os
import re
import sys
import time
import subprocess
from typing import Optional
from .config import RuntimeProfile, save_state, load_state
from .ui import live_print
from . import installer

def get_ngrok_token() -> str:
    """Attempts to retrieve Ngrok token from environment or Kaggle Secrets."""
    # 1. Check environment
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if token:
        return token
        
    # 2. Check Kaggle Secrets dynamically to avoid import errors on Colab
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("NGROK_AUTHTOKEN").strip()
        if token:
            return token
    except Exception:
        pass
        
    return ""


def start_ngrok(port: int, token: str) -> str:
    """Installs pyngrok, configures token, and starts public HTTP tunnel."""
    installer.install_pyngrok()
    
    from pyngrok import ngrok, conf
    
    live_print(["Configuring Ngrok authenticator token..."], title="Ngrok Tunnel", force=True)
    ngrok.set_auth_token(token)
    
    # Expose port
    live_print([f"Opening Ngrok tunnel on port {port}..."], title="Ngrok Tunnel", force=True)
    tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    
    # Ensure https representation
    if public_url.startswith("http://"):
        public_url = public_url.replace("http://", "https://")
        
    return public_url


def start_cloudflare(profile: RuntimeProfile, port: int) -> str:
    """Downloads cloudflared if missing, launches Quick Tunnel, and parses target URL."""
    cf_bin = installer.install_cloudflared(profile)
    
    os.makedirs(profile.log_dir, exist_ok=True)
    log_path = os.path.join(profile.log_dir, "cloudflare.log")
    
    # Clean previous cloudflare log
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception:
            pass
            
    # Launch background Quick Tunnel
    cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    
    live_print(["Launching Cloudflare Quick Tunnel background process...", f"Command: {' '.join(cmd)}"], title="Cloudflare Tunnel", force=True)
    
    log_file = open(log_path, "w", encoding="utf-8")
    
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Store PID in runtime state immediately so it can be cleaned up on stop
    save_state(profile, {"tunnel_pid": process.pid})
    
    # Scan log output for the trycloudflare URL (timeout 30s)
    start_wait = time.time()
    public_url = ""
    
    regex = r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
    
    while time.time() - start_wait < 30.0:
        if process.poll() is not None:
            log_file.close()
            raise RuntimeError(f"cloudflared process terminated prematurely with exit code {process.returncode}.")
            
        # Read log file content
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    matches = re.findall(regex, content)
                    if matches:
                        public_url = matches[0]
                        break
            except Exception:
                pass
                
        time.sleep(1.0)
        
    log_file.close()
    
    if not public_url:
        process.terminate()
        process.wait()
        # Fetch log tail for debugging
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                tail = f.read().split("\n")[-15:]
        except Exception:
            tail = ["Failed to read cloudflare logs."]
        raise TimeoutError(f"Cloudflare Tunnel failed to initialize within 30s.\nLog Tail:\n" + "\n".join(tail))
        
    return public_url


def open_tunnel(port: int = 8080, ngrok_token: Optional[str] = None, prefer: str = "ngrok", fallback: str = "cloudflare") -> str:
    """Establishes a public web tunnel to the local server port.
    
    Prioritizes Ngrok if token is provided/detected, otherwise falls back to Cloudflare.
    """
    profile_name = "kaggle_t4x2" if "kaggle" in os.getcwd().lower() else "colab_t4x1"
    profile = RuntimeProfile.from_name(profile_name)
    
    # Resolve token
    token = (ngrok_token or get_ngrok_token() or "").strip()
    
    public_url = ""
    provider_used = ""
    
    if token and prefer == "ngrok":
        try:
            public_url = start_ngrok(port, token)
            provider_used = "Ngrok"
        except Exception as e:
            if fallback == "cloudflare":
                live_print([f"WARNING: Ngrok launch failed: {str(e)}", "Falling back to Cloudflare Quick Tunnel..."], title="Tunnel Router", force=True)
            else:
                raise e
                
    if not public_url:
        public_url = start_cloudflare(profile, port)
        provider_used = "Cloudflare Quick Tunnel"
        
    # Save state
    save_state(profile, {
        "public_url": public_url,
        "tunnel_provider": provider_used
    })
    
    # Formatting endpoint outputs exactly as requested by PRD
    output_lines = [
        "Public Tunnel Active!",
        f"Provider: {provider_used}",
        f"PUBLIC_URL: {public_url}",
        f"CHAT_ENDPOINT: {public_url}/v1/chat/completions"
    ]
    live_print(output_lines, title="Tunnel Exposer", force=True)
    
    return public_url
