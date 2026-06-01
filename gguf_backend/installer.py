import os
import sys
import shutil
import urllib.request
import json
import tarfile
from typing import Optional, Dict, Any
from .config import RuntimeProfile
from .ui import live_print
from . import shell

FALLBACK_BUILD_TAG = "b3145"
FALLBACK_CUDA_VER = "12.8"

def install_system_dependencies() -> None:
    """Installs system dependencies like aria2 if on a debian-based Linux system."""
    if sys.platform.startswith("linux"):
        if shutil.which("aria2c") is None:
            live_print(["Installing aria2 via apt-get..."], title="System Setup", force=True)
            try:
                # Kaggle/Colab run as root with passwordless sudo
                shell.run(["sudo", "apt-get", "update", "-qq"])
                shell.run(["sudo", "apt-get", "install", "-y", "-qq", "aria2"])
                live_print(["aria2 installation: SUCCESS"], title="System Setup", force=True)
            except Exception as e:
                live_print([f"WARNING: Failed to install aria2 via apt-get: {str(e)}", "Will try direct execution of fallback paths."], title="System Setup", force=True)


def get_latest_llama_release_url(cuda_version: str = "12") -> str:
    """Queries GitHub API to find the latest prebuilt CUDA release URL, with hardcoded fallback."""
    # Build tag and download URL fallbacks in case of GitHub rate limiting
    fallback_url = f"https://github.com/ai-dock/llama.cpp-cuda/releases/download/{FALLBACK_BUILD_TAG}/llama.cpp-{FALLBACK_BUILD_TAG}-cuda-{FALLBACK_CUDA_VER}-amd64.tar.gz"
    
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5.0) as r:
            data = json.loads(r.read().decode())
            
        assets = data.get("assets", [])
        for asset in assets:
            name = asset.get("name", "")
            # We want amd64 and matching CUDA version (e.g. cuda-12)
            if "amd64" in name and f"cuda-{cuda_version}" in name and name.endswith(".tar.gz"):
                url = asset.get("browser_download_url", "")
                if url:
                    return url
    except Exception as e:
        # Fallback to hardcoded URL on rate-limiting or network issues
        pass
        
    return fallback_url


def install_llama_server(profile: RuntimeProfile) -> str:
    """Downloads and extracts prebuilt llama-server CUDA binary to runtime workspace."""
    os.makedirs(profile.bin_dir, exist_ok=True)
    os.makedirs(profile.work_dir, exist_ok=True)
    
    server_path = os.path.join(profile.bin_dir, "llama-server")
    if os.path.exists(server_path) and os.access(server_path, os.X_OK):
        return server_path
        
    live_print(["Fetching prebuilt llama-server CUDA binary URL..."], title="Runtime Installer", force=True)
    download_url = get_latest_llama_release_url()
    
    filename = download_url.split("/")[-1]
    tar_path = os.path.join(profile.work_dir, filename)
    
    live_print([f"Downloading {filename}...", f"URL: {download_url}"], title="Runtime Installer", force=True)
    
    # Download tarball using urllib (fallback if aria2c not ready yet)
    try:
        urllib.request.urlretrieve(download_url, tar_path)
    except Exception as e:
        # If urllib retrieval fails, let's try with curl
        try:
            shell.run(["curl", "-L", "-o", tar_path, download_url])
        except Exception as curl_err:
            raise RuntimeError(f"Failed to download prebuilt llama-server: {str(e)} / {str(curl_err)}")
            
    live_print([f"Extracting {filename} to {profile.bin_dir}..."], title="Runtime Installer", force=True)
    
    # Extract only the executable binaries we need to save disk and speed up
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if "llama-server" in member.name or "llama-cli" in member.name:
                    # Flatten target directory structure if needed, or extract
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=profile.bin_dir)
    finally:
        # Clean up tarball immediately to save disk
        if os.path.exists(tar_path):
            os.remove(tar_path)
            
    # Add execution permissions
    for bin_name in ["llama-server", "llama-cli"]:
        p = os.path.join(profile.bin_dir, bin_name)
        if os.path.exists(p):
            os.chmod(p, 0o755)
            
    if not os.path.exists(server_path):
        raise RuntimeError("llama-server binary was not found in the extracted files.")
        
    return server_path


def install_cloudflared(profile: RuntimeProfile) -> str:
    """Downloads official cloudflared binary to local workspace if not present."""
    os.makedirs(profile.bin_dir, exist_ok=True)
    
    bin_name = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    cf_path = os.path.join(profile.bin_dir, bin_name)
    
    if os.path.exists(cf_path) and os.access(cf_path, os.X_OK):
        return cf_path
        
    live_print(["Downloading cloudflared for public tunneling..."], title="Tunnel Installer", force=True)
    
    if os.name == "nt":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    else:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        
    try:
        urllib.request.urlretrieve(url, cf_path)
        if os.name != "nt":
            os.chmod(cf_path, 0o755)
    except Exception as e:
        # Try downloading with curl as fallback
        try:
            shell.run(["curl", "-L", "-o", cf_path, url])
            if os.name != "nt":
                os.chmod(cf_path, 0o755)
        except Exception as curl_err:
            raise RuntimeError(f"Failed to download cloudflared: {str(e)} / {str(curl_err)}")
            
    return cf_path


def install_pyngrok() -> None:
    """Installs pyngrok package dynamically in runtime if not already installed."""
    try:
        import pyngrok
    except ImportError:
        live_print(["Installing pyngrok via pip..."], title="Dependency Installer", force=True)
        shell.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])


def install_runtime(profile: str = "colab_t4x1") -> Dict[str, Any]:
    """Orchestrates total runtime installation flow."""
    prof = RuntimeProfile.from_name(profile)
    
    # 1. Install system utilities
    install_system_dependencies()
    
    # 2. Download and extract llama-server
    server_path = install_llama_server(prof)
    
    # 3. Ensure cloudflared is present
    cf_path = install_cloudflared(prof)
    
    # 4. Verify binary installations
    try:
        ver_res = shell.run([server_path, "--version"])
        server_ver = ver_res.stdout.strip()
    except Exception as e:
        server_ver = f"Failed to get version: {str(e)}"
        
    try:
        dev_res = shell.run([server_path, "--list-devices"])
        devices = dev_res.stdout.strip()
    except Exception:
        devices = "Failed to list devices (expected on CPU/non-CUDA runtimes)"
        
    results = [
        "All Binaries Ready!",
        f"llama-server: {server_ver}",
        f"Devices: {devices}",
        f"cloudflared: OK"
    ]
    
    live_print(results, title="Runtime Setup Complete", force=True)
    
    return {
        "server_path": server_path,
        "cloudflared_path": cf_path,
        "server_version": server_ver,
        "devices": devices
    }
