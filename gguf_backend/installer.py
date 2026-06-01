import json
import os
import platform
import shutil
import stat
import tarfile
import urllib.request
from pathlib import Path
from .shell import run, run_live

GITHUB_API_LLAMA_CUDA = "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest"
CLOUDFLARED_AMD64 = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_ARM64 = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"

def ensure_apt_tools():
    packages = ["aria2", "curl", "wget", "tar", "unzip", "git"]
    missing = [p for p in packages if shutil.which(p) is None]
    if missing:
        run("apt-get update -qq && apt-get install -y -qq " + " ".join(missing), check=True)

def arch_name():
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "amd64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    raise RuntimeError(f"Unsupported architecture: {m}")

def latest_llama_cuda_asset(cuda_preference="12.8"):
    req = urllib.request.Request(GITHUB_API_LLAMA_CUDA, headers={"User-Agent": "llama-cpp-notebook"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rel = json.load(r)
    arch = arch_name()
    assets = rel.get("assets", [])
    def candidates(token):
        return [a for a in assets if a["name"].lower().endswith(".tar.gz") and arch in a["name"].lower() and token in a["name"].lower()]
    picked = candidates(f"cuda-{cuda_preference}") if cuda_preference else []
    if not picked:
        picked = candidates("cuda-12")
    if not picked:
        raise RuntimeError("No CUDA prebuilt asset found in ai-dock/llama.cpp-cuda latest release.")
    return rel, picked[0]

def save_env(env_path, llama_dir, bin_dir, server):
    Path(env_path).write_text(
        f'export LLAMA_CPP_DIR="{llama_dir}"\n'
        f'export LLAMA_BIN_DIR="{bin_dir}"\n'
        f'export LLAMA_SERVER="{server}"\n'
        f'export PATH="{bin_dir}:$PATH"\n'
        f'export LD_LIBRARY_PATH="{bin_dir}:$LD_LIBRARY_PATH"\n'
    )

def install_llama_cpp_prebuilt(root_dir, *, cuda_preference="12.8", force=False):
    root_dir = Path(root_dir)
    dl_dir = root_dir / "_downloads"
    llama_dir = root_dir / "llama.cpp-cuda"
    dl_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = list(llama_dir.rglob("llama-server"))
        if existing:
            server = existing[0]
            bin_dir = server.parent
            env_path = root_dir / "llama_env.sh"
            save_env(env_path, llama_dir, bin_dir, server)
            return {"llama_dir": str(llama_dir), "bin_dir": str(bin_dir), "server": str(server), "env_path": str(env_path), "skipped": True}

    ensure_apt_tools()
    rel, asset = latest_llama_cuda_asset(cuda_preference=cuda_preference)
    url, name = asset["browser_download_url"], asset["name"]
    tar_path = dl_dir / name

    run_live(["aria2c", "-c", "-x", "8", "-s", "8", "-k", "1M", "--summary-interval=1", "-d", str(dl_dir), "-o", name, url], label=f"download llama.cpp prebuilt: {name}")

    if llama_dir.exists():
        shutil.rmtree(llama_dir)
    llama_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(llama_dir)

    servers = list(llama_dir.rglob("llama-server"))
    if not servers:
        raise RuntimeError("llama-server not found after extraction.")
    server = servers[0]
    bin_dir = server.parent
    for p in list(bin_dir.glob("llama-*")) + list(bin_dir.glob("*.so*")):
        try:
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass
    env_path = root_dir / "llama_env.sh"
    save_env(env_path, llama_dir, bin_dir, server)
    return {"tag": rel.get("tag_name"), "asset": name, "llama_dir": str(llama_dir), "bin_dir": str(bin_dir), "server": str(server), "env_path": str(env_path), "skipped": False}

def read_env_file(path):
    env = os.environ.copy()
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                k, v = line.replace("export ", "", 1).split("=", 1)
                env[k] = v.strip().strip('"')
    return env

def ensure_cloudflared(target_dir):
    existing = shutil.which("cloudflared")
    if existing:
        return existing
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    url = CLOUDFLARED_AMD64 if arch_name() == "amd64" else CLOUDFLARED_ARM64
    target = target_dir / "cloudflared"
    urllib.request.urlretrieve(url, target)
    target.chmod(0o755)
    return str(target)
