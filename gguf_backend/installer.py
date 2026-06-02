import json
import os
import platform
import shutil
import stat
import tarfile
import urllib.request
from pathlib import Path

from .panel import run_command, show_summary

GITHUB_API_LLAMA_CUDA = "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest"
CLOUDFLARED_AMD64 = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_ARM64 = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"


def ensure_apt_tools(panel=None):
    packages = ["aria2", "curl", "wget", "tar", "unzip"]
    missing = [p for p in packages if shutil.which(p) is None]
    if missing:
        run_command(
            "apt-get update -qq && apt-get install -y -qq " + " ".join(missing),
            check=True,
            label="install apt tools",
            panel=panel,
            finalize=False if panel else True,
            tail_lines=160,
        )
    elif panel:
        panel.append("apt tools: already available")
        panel.render()


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

    def candidates_for(token):
        out = []
        for asset in assets:
            name = asset["name"].lower()
            if name.endswith(".tar.gz") and arch in name and token in name:
                out.append(asset)
        return out

    candidates = candidates_for(f"cuda-{cuda_preference}") if cuda_preference else []
    if not candidates:
        candidates = candidates_for("cuda-12")
    if not candidates:
        raise RuntimeError("No CUDA prebuilt asset found.")

    return rel, candidates[0]


def save_env(env_path, llama_dir, bin_dir, server):
    llama_dir = Path(llama_dir)
    bin_dir = Path(bin_dir)
    so_dirs = sorted({str(p.parent) for p in llama_dir.rglob("*.so*")})
    lib_path = ":".join([str(bin_dir), *so_dirs, "$LD_LIBRARY_PATH"])
    Path(env_path).write_text(
        f'export LLAMA_CPP_DIR="{llama_dir}"\n'
        f'export LLAMA_BIN_DIR="{bin_dir}"\n'
        f'export LLAMA_SERVER="{server}"\n'
        f'export PATH="{bin_dir}:$PATH"\n'
        f'export LD_LIBRARY_PATH="{lib_path}"\n'
    )


def install_llama_cpp_prebuilt(root_dir, *, cuda_preference="12.8", force=False, panel=None):
    root_dir = Path(root_dir)
    dl_dir = root_dir / "_downloads"
    llama_dir = root_dir / "llama.cpp-cuda"
    dl_dir.mkdir(parents=True, exist_ok=True)

    if panel:
        panel.section("install llama.cpp prebuilt")

    if not force:
        existing = list(llama_dir.rglob("llama-server"))
        if existing:
            server = existing[0]
            bin_dir = server.parent
            env_path = root_dir / "llama_env.sh"
            save_env(env_path, llama_dir, bin_dir, server)
            info = {"llama_dir": str(llama_dir), "bin_dir": str(bin_dir), "server": str(server), "skipped": True}
            if panel:
                panel.set_summary("llama.cpp prebuilt", data=info)
                panel.append(f"llama.cpp prebuilt: already installed at {server}")
                panel.render()
            else:
                show_summary("llama.cpp prebuilt", info)
            return info

    rel, asset = latest_llama_cuda_asset(cuda_preference=cuda_preference)
    url = asset["browser_download_url"]
    name = asset["name"]
    tar_path = dl_dir / name

    ensure_apt_tools(panel=panel)
    run_command(
        [
            "aria2c", "-c", "-x", "8", "-s", "8", "-k", "1M",
            "--summary-interval=1", "--console-log-level=warn", "--show-console-readout=true",
            "--allow-overwrite=true", "--auto-file-renaming=false",
            "-d", str(dl_dir), "-o", name, url,
        ],
        label=f"download llama.cpp prebuilt: {name}",
        mode="download",
        check=True,
        panel=panel,
        finalize=False if panel else True,
        log_name=f"download_{name}.log".replace("/", "_"),
        tail_lines=140,
        refresh_interval=0.10,
    )

    if panel:
        panel.set_progress_visible(False)
        panel.set_status("extracting llama.cpp prebuilt...")
        panel.append(f"extract: {tar_path} -> {llama_dir}")
        panel.render()

    if llama_dir.exists():
        shutil.rmtree(llama_dir)
    llama_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(llama_dir)

    candidates = list(llama_dir.rglob("llama-server"))
    if not candidates:
        raise RuntimeError("llama-server not found after extraction.")

    server = candidates[0]
    bin_dir = server.parent

    for p in list(bin_dir.glob("llama-*")) + list(bin_dir.glob("*.so*")):
        try:
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass

    env_path = root_dir / "llama_env.sh"
    save_env(env_path, llama_dir, bin_dir, server)

    info = {
        "tag": rel.get("tag_name"),
        "asset": name,
        "llama_dir": str(llama_dir),
        "bin_dir": str(bin_dir),
        "server": str(server),
        "skipped": False,
    }
    if panel:
        panel.set_summary("llama.cpp prebuilt", data=info)
        panel.append(f"server: {server}")
        panel.render()
    else:
        show_summary("llama.cpp prebuilt", info)
    return info


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


def ensure_cloudflared(target_dir, panel=None):
    existing = shutil.which("cloudflared")
    if existing:
        if panel:
            panel.append(f"cloudflared: {existing}")
            panel.render()
        return existing

    url = CLOUDFLARED_AMD64 if arch_name() == "amd64" else CLOUDFLARED_ARM64
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "cloudflared"

    if panel:
        panel.append(f"download cloudflared: {url}")
        panel.render()
    urllib.request.urlretrieve(url, target)
    target.chmod(0o755)
    return str(target)
