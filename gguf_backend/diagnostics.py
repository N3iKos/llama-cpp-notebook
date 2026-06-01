import shutil
import urllib.request
import subprocess
from typing import List, Dict, Any
from .config import RuntimeProfile
from .ui import live_print

def check_gpu() -> List[str]:
    """Retrieves list of available GPUs and VRAM using nvidia-smi."""
    gpus = []
    try:
        # Run nvidia-smi query
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        for idx, line in enumerate(res.stdout.strip().split("\n")):
            if line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name, free_mem = parts[0], parts[1]
                    gpus.append(f"GPU{idx}: {name}, {free_mem} MiB free")
    except Exception:
        # GPU query failed, likely CPU fallback
        pass
    return gpus


def check_cuda() -> str:
    """Detects CUDA version using nvidia-smi or nvcc."""
    try:
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in res.stdout.split("\n"):
            if "CUDA Version:" in line:
                parts = line.split("CUDA Version:")
                if len(parts) >= 2:
                    return parts[1].split()[0].strip()
    except Exception:
        pass
        
    try:
        res = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in res.stdout.split("\n"):
            if "release" in line:
                parts = line.split("release")
                if len(parts) >= 2:
                    return parts[1].split(",")[0].strip()
    except Exception:
        pass
        
    return "Not Detected"


def check_disk(work_dir: str) -> str:
    """Validates disk space of root runtime path."""
    try:
        # Make sure work_dir exists
        import os
        os.makedirs(work_dir, exist_ok=True)
        total, used, free = shutil.disk_usage(work_dir)
        free_gb = free / (1024**3)
        if free_gb < 10.0:
            return f"WARNING: Low disk space ({free_gb:.1f} GB free)"
        return f"OK ({free_gb:.1f} GB free)"
    except Exception as e:
        return f"Error checking: {str(e)}"


def check_internet() -> str:
    """Verifies access to GitHub and HuggingFace endpoints."""
    results = []
    
    # Test GitHub
    try:
        req = urllib.request.Request("https://github.com", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5.0) as r:
            if r.status == 200:
                results.append("GitHub OK")
            else:
                results.append(f"GitHub HTTP {r.status}")
    except Exception as e:
        results.append(f"GitHub FAILED ({type(e).__name__})")
        
    # Test Hugging Face
    try:
        req = urllib.request.Request("https://huggingface.co", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5.0) as r:
            if r.status == 200:
                results.append("Hugging Face OK")
            else:
                results.append(f"Hugging Face HTTP {r.status}")
    except Exception as e:
        results.append(f"Hugging Face FAILED ({type(e).__name__})")
        
    return ", ".join(results)


def run(profile: str = "colab_t4x1") -> Dict[str, Any]:
    """Runs diagnostics checks and outputs clean Jupyter logs."""
    prof = RuntimeProfile.from_name(profile)
    
    gpus = check_gpu()
    cuda_ver = check_cuda()
    disk_status = check_disk(prof.work_dir)
    internet_status = check_internet()
    
    # Check essential commands
    commands = ["tar", "curl"]
    missing = [cmd for cmd in commands if shutil.which(cmd) is None]
    cmd_status = "OK" if not missing else f"Missing: {', '.join(missing)}"
    
    lines = []
    if gpus:
        lines.extend(gpus)
    else:
        lines.append("GPU: None (CPU fallback active)")
        
    lines.append(f"CUDA: {cuda_ver}")
    lines.append(f"Disk: {disk_status}")
    lines.append(f"Internet: {internet_status}")
    lines.append(f"System Utilities: {cmd_status}")
    
    live_print(lines, title="Notebook Backend Diagnostics", force=True)
    
    return {
        "gpus": gpus,
        "cuda_version": cuda_ver,
        "disk": disk_status,
        "internet": internet_status,
        "utilities": cmd_status
    }
