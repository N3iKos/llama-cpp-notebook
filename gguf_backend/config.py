import os
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any

# Profile defaults matching the PRD specifications
COLAB_T4X1 = {
    "profile_name": "colab_t4x1",
    "root": "/content",
    "work_dir": "/content/gguf_backend_work",
    "cuda_visible_devices": "0",
    "split_mode": "none",
    "fallback_split_mode": None,
    "tensor_split": None,
    "ctx_size": 4096,
    "batch_size": 1024,
    "ubatch_size": 256,
}

KAGGLE_T4X2 = {
    "profile_name": "kaggle_t4x2",
    "root": "/kaggle/working",
    "work_dir": "/kaggle/working/gguf_backend_work",
    "cuda_visible_devices": "0,1",
    "split_mode": "row",
    "fallback_split_mode": "layer",
    "tensor_split": "1,1",
    "ctx_size": 8192,
    "batch_size": 2048,
    "ubatch_size": 512,
}

@dataclass
class RuntimeProfile:
    profile_name: str
    root: str
    work_dir: str
    cuda_visible_devices: str
    split_mode: str
    fallback_split_mode: Optional[str]
    tensor_split: Optional[str]
    ctx_size: int
    batch_size: int
    ubatch_size: int

    @property
    def bin_dir(self) -> str:
        return os.path.join(self.work_dir, "bin")

    @property
    def model_dir(self) -> str:
        return os.path.join(self.work_dir, "models")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.work_dir, "logs")

    @property
    def state_file(self) -> str:
        return os.path.join(self.work_dir, "runtime_state.json")

    @classmethod
    def from_name(cls, name: str) -> "RuntimeProfile":
        if "kaggle" in name.lower():
            return cls(**KAGGLE_T4X2)
        else:
            return cls(**COLAB_T4X1)


@dataclass
class ServerConfig:
    profile: str = "colab_t4x1"
    alias: str = "local-vl"
    ctx_size: int = 4096
    split_mode: str = "none"
    fallback_split_mode: Optional[str] = None
    tensor_split: Optional[str] = None
    batch_size: int = 1024
    ubatch_size: int = 256
    parallel: int = 1
    flash_attn: bool = True
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    image_min_tokens: Optional[int] = None
    image_max_tokens: Optional[int] = None
    chat_template_kwargs: Optional[str] = None
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Override default server config values based on the selected profile if not overridden
        p = RuntimeProfile.from_name(self.profile)
        # Default override logic if standard defaults are provided
        if self.profile == "kaggle_t4x2" and self.ctx_size == 4096:
            self.ctx_size = p.ctx_size
            self.split_mode = p.split_mode
            self.fallback_split_mode = p.fallback_split_mode
            self.tensor_split = p.tensor_split
            self.batch_size = p.batch_size
            self.ubatch_size = p.ubatch_size


@dataclass
class DownloadConfig:
    model_url: str
    mmproj_url: Optional[str] = None
    connections: int = 16
    hf_token: Optional[str] = None


@dataclass
class TunnelConfig:
    port: int = 8080
    ngrok_token: Optional[str] = None
    prefer: str = "ngrok"
    fallback: str = "cloudflare"


def save_state(profile: RuntimeProfile, state_data: Dict[str, Any]) -> None:
    """Saves runtime state data to state file."""
    os.makedirs(profile.work_dir, exist_ok=True)
    state_path = profile.state_file
    
    # Read existing state if it exists
    existing = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                existing = json.load(f)
        except Exception:
            pass
            
    existing.update(state_data)
    with open(state_path, "w") as f:
        json.dump(existing, f, indent=2)


def load_state(profile: RuntimeProfile) -> Dict[str, Any]:
    """Loads runtime state data from state file."""
    state_path = profile.state_file
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
