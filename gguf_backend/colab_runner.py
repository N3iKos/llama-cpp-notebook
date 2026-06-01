import os
from pathlib import Path
from .installer import ensure_apt_tools, install_llama_cpp_prebuilt
from .downloader import download_model_pair
from .server import ServerConfig, start_server
from .tunnel import start_tunnels

def run_colab(
    *,
    model_url,
    mmproj_url="",
    ngrok_authtoken="",
    tunnel_mode="both",
    cuda_preference="12.8",
    ctx_size=8192,
    split_mode="none",
    tensor_split="1",
    batch_size=2048,
    ubatch_size=512,
    parallel=1,
    flash_attn=True,
    cache_type_k="f16",
    cache_type_v="f16",
    image_min_tokens=None,
    image_max_tokens=None,
    chat_template_kwargs=None,
    mmproj_offload=True,
    port=8080,
    alias="local-vl",
    hf_token="",
):
    root = Path("/content")
    model_dir = root / "models" / "current"
    ensure_apt_tools()
    install_llama_cpp_prebuilt(root, cuda_preference=cuda_preference)
    cfg = download_model_pair(model_url, mmproj_url, model_dir, hf_token=hf_token or os.environ.get("HF_TOKEN", ""), connections=16)
    server_cfg = ServerConfig(
        root_dir=str(root),
        model_path=cfg["model_path"],
        mmproj_path=cfg.get("mmproj_path", ""),
        port=port,
        alias=alias,
        ctx_size=ctx_size,
        split_mode=split_mode,
        fallback_split_mode="layer" if split_mode != "none" else "",
        tensor_split=tensor_split,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        parallel=parallel,
        flash_attn=flash_attn,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        image_min_tokens=image_min_tokens,
        image_max_tokens=image_max_tokens,
        chat_template_kwargs=chat_template_kwargs,
        mmproj_offload=mmproj_offload,
        cuda_visible_devices="0",
    )
    server_info = start_server(server_cfg, warmup=True)
    urls = start_tunnels(port, str(root), mode=tunnel_mode, ngrok_token=ngrok_authtoken, fallback_cloudflare=True)
    return {"server": server_info, "tunnels": urls}
