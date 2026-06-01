import os
import pytest
from gguf_backend import ServerConfig, server

def test_command_builder_basic():
    # Construct base configurations
    cfg = ServerConfig(
        profile="colab_t4x1",
        alias="test-llm",
        ctx_size=2048,
        batch_size=512,
        ubatch_size=128,
        flash_attn=True,
        cache_type_k="q8_0",
        cache_type_v="q8_0"
    )
    
    cmd = server.build_server_command(
        server_path="/path/to/llama-server",
        config=cfg,
        model_path="/models/model.gguf",
        mmproj_path=None
    )
    
    # Assert structural parameters are present
    assert cmd[0] == "/path/to/llama-server"
    assert "-m" in cmd
    assert "/models/model.gguf" in cmd
    assert "--alias" in cmd
    assert "test-llm" in cmd
    assert "-c" in cmd
    assert "2048" in cmd
    assert "-b" in cmd
    assert "512" in cmd
    assert "--ubatch-size" in cmd
    assert "128" in cmd
    
    # Assert flash attention is injected
    assert "--flash-attn" in cmd
    
    # Assert KV cache types are set
    assert "--cache-type-k" in cmd
    assert "q8_0" in cmd
    assert "--cache-type-v" in cmd
    assert "q8_0" in cmd
    
    # Assert multimodal projector is NOT injected since None
    assert "--mmproj" not in cmd


def test_command_builder_multimodal(tmp_path):
    cfg = ServerConfig(profile="colab_t4x1")
    
    # Touch temporary files to satisfy the file existence check
    model_file = tmp_path / "model.gguf"
    model_file.touch()
    mmproj_file = tmp_path / "mmproj.gguf"
    mmproj_file.touch()
    
    # We pass valid paths
    cmd = server.build_server_command(
        server_path="llama-server",
        config=cfg,
        model_path=str(model_file),
        mmproj_path=str(mmproj_file)
    )
    
    # Verify mmproj is set correctly when file is provided (mock paths)
    assert "--mmproj" in cmd
    assert str(mmproj_file) in cmd


def test_command_builder_kaggle_gpu_splitting(monkeypatch):
    # Mock CUDA environment to dual-GPU
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    
    cfg = ServerConfig(
        profile="kaggle_t4x2",
        split_mode="row",
        tensor_split="1,1"
    )
    
    cmd = server.build_server_command(
        server_path="llama-server",
        config=cfg,
        model_path="model.gguf"
    )
    
    # Verify row split mode and tensor splits are passed correctly
    assert "--split-mode" in cmd
    assert "row" in cmd
    assert "--tensor-split" in cmd
    assert "1,1" in cmd


def test_command_builder_colab_no_tensor_splitting(monkeypatch):
    # Mock CUDA environment to single-GPU
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    
    cfg = ServerConfig(
        profile="colab_t4x1",
        split_mode="none"
    )
    
    cmd = server.build_server_command(
        server_path="llama-server",
        config=cfg,
        model_path="model.gguf"
    )
    
    # Verify single GPU profile sets split-mode to none
    assert "--split-mode" in cmd
    assert "none" in cmd
    assert "--tensor-split" not in cmd
