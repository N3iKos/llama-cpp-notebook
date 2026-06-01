import os
import sys
import json
import tempfile
import pytest
from gguf_backend import ServerConfig, RuntimeProfile, config, ui, shell, diagnostics

def test_config_profiles():
    # Test Colab Profile defaults
    colab = RuntimeProfile.from_name("colab_t4x1")
    assert colab.profile_name == "colab_t4x1"
    assert colab.split_mode == "none"
    assert colab.ctx_size == 4096
    
    # Test Kaggle Profile defaults
    kaggle = RuntimeProfile.from_name("kaggle_t4x2")
    assert kaggle.profile_name == "kaggle_t4x2"
    assert kaggle.split_mode == "row"
    assert kaggle.ctx_size == 8192


def test_server_config():
    # Test colab config
    colab_cfg = ServerConfig(profile="colab_t4x1")
    assert colab_cfg.profile == "colab_t4x1"
    assert colab_cfg.split_mode == "none"
    assert colab_cfg.ctx_size == 4096

    # Test kaggle config
    kaggle_cfg = ServerConfig(profile="kaggle_t4x2")
    assert kaggle_cfg.profile == "kaggle_t4x2"
    assert kaggle_cfg.split_mode == "row"
    assert kaggle_cfg.ctx_size == 8192


def test_state_saving_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prof = RuntimeProfile.from_name("colab_t4x1")
        prof.work_dir = tmp_dir # Override work dir to temp dir
        
        state = {"model_path": "/some/path/model.gguf", "status": "downloaded"}
        config.save_state(prof, state)
        
        loaded = config.load_state(prof)
        assert loaded["model_path"] == "/some/path/model.gguf"
        assert loaded["status"] == "downloaded"


def test_shell_runners():
    # Test basic command runner
    res = shell.run([sys.executable, "-c", "print('hello')"])
    assert res.returncode == 0
    assert "hello" in res.stdout.strip().lower()

    # Test failing command runner raising ShellError
    with pytest.raises(shell.ShellError):
        shell.run(["non_existent_command_name_xyz"], check=True)


def test_diagnostics_structure():
    # Basic structural check
    res = diagnostics.run("colab_t4x1")
    assert "disk" in res
    assert "internet" in res
    assert "cuda_version" in res
