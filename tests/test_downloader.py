import os
import re
import pytest
from unittest.mock import patch, MagicMock
from gguf_backend import installer, downloader, config

def test_fallback_llama_release_url():
    # Make sure default fallback works and contains the tag and filename format
    url = installer.get_latest_llama_release_url()
    assert "github.com/ai-dock/llama.cpp-cuda" in url
    assert "tar.gz" in url


def test_aria2_regex_parsing():
    # Progress line sample matching aria2 standard stdout format
    sample_line = "[#4dc417 584MiB/6.6GiB(8%) CN:16 DL:30MiB ETA:3m26s]"
    
    # Run parsing check via regex directly
    match = re.search(
        r'\[#([0-9a-fA-F]+)\s+([0-9.]+[a-zA-Z]+)/([0-9.]+[a-zA-Z]+)\(([0-9]+)%\)\s+CN:(\d+)\s+DL:([0-9.]+[a-zA-Z]+)(?:\s+ETA:([0-9a-zA-Z]+))?\]',
        sample_line
    )
    assert match is not None
    gid, downloaded, total, percent, cn, speed, eta = match.groups()
    assert gid == "4dc417"
    assert downloaded == "584MiB"
    assert total == "6.6GiB"
    assert percent == "8"
    assert cn == "16"
    assert speed == "30MiB"
    assert eta == "3m26s"


def test_aria2_regex_parsing_no_eta():
    # Progress line sample without ETA
    sample_line_no_eta = "[#a8c3d2 12.5MiB/4.2GiB(0%) CN:8 DL:1.2MiB]"
    match = re.search(
        r'\[#([0-9a-fA-F]+)\s+([0-9.]+[a-zA-Z]+)/([0-9.]+[a-zA-Z]+)\(([0-9]+)%\)\s+CN:(\d+)\s+DL:([0-9.]+[a-zA-Z]+)(?:\s+ETA:([0-9a-zA-Z]+))?\]',
        sample_line_no_eta
    )
    assert match is not None
    gid, downloaded, total, percent, cn, speed, eta = match.groups()
    assert gid == "a8c3d2"
    assert downloaded == "12.5MiB"
    assert total == "4.2GiB"
    assert percent == "0"
    assert cn == "8"
    assert speed == "1.2MiB"
    assert eta is None


@patch('urllib.request.urlretrieve')
@patch('shutil.which')
@patch('gguf_backend.shell.stream')
def test_download_model_saves_config(mock_stream, mock_which, mock_urlretrieve, tmp_path):
    mock_which.return_value = None # Force fallback mode for test stability
    mock_stream.return_value = 0
    
    # Configure workspace directories to temporary path
    prof = config.RuntimeProfile.from_name("colab_t4x1")
    prof.work_dir = str(tmp_path / "work")
    model_dir = str(tmp_path / "models")
    
    model_url = "https://huggingface.co/gutris1/webui/resolve/main/misc/card-no-preview.png"
    
    # Trigger download
    res = downloader.download_model(
        model_url=model_url,
        mmproj_url=None,
        output_dir=model_dir,
        profile=prof,
    )
    
    # Confirm config saves correctly
    assert res["model_url"] == model_url
    assert "model_config.json" in os.listdir(model_dir)
    
    # Confirm saved state updates correctly
    state = config.load_state(prof)
    assert "model_path" in state
