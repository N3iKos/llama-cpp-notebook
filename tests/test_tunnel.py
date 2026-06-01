import re
import pytest
from unittest.mock import patch, MagicMock
from gguf_backend import tunnel

def test_cloudflare_regex_parsing():
    sample_log = (
        "2026-06-02T00:55:00Z INF Thank you for trying Cloudflare Tunnel. "
        "Your quick Tunnel has been created! "
        "URL: https://gentle-winds-occur.trycloudflare.com "
        "2026-06-02T00:55:01Z INF Connection established."
    )
    
    regex = r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
    matches = re.findall(regex, sample_log)
    
    assert len(matches) == 1
    assert matches[0] == "https://gentle-winds-occur.trycloudflare.com"


@patch('gguf_backend.tunnel.get_ngrok_token')
@patch('gguf_backend.tunnel.start_cloudflare')
def test_tunnel_selection_cloudflare(mock_cf, mock_ngrok_token):
    # Mock token to empty
    mock_ngrok_token.return_value = ""
    mock_cf.return_value = "https://mock.trycloudflare.com"
    
    url = tunnel.open_tunnel(port=8080, ngrok_token=None, prefer="ngrok", fallback="cloudflare")
    
    # Assert cloudflare was chosen
    assert url == "https://mock.trycloudflare.com"
    mock_cf.assert_called_once()


@patch('gguf_backend.tunnel.start_ngrok')
def test_tunnel_selection_ngrok(mock_ngrok):
    mock_ngrok.return_value = "https://mock.ngrok-free.app"
    
    url = tunnel.open_tunnel(port=8080, ngrok_token="valid_token_xyz", prefer="ngrok")
    
    # Assert ngrok was chosen
    assert url == "https://mock.ngrok-free.app"
    mock_ngrok.assert_called_once_with(8080, "valid_token_xyz")
