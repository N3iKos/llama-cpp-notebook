import time
import json
import base64
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from .ui import live_print

def chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 120,
    temperature: float = 0.7,
    stream: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Sends an OpenAI-compatible chat completion request using standard library utilities."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        **kwargs
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60.0) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8") if e.fp else str(e)
        raise RuntimeError(f"HTTP completion request failed (HTTP {e.code}): {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Completion request failed: {str(e)}")


def encode_image_base64(image_path: str) -> str:
    """Reads a local image and encodes it to a base64 string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_text(base_url: str, model: str, prompt: str, max_tokens: int = 120) -> str:
    """Sends a text-only chat completion, calculating and displaying processing stats."""
    messages = [{"role": "user", "content": prompt}]
    
    live_print([
        f"Sending text prompt to model '{model}'...",
        f"Prompt: '{prompt}'",
        f"Endpoint: {base_url}"
    ], title="Client Text Test", force=True)
    
    start_time = time.time()
    try:
        response = chat(base_url, model, messages, max_tokens=max_tokens)
        duration = time.time() - start_time
        
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"Invalid server response, no choices found: {json.dumps(response)}")
            
        content = choices[0]["message"]["content"]
        
        # Extract token usage if available
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        stats = [
            f"Response received successfully!",
            f"Content: '{content.strip()}'",
            "",
            "=== Operational Stats ===",
            f"Client Latency: {duration:.2f} seconds"
        ]
        
        if completion_tokens > 0:
            tokens_per_sec = completion_tokens / duration
            stats.append(f"Prompt Tokens: {prompt_tokens}")
            stats.append(f"Completion Tokens: {completion_tokens}")
            stats.append(f"Processing Speed: {tokens_per_sec:.1f} tokens/second")
            
        live_print(stats, title="Client Test Success", force=True)
        return content
        
    except Exception as e:
        live_print([
            "CRITICAL: Chat completion test failed.",
            f"Error: {str(e)}"
        ], title="Client Test Failure", force=True)
        raise e


def test_image(base_url: str, model: str, image_path: str, prompt: str, max_tokens: int = 120) -> str:
    """Sends an image-encoded multimodal completion prompt to verify VLM layers."""
    import os
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Sample verification image was not found at: {image_path}")
        
    # Standard format for base64 type identification
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
        
    base64_data = encode_image_base64(image_path)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{ext};base64,{base64_data}"
                }
            }
        ]
    }]
    
    live_print([
        f"Sending multimodal image prompt to model '{model}'...",
        f"Image: {image_path}",
        f"Prompt: '{prompt}'"
    ], title="Client Multimodal Test", force=True)
    
    start_time = time.time()
    try:
        response = chat(base_url, model, messages, max_tokens=max_tokens)
        duration = time.time() - start_time
        
        choices = response.get("choices", [])
        content = choices[0]["message"]["content"]
        
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        
        stats = [
            "VLM Response received successfully!",
            f"Content: '{content.strip()}'",
            "",
            "=== Operational Stats ===",
            f"Client Latency: {duration:.2f} seconds"
        ]
        
        if completion_tokens > 0:
            tokens_per_sec = completion_tokens / duration
            stats.append(f"Completion Tokens: {completion_tokens}")
            stats.append(f"Processing Speed: {tokens_per_sec:.1f} tokens/second")
            
        live_print(stats, title="Client Test Success", force=True)
        return content
        
    except Exception as e:
        live_print([
            "CRITICAL: Multimodal completion test failed.",
            f"Error: {str(e)}"
        ], title="Client Test Failure", force=True)
        raise e
