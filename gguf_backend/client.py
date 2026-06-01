import json
import urllib.request


def post_json(url, payload, timeout=180):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", errors="replace")
        return r.status, json.loads(body)


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "true"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", errors="replace")
        try:
            return r.status, json.loads(body)
        except Exception:
            return r.status, body


def chat(base_url, model, message, *, max_tokens=128, temperature=0.2, extra=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    return post_json(base_url.rstrip("/") + "/v1/chat/completions", payload)
