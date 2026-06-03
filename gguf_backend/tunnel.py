import os
import subprocess
import time
from pathlib import Path

from .installer import ensure_cloudflared
from .shell import parse_trycloudflare_url
from .ui import live_print


def start_ngrok(port, token):
    if not token:
        raise RuntimeError("ngrok token is empty.")

    subprocess.run(["python3", "-m", "pip", "install", "-q", "pyngrok"], check=False)

    from pyngrok import ngrok
    ngrok.set_auth_token(token)
    ngrok.kill()
    tunnel = ngrok.connect(port, "http", bind_tls=True)
    return tunnel.public_url.rstrip("/")


def start_cloudflare(port, root_dir):
    root_dir = Path(root_dir)
    cloudflared = ensure_cloudflared(target_dir=root_dir)

    log_path = root_dir / "cloudflared.log"
    pid_path = root_dir / "cloudflared.pid"

    if pid_path.exists():
        try:
            os.kill(int(pid_path.read_text().strip()), 15)
            time.sleep(1)
        except Exception:
            pass

    cmd = [cloudflared, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]

    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(root_dir), bufsize=1)
    pid_path.write_text(str(proc.pid))

    start = time.time()
    lines = []
    url = None

    while time.time() - start < 60:
        if proc.poll() is not None:
            raise RuntimeError("cloudflared exited before URL was generated.")

        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            logf.write(line)
            logf.flush()
            lines.append(line.strip())
            lines = lines[-15:]

            found = parse_trycloudflare_url(line)
            if found:
                url = found.rstrip("/")
                break

        live_print(
            "\n".join(["cloudflare tunnel starting", f"elapsed: {int(time.time() - start)}s", *lines[-8:]]),
            clear=True,
        )
        time.sleep(0.2)

    if not url:
        raise RuntimeError("cloudflare tunnel URL not found in output.")

    return url


def start_tunnels(port, root_dir, *, mode="both", ngrok_token="", fallback_cloudflare=True):
    urls = {}

    if mode not in ("both", "ngrok", "cloudflare"):
        raise ValueError("mode must be: both, ngrok, or cloudflare")

    if mode in ("both", "ngrok"):
        try:
            if ngrok_token:
                urls["ngrok"] = start_ngrok(port, ngrok_token)
            elif mode == "ngrok" and not fallback_cloudflare:
                raise RuntimeError("ngrok token required.")
        except Exception as e:
            urls["ngrok_error"] = str(e)
            if mode == "ngrok" and not fallback_cloudflare:
                raise

    if mode in ("both", "cloudflare") or (mode == "ngrok" and fallback_cloudflare and "ngrok" not in urls):
        try:
            urls["cloudflare"] = start_cloudflare(port, root_dir)
        except Exception as e:
            urls["cloudflare_error"] = str(e)

    lines = ["tunnel endpoints"]
    for name, url in urls.items():
        if name.endswith("_error"):
            lines.append(f"{name}: {url}")
        else:
            lines.append(f"{name}: {url}/v1/chat/completions")
    live_print("\n".join(lines), clear=True)

    return urls
