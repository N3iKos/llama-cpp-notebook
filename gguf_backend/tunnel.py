import os
import subprocess
import time
from pathlib import Path

from .installer import ensure_cloudflared
from .shell import parse_trycloudflare_url, run
from .panel import NotebookPanel, show_summary


def start_ngrok(port, token):
    if not token:
        raise RuntimeError("ngrok token is empty.")

    run(["python3", "-m", "pip", "install", "-q", "pyngrok"], check=False, label="install pyngrok")

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

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(root_dir), bufsize=1)
    pid_path.write_text(str(proc.pid))

    panel = NotebookPanel("cloudflare tunnel", show_progress=False, height=260)
    panel.display_panel()
    panel.set_footer(f"log: {log_path}")
    panel.set_status("starting tunnel...")
    panel.append("$ " + " ".join(map(str, cmd)))

    start = time.time()
    url = None

    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        while time.time() - start < 60:
            if proc.poll() is not None:
                panel.finish(False, "cloudflared exited before URL was generated")
                raise RuntimeError("cloudflared exited before URL was generated.")

            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                logf.write(line)
                logf.flush()
                clean = line.strip()
                if clean:
                    panel.append(clean, max_lines=120)
                    found = parse_trycloudflare_url(clean)
                    if found:
                        url = found.rstrip("/")
                        break

            panel.set_status(f"starting | elapsed {int(time.time() - start)}s")
            panel.render()
            time.sleep(0.2)

    if not url:
        panel.finish(False, "cloudflare tunnel URL not found")
        raise RuntimeError("cloudflare tunnel URL not found in output.")

    panel.append("")
    panel.append(f"public: {url}/v1/chat/completions")
    panel.finish(True, "cloudflare tunnel ready")
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

    lines = []
    for name, url in urls.items():
        if name.endswith("_error"):
            lines.append(f"{name}: {url}")
        else:
            lines.append(f"{name}: {url}/v1/chat/completions")
    show_summary("tunnel endpoints", lines=lines)

    return urls
