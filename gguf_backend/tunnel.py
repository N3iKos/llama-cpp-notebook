import os
import subprocess
import time
from pathlib import Path

from .installer import ensure_cloudflared
from .shell import parse_trycloudflare_url
from .panel import NotebookPanel, run_command, show_summary, sanitize_log_line


def start_ngrok(port, token, *, panel=None):
    if not token:
        raise RuntimeError("ngrok token is empty.")

    run_command(
        ["python3", "-m", "pip", "install", "-q", "pyngrok"],
        check=False,
        label="install pyngrok",
        panel=panel,
        finalize=False if panel else True,
        tail_lines=100,
    )

    if panel:
        panel.section("ngrok tunnel")
        panel.set_status("starting ngrok tunnel...")

    from pyngrok import ngrok
    ngrok.set_auth_token(token)
    ngrok.kill()
    tunnel = ngrok.connect(port, "http", bind_tls=True)
    url = tunnel.public_url.rstrip("/")
    if panel:
        panel.append(f"ngrok: {url}/v1/chat/completions")
        panel.render()
    return url


def start_cloudflare(port, root_dir, *, panel=None):
    root_dir = Path(root_dir)
    cloudflared = ensure_cloudflared(target_dir=root_dir, panel=panel)

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

    own_panel = panel is None
    if panel is None:
        panel = NotebookPanel("cloudflare tunnel", show_progress=False, height=300)
    panel.display_panel()
    panel.set_progress_visible(False)
    panel.set_footer(f"log: {log_path}")
    panel.section("cloudflare tunnel")
    panel.set_status("starting cloudflare tunnel...")
    panel.append("$ " + " ".join(map(str, cmd)))
    panel.render()

    start = time.time()
    url = None

    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        while time.time() - start < 75:
            if proc.poll() is not None:
                if own_panel:
                    panel.finish(False, "cloudflared exited before URL was generated")
                else:
                    panel.set_status("cloudflared exited before URL was generated")
                    panel.render()
                raise RuntimeError("cloudflared exited before URL was generated.")

            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                logf.write(line)
                logf.flush()
                clean = sanitize_log_line(line, hide_redirects=False, max_len=300)
                if clean:
                    panel.append(clean, max_lines=160)
                    found = parse_trycloudflare_url(clean)
                    if found:
                        url = found.rstrip("/")
                        break

            panel.set_status(f"starting cloudflare | elapsed {int(time.time() - start)}s")
            panel.render()
            time.sleep(0.2)

    if not url:
        if own_panel:
            panel.finish(False, "cloudflare tunnel URL not found")
        else:
            panel.set_status("cloudflare tunnel URL not found")
            panel.render()
        raise RuntimeError("cloudflare tunnel URL not found in output.")

    panel.append("")
    panel.append(f"cloudflare: {url}/v1/chat/completions")
    panel.set_status("cloudflare tunnel ready")
    panel.render()
    if own_panel:
        panel.set_links("tunnel result", {"cloudflare chat": url + "/v1/chat/completions"})
        panel.finish(True, "cloudflare tunnel ready")
    return url


def start_tunnels(port, root_dir, *, mode="both", ngrok_token="", fallback_cloudflare=True, panel=None, finalize=True):
    urls = {}

    own_panel = panel is None
    if panel is None:
        panel = NotebookPanel("tunnels", show_progress=False, height=320)
    panel.display_panel()
    panel.set_progress_visible(False)

    if mode not in ("both", "ngrok", "cloudflare"):
        raise ValueError("mode must be: both, ngrok, or cloudflare")

    if mode in ("both", "ngrok"):
        try:
            if ngrok_token:
                urls["ngrok"] = start_ngrok(port, ngrok_token, panel=panel)
            elif mode == "ngrok" and not fallback_cloudflare:
                raise RuntimeError("ngrok token required.")
            else:
                panel.append("ngrok: skipped, token not provided")
                panel.render()
        except Exception as e:
            urls["ngrok_error"] = str(e)
            panel.append(f"ngrok error: {e}")
            panel.render()
            if mode == "ngrok" and not fallback_cloudflare:
                if finalize or own_panel:
                    panel.finish(False, "ngrok failed")
                raise

    need_cloudflare = mode in ("both", "cloudflare") or (mode == "ngrok" and fallback_cloudflare and "ngrok" not in urls)
    if need_cloudflare:
        try:
            urls["cloudflare"] = start_cloudflare(port, root_dir, panel=panel)
        except Exception as e:
            urls["cloudflare_error"] = str(e)
            panel.append(f"cloudflare error: {e}")
            panel.render()

    links = {}
    for name, url in urls.items():
        if not name.endswith("_error"):
            links[f"{name} chat"] = url.rstrip("/") + "/v1/chat/completions"
            links[f"{name} base"] = url.rstrip("/")

    if links:
        panel.set_links("tunnel result", links, note="Use the chat endpoint for OpenAI-compatible /v1/chat/completions clients.")
    else:
        panel.set_summary("tunnel result", data=urls)

    if finalize or own_panel:
        ok = bool(links)
        panel.finish(ok, "tunnel ready" if ok else "tunnel failed")

    return urls
