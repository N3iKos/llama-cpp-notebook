import re
import time
import subprocess
from .ui import live_print

CLOUDFLARE_URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")

def run(cmd, *, timeout=None, env=None, check=False, cwd=None, quiet=False):
    p = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
        cwd=cwd,
    )
    if not quiet:
        print("$ " + (cmd if isinstance(cmd, str) else " ".join(map(str, cmd))))
        print(p.stdout.strip() or "(no output)")
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}: {cmd}")
    return p

def run_live(cmd, *, label="process", env=None, cwd=None, timeout=None, clear=True, keep_last=25):
    start = time.time()
    cmd_display = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
    proc = subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
        bufsize=1,
    )
    lines = []
    last_panel = 0.0
    try:
        while True:
            if timeout and time.time() - start > timeout:
                proc.kill()
                raise TimeoutError(f"{label} timeout after {timeout}s")
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                line = line.rstrip("\n")
                if line:
                    lines.append(line)
                    lines = lines[-keep_last:]
                now = time.time()
                if now - last_panel >= 0.5:
                    live_print("\n".join([label, f"$ {cmd_display}", f"elapsed: {int(now-start)}s", "", *lines]), clear=clear)
                    last_panel = now
            if proc.poll() is not None:
                rest = proc.stdout.read() if proc.stdout else ""
                if rest:
                    lines.extend([x for x in rest.splitlines() if x.strip()])
                    lines = lines[-keep_last:]
                break
            if not line:
                time.sleep(0.1)
        live_print("\n".join([label, f"$ {cmd_display}", f"elapsed: {int(time.time()-start)}s", f"exit: {proc.returncode}", "", *lines]), clear=clear)
        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed with code {proc.returncode}")
        return "\n".join(lines)
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

def parse_trycloudflare_url(text):
    match = CLOUDFLARE_URL_RE.search(text or "")
    return match.group(0) if match else None
