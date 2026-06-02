import re

from .panel import run_command

CLOUDFLARE_URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")


def run(cmd, *, timeout=None, env=None, check=False, cwd=None, quiet=False, label=None, panel=None, finalize=None):
    if finalize is None:
        finalize = panel is None
    return run_command(
        cmd,
        label=label or "command",
        env=env,
        cwd=cwd,
        timeout=timeout,
        check=check,
        mode="terminal",
        show=not quiet,
        panel=panel,
        finalize=finalize,
    )


def run_live(cmd, *, label="process", env=None, cwd=None, timeout=None, clear=True, keep_last=25, panel=None, finalize=None):
    if finalize is None:
        finalize = panel is None
    result = run_command(
        cmd,
        label=label,
        env=env,
        cwd=cwd,
        timeout=timeout,
        check=True,
        mode="download" if _looks_like_downloader(cmd) else "terminal",
        tail_lines=max(keep_last, 100),
        show=True,
        panel=panel,
        finalize=finalize,
    )
    return result.stdout


def _looks_like_downloader(cmd):
    text = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
    return "aria2c" in text or "wget " in text or "curl " in text


def parse_trycloudflare_url(text):
    match = CLOUDFLARE_URL_RE.search(text or "")
    return match.group(0) if match else None
