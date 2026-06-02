import re
import subprocess

from .terminal_panel import TerminalPanelConfig, run_terminal_panel

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


def run_live(
    cmd,
    *,
    label="process",
    env=None,
    cwd=None,
    timeout=None,
    clear=True,
    keep_last=25,
    log_path=None,
):
    del clear  # Kept for backward-compatible callers; panel updates in place.
    result = run_terminal_panel(
        cmd,
        TerminalPanelConfig(
            label=label,
            env=env,
            cwd=cwd,
            log_path=log_path,
            tail_lines=keep_last,
            failure_tail_lines=max(20, min(40, keep_last if keep_last > 25 else 30)),
        ),
        timeout=timeout,
        check=True,
    )
    return "\n".join(result.tail_lines)


def parse_trycloudflare_url(text):
    match = CLOUDFLARE_URL_RE.search(text or "")
    return match.group(0) if match else None
