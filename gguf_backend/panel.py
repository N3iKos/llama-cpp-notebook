"""Notebook-friendly realtime output panels for Kaggle/Colab.

This module keeps raw stdout/stderr in log files while rendering compact,
auto-scrolling ipywidgets panels in notebooks. It falls back to plain text when
ipywidgets/IPython are unavailable.
"""

from __future__ import annotations

import html
import os
import re
import select
import subprocess
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PERCENT_PATTERNS = [re.compile(r"\((\d+(?:\.\d+)?)%\)"), re.compile(r"\b(\d+(?:\.\d+)?)%\b")]
DL_RE = re.compile(r"\bDL:([^\s\]]+)")
ETA_RE = re.compile(r"\bETA:([^\s\]]+)")
SIZE_RE = re.compile(r"([0-9.]+(?:KiB|MiB|GiB|TiB|KB|MB|GB|TB|B))/([0-9.]+(?:KiB|MiB|GiB|TiB|KB|MB|GB|TB|B))")
CN_RE = re.compile(r"\bCN:([^\s\]]+)")
SEED_RE = re.compile(r"\bSEED:([^\s\]]+)")


def default_log_dir() -> Path:
    if Path("/kaggle/working").exists():
        root = Path("/kaggle/working")
    elif Path("/content").exists():
        root = Path("/content")
    else:
        root = Path.cwd()
    path = root / "_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def command_text(cmd) -> str:
    if isinstance(cmd, str):
        return cmd
    return " ".join(str(x) for x in cmd)


def sanitize_log_line(line: str, *, max_len=280, hide_redirects=True) -> Optional[str]:
    line = strip_ansi(line).strip()
    if not line:
        return None
    low = line.lower()
    if hide_redirects and (
        "redirecting to" in low
        or "release-assets.githubusercontent.com" in low
        or "x-amz-" in low
        or "sig=" in low
        or "jwt=" in low
        or "response-content-disposition" in low
    ):
        return None
    if len(line) > max_len:
        line = line[:max_len] + " ..."
    return line


def tail_text(path, n=80) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return ""


class NotebookPanel:
    """A compact, auto-scrolling notebook output panel."""

    def __init__(self, title="Notebook CLI", *, show_progress=False, height=330):
        self.title_text = title
        self.show_progress = show_progress
        self.height = height
        self.lines: list[str] = []
        self.term_id = "cli_term_" + uuid.uuid4().hex
        self._displayed = False
        self._widgets_ok = False
        self._init_widgets()

    def _init_widgets(self):
        try:
            import ipywidgets as widgets
            from IPython.display import Javascript, display

            self.widgets = widgets
            self.display = display
            self.Javascript = Javascript
            self.title = widgets.HTML(value=self._title_html(self.title_text))
            self.status = widgets.HTML(value=self._status_html("idle"))
            self.progress = widgets.FloatProgress(value=0, min=0, max=100, description="0%", bar_style="", layout=widgets.Layout(width="100%"))
            self.progress_info = widgets.HTML(value=self._progress_info_html("-"))
            self.progress_box = widgets.VBox([self.progress, self.progress_info], layout=widgets.Layout(border="1px solid #333", padding="8px 10px", display="block" if self.show_progress else "none"))
            self.terminal = widgets.HTML(value=self._terminal_html("terminal ready"))
            self.footer = widgets.HTML(value=self._footer_html("log: -"))
            self.ui = widgets.VBox([self.title, self.status, self.progress_box, self.terminal, self.footer])
            self._widgets_ok = True
        except Exception:
            self._widgets_ok = False

    def display_panel(self):
        if self._displayed:
            return
        self._displayed = True
        if not self._widgets_ok:
            print(self.title_text)
            return
        self.display(self.ui)
        self.display(self.Javascript(f"""
(function() {{
  const termId = "{self.term_id}";
  const key = "__autoscr_" + termId;
  if (window[key]) {{ clearInterval(window[key]); }}
  window[key] = setInterval(function() {{
    const el = document.getElementById(termId);
    if (el) {{ el.scrollTop = el.scrollHeight; }}
  }}, 100);
}})();
"""))

    def _title_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:15px;font-weight:700;padding:8px 10px;border:1px solid #333;border-bottom:0;background:#111;color:#f5f5f5;">{esc(text)}</div>'

    def _status_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:8px 10px;border-left:1px solid #333;border-right:1px solid #333;background:#181818;color:#ddd;">{esc(text)}</div>'

    def _progress_info_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#bbb;padding-top:4px;">{esc(text)}</div>'

    def _terminal_html(self, body):
        return f'<div id="{self.term_id}" style="margin:0;padding:10px;height:{self.height}px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;background:#050505;color:#e8e8e8;border:1px solid #333;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.35;">{esc(body)}</div>'

    def _footer_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;padding:7px 10px;border:1px solid #333;border-top:0;background:#111;color:#aaa;">{esc(text)}</div>'

    def set_status(self, text):
        if self._widgets_ok:
            self.status.value = self._status_html(text)
        else:
            print(text)

    def set_footer(self, text):
        if self._widgets_ok:
            self.footer.value = self._footer_html(text)

    def set_progress_visible(self, visible: bool):
        self.show_progress = visible
        if self._widgets_ok:
            self.progress_box.layout.display = "block" if visible else "none"

    def set_progress(self, pct: float, info="", style="info"):
        pct = max(0.0, min(100.0, float(pct)))
        if self._widgets_ok:
            self.progress.value = pct
            self.progress.description = f"{pct:.1f}%"
            self.progress.bar_style = style
            self.progress_info.value = self._progress_info_html(info or f"{pct:.1f}%")
        else:
            print(info or f"{pct:.1f}%")

    def append(self, line: str, *, max_lines=500):
        if line:
            self.lines.append(line)
            self.lines = self.lines[-max_lines:]

    def render(self, current=""):
        body = "\n".join(self.lines[-500:])
        if current and current.strip():
            body = body + ("\n" if body else "") + current.strip()
        if self._widgets_ok:
            self.terminal.value = self._terminal_html(body)
        else:
            # Fallback intentionally sparse to avoid flooding non-notebook logs.
            pass

    def finish(self, ok=True, message="completed"):
        self.set_status(message)
        if self.show_progress and ok:
            self.set_progress(100, "100% | completed", "success")
        elif self.show_progress and not ok and self._widgets_ok:
            self.progress.bar_style = "danger"
        self.render()



class SilentPanel:
    def __init__(self):
        self.lines = []
        self.show_progress = False
    def display_panel(self): pass
    def set_footer(self, text): pass
    def set_status(self, text): pass
    def set_progress_visible(self, visible): self.show_progress = visible
    def set_progress(self, pct, info="", style="info"): pass
    def append(self, line, *, max_lines=500):
        if line:
            self.lines.append(line)
            self.lines = self.lines[-max_lines:]
    def render(self, current=""): pass
    def finish(self, ok=True, message="completed"): pass

def parse_aria2_progress(line: str):
    pct = None
    for pat in PERCENT_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                pct = float(m.group(1))
                break
            except Exception:
                pass
    data = {"pct": pct, "dl": None, "eta": None, "size": None, "conn": None, "seed": None}
    for key, regex in (("dl", DL_RE), ("eta", ETA_RE), ("conn", CN_RE), ("seed", SEED_RE)):
        m = regex.search(line)
        if m:
            data[key] = m.group(1)
    m = SIZE_RE.search(line)
    if m:
        data["size"] = f"{m.group(1)} / {m.group(2)}"
    return data


def update_download_progress(panel: NotebookPanel, line: str) -> bool:
    data = parse_aria2_progress(line)
    if data["pct"] is None:
        return False
    pct = data["pct"]
    parts = [f"{pct:.1f}%"]
    if data["size"]:
        parts.append(data["size"])
    if data["dl"]:
        parts.append(f"DL {data['dl']}")
    if data["eta"]:
        parts.append(f"ETA {data['eta']}")
    if data["conn"]:
        parts.append(f"CN {data['conn']}")
    if data["seed"]:
        parts.append(f"SEED {data['seed']}")
    info = " | ".join(parts)
    panel.set_status("download | " + info)
    panel.set_progress(pct, info, "success" if pct >= 100 else "info")
    return True


def run_command(
    cmd,
    *,
    label="process",
    env=None,
    cwd=None,
    timeout=None,
    check=False,
    mode="terminal",
    log_name=None,
    log_dir=None,
    tail_lines=120,
    refresh_interval=0.12,
    hide_redirects=True,
    show=True,
):
    """Run a command via Popen and render a realtime notebook panel.

    Returns subprocess.CompletedProcess(args, returncode, stdout, stderr=None).
    """
    log_dir = Path(log_dir) if log_dir else default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip().lower()) or "command"
    log_path = Path(log_dir) / (log_name or f"{int(time.time())}_{safe_label}.log")

    panel = NotebookPanel(label, show_progress=(mode == "download")) if show else SilentPanel()
    if show:
        panel.display_panel()
    panel.set_footer(f"log: {log_path}")
    panel.set_status("starting download..." if mode == "download" else "running command...")
    if mode == "download":
        panel.set_progress_visible(True)
        panel.set_progress(0, "0%", "")

    shown_cmd = command_text(cmd)
    panel.append("$ " + shown_cmd)
    panel.render()

    popen_cmd = cmd
    shell = isinstance(cmd, str)
    env2 = os.environ.copy()
    if env:
        env2.update(env)

    start = time.time()
    stdout_chunks = []
    current = ""
    last_refresh = 0.0

    proc = subprocess.Popen(
        popen_cmd,
        shell=shell,
        cwd=cwd,
        env=env2,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        bufsize=0,
    )

    fd = proc.stdout.fileno() if proc.stdout else None
    if fd is not None:
        os.set_blocking(fd, False)

    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            while True:
                if timeout and time.time() - start > timeout:
                    proc.kill()
                    panel.finish(False, f"timeout after {timeout}s")
                    raise TimeoutError(f"{label} timeout after {timeout}s")

                if fd is not None:
                    ready, _, _ = select.select([fd], [], [], 0.1)
                else:
                    ready = []

                if ready:
                    try:
                        raw = os.read(fd, 8192)
                    except BlockingIOError:
                        raw = b""
                    if raw:
                        chunk = raw.decode("utf-8", errors="replace")
                        stdout_chunks.append(chunk)
                        log.write(chunk)
                        log.flush()

                        chunk = strip_ansi(chunk).replace("\r", "\n")
                        pieces = chunk.split("\n")
                        pieces[0] = current + pieces[0]
                        current = pieces[-1]

                        for raw_line in pieces[:-1]:
                            line = raw_line.strip()
                            if not line:
                                continue
                            if mode == "download":
                                update_download_progress(panel, line)
                            clean = sanitize_log_line(line, hide_redirects=hide_redirects)
                            if clean:
                                panel.append(clean, max_lines=tail_lines)

                now = time.time()
                if now - last_refresh >= refresh_interval:
                    elapsed = int(now - start)
                    if mode != "download":
                        panel.set_status(f"running | elapsed {elapsed}s")
                    cur = sanitize_log_line(current, hide_redirects=hide_redirects) or ""
                    panel.render(cur)
                    last_refresh = now

                if proc.poll() is not None:
                    if fd is not None:
                        try:
                            rest = os.read(fd, 65536)
                        except Exception:
                            rest = b""
                        if rest:
                            chunk = rest.decode("utf-8", errors="replace")
                            stdout_chunks.append(chunk)
                            log.write(chunk)
                            chunk = strip_ansi(chunk).replace("\r", "\n")
                            for raw_line in chunk.split("\n"):
                                line = raw_line.strip()
                                if not line:
                                    continue
                                if mode == "download":
                                    update_download_progress(panel, line)
                                clean = sanitize_log_line(line, hide_redirects=hide_redirects)
                                if clean:
                                    panel.append(clean, max_lines=tail_lines)
                    break
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    rc = proc.wait()
    stdout = "".join(stdout_chunks)
    if rc == 0:
        panel.append(f"$ exit {rc}")
        panel.finish(True, "download completed" if mode == "download" else "command completed")
    else:
        panel.append(f"$ exit {rc}")
        panel.append("")
        panel.append("--- tail log ---")
        for line in tail_text(log_path, 40).splitlines():
            clean = sanitize_log_line(line, hide_redirects=hide_redirects)
            if clean:
                panel.append(clean)
        panel.finish(False, f"command failed | exit {rc}")

    result = subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=None)
    result.log_path = str(log_path)  # convenience attribute for notebook users
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=stdout)
    return result


def show_summary(title: str, data=None, *, lines: Optional[Iterable[str]] = None, log_path=None):
    panel = NotebookPanel(title, show_progress=False, height=220)
    panel.display_panel()
    panel.set_status("summary")
    if log_path:
        panel.set_footer(f"log: {log_path}")
    if lines is None:
        if isinstance(data, dict):
            lines = [f"{k}: {v}" for k, v in data.items()]
        elif data is None:
            lines = []
        else:
            lines = [str(data)]
    for line in lines:
        panel.append(str(line))
    panel.render()
    return panel
