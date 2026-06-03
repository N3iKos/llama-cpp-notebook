"""Notebook-first realtime output panels for Kaggle/Colab.

This version is intentionally conservative for Kaggle:
- one ipywidgets panel is displayed per cell;
- no repeated display(Javascript(...)) calls during streaming;
- command stdout/stderr is captured to log files;
- the visible terminal is a widget state update, not thousands of notebook outputs.

The terminal uses a bottom-anchored CSS layout so the latest log stays visible
without forcing scroll via JavaScript on every update. Full raw output is kept in
log files; the panel shows a notebook-friendly live tail.
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


def default_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    if Path("/content").exists():
        return Path("/content")
    return Path.cwd()


def default_log_dir() -> Path:
    path = default_root() / "_logs"
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


def sanitize_log_line(line: str, *, max_len=1200, hide_redirects=True) -> Optional[str]:
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
        or "response-content-type" in low
        or "x-goog-signature" in low
        or "x-amz-signature" in low
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


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "command").strip().lower()).strip("_") or "command"


class NotebookPanel:
    """A single structured output cell: title, status, optional progress, dashboard, terminal, footer.

    Supports optional live monitoring mode with a shutdown button and live status
    indicator. When ``add_shutdown_button`` is called, a control bar with a red
    shutdown button and a live status dot is inserted between the dashboard and
    the terminal.
    """

    def __init__(self, title="Notebook CLI", *, show_progress=False, height=360, max_lines=800):
        self.title_text = title
        self.show_progress = show_progress
        self.height = height
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.term_id = "cli_term_" + uuid.uuid4().hex
        self._displayed = False
        self._widgets_ok = False
        self._shutdown_btn = None
        self._live_status_html_widget = None
        self._live_control_box = None
        self._init_widgets()

    def _init_widgets(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display

            self.widgets = widgets
            self.display = display
            self.title = widgets.HTML(value=self._title_html(self.title_text))
            self.status = widgets.HTML(value=self._status_html("idle"))
            self.progress = widgets.FloatProgress(
                value=0,
                min=0,
                max=100,
                description="0%",
                bar_style="",
                layout=widgets.Layout(width="100%"),
            )
            self.progress_info = widgets.HTML(value=self._progress_info_html("-"))
            self.progress_box = widgets.VBox(
                [self.progress, self.progress_info],
                layout=widgets.Layout(
                    border="1px solid #333",
                    padding="8px 10px",
                    display="block" if self.show_progress else "none",
                ),
            )
            self.dashboard = widgets.HTML(value=self._dashboard_html(""))
            self.dashboard.layout.display = "none"
            self.terminal = widgets.HTML(value=self._terminal_html("terminal ready"))
            self.footer = widgets.HTML(value=self._footer_html("log: -"))
            self.ui = widgets.VBox([self.title, self.status, self.progress_box, self.dashboard, self.terminal, self.footer])
            self._widgets_ok = True
        except Exception:
            self._widgets_ok = False

    def display_panel(self):
        if self._displayed:
            return self
        self._displayed = True
        if not self._widgets_ok:
            print(self.title_text)
            return self
        self.display(self.ui)
        return self

    def start_autoscroll(self):
        # Kept for API compatibility. The CSS terminal is bottom-anchored.
        pass

    def stop_autoscroll(self):
        # Kept for API compatibility. User can scroll inside the terminal after completion.
        pass

    def _title_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:15px;font-weight:700;padding:8px 10px;border:1px solid #333;border-bottom:0;background:#111;color:#f5f5f5;">{esc(text)}</div>'

    def _status_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:8px 10px;border-left:1px solid #333;border-right:1px solid #333;background:#181818;color:#ddd;">{esc(text)}</div>'

    def _progress_info_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#bbb;padding-top:4px;white-space:pre-wrap;word-break:break-word;">{esc(text)}</div>'

    def _dashboard_html(self, body):
        if not body:
            return '<div></div>'
        return f'<div style="border-left:1px solid #333;border-right:1px solid #333;background:#0b0b0b;color:#e8e8e8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.35;padding:10px;">{body}</div>'

    def _terminal_html(self, current=""):
        visible = self.lines[-self.max_lines:]
        if current:
            visible = visible + [current]
        if not visible:
            visible = ["terminal ready"]

        # Bottom-anchored trick: render newest-to-oldest inside column-reverse.
        # The visual order remains old-to-new, with latest anchored near the bottom,
        # without repeated JavaScript scroll commands.
        reversed_lines = list(reversed([str(x) for x in visible]))
        rows = "".join(
            f'<div style="min-height:1.35em;white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;">{esc(line)}</div>'
            for line in reversed_lines
        )
        meta = f"live tail: last {min(len(visible), self.max_lines)} lines | full log in footer"
        return f'''<div style="border:1px solid #333;background:#050505;color:#e8e8e8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.35;">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:7px 10px;background:#101010;border-bottom:1px solid #333;color:#bbb;">
    <div>{esc(meta)}</div>
  </div>
  <div id="{self.term_id}" style="height:{self.height}px;overflow-y:auto;padding:10px;white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;background:#050505;color:#e8e8e8;display:flex;flex-direction:column-reverse;">
    {rows}
  </div>
</div>'''

    def _footer_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;padding:7px 10px;border:1px solid #333;border-top:0;background:#111;color:#aaa;white-space:pre-wrap;word-break:break-word;">{esc(text)}</div>'

    def set_title(self, text):
        self.title_text = text
        if self._widgets_ok:
            self.title.value = self._title_html(text)

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

    def set_dashboard(self, html_body: str):
        if not self._widgets_ok:
            return
        self.dashboard.value = self._dashboard_html(html_body)
        self.dashboard.layout.display = "block" if html_body else "none"

    def set_summary(self, title: str, lines: Iterable[str] | None = None, data=None):
        if lines is None:
            if isinstance(data, dict):
                lines = [f"{k}: {v}" for k, v in data.items()]
            elif data is None:
                lines = []
            else:
                lines = [str(data)]
        rows = "".join(f'<div style="padding:2px 0;white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;">{esc(line)}</div>' for line in lines)
        body = f'<div style="font-weight:700;margin-bottom:6px;">{esc(title)}</div>{rows}'
        self.set_dashboard(body)

    def set_links(self, title: str, links: dict[str, str], *, note: str = ""):
        """Render endpoint rows with open/copy actions. Copy uses inline JS, no streaming display calls."""
        cards = [f'<div style="font-weight:700;margin-bottom:8px;">{esc(title)}</div>']
        if note:
            cards.append(f'<div style="color:#bbb;margin-bottom:8px;">{esc(note)}</div>')

        for label, item in links.items():
            if isinstance(item, dict):
                url = str(item.get("url", ""))
                can_copy = bool(item.get("copy", True))
                can_open = bool(item.get("open", False))
            else:
                url = str(item)
                can_copy = True
                low_label = str(label).lower()
                can_open = any(k in low_label for k in ("public", "playground", "base")) and url.startswith(("http://", "https://"))

            input_id = f"inp_{uuid.uuid4().hex}"
            safe_url_attr = html.escape(url, quote=True)
            actions = []
            if can_open:
                actions.append(
                    f'<a href="{safe_url_attr}" target="_blank" rel="noopener noreferrer" '
                    'style="background:#1f2937;color:#f8fafc;border:1px solid #64748b;'
                    'border-radius:16px;padding:6px 12px;text-decoration:none;cursor:pointer;">open</a>'
                )
            if can_copy:
                actions.append(
                    f'<button onclick="(async()=>{{const i=document.getElementById(\'{input_id}\'); if(i){{try{{await navigator.clipboard.writeText(i.value); this.textContent=\'copied\'; setTimeout(()=>this.textContent=\'copy\',1200);}}catch(e){{i.select();document.execCommand(\'copy\');}}}}}})()" '
                    'style="background:#222;color:#eee;border:1px solid #555;border-radius:16px;padding:6px 12px;cursor:pointer;">copy</button>'
                )

            cards.append(f"""
<div style=\"display:flex;gap:8px;align-items:center;margin:7px 0;\">
  <div style=\"min-width:145px;color:#bbb;\">{esc(label)}</div>
  <input id=\"{input_id}\" value=\"{safe_url_attr}\" readonly style=\"flex:1;background:#050505;color:#e8e8e8;border:1px solid #444;padding:7px;font-family:inherit;font-size:12px;min-width:120px;\" />
  <div style=\"display:flex;gap:6px;align-items:center;\">{''.join(actions)}</div>
</div>
""")

        self.set_dashboard("".join(cards))

    def section(self, text):
        self.append("")
        self.append("═" * 80)
        self.append(text)
        self.append("═" * 80)
        self.render()

    def append(self, line: str, *, max_lines=None):
        if line is not None:
            line = str(line)
            if line:
                self.lines.append(line)
                keep = max(self.max_lines, int(max_lines or 0)) if max_lines else self.max_lines
                self.lines = self.lines[-keep:]

    def set_lines(self, lines: Iterable[str]):
        self.lines = [str(x) for x in lines][-self.max_lines:]
        self.render()

    def render(self, current=""):
        if self._widgets_ok and self._displayed:
            self.terminal.value = self._terminal_html(current.strip() if current else "")
        elif current:
            print(current)

    def finish(self, ok=True, message="completed"):
        self.set_status(message)
        if self.show_progress and ok:
            self.set_progress(100, "100% | completed", "success")
        elif self.show_progress and not ok and self._widgets_ok:
            self.progress.bar_style = "danger"
        self.render()
        self.stop_autoscroll()

    # -----------------------------------------------------------------
    # Live monitoring controls (shutdown button + live status indicator)
    # -----------------------------------------------------------------

    def _live_status_dot_html(self, text, color="#4ade80"):
        """Render a color-coded status dot with text.

        Colors:
            #4ade80 = green (running / healthy)
            #f87171 = red (stopped / error)
            #facc15 = yellow (shutting down / warning)
            #60a5fa = blue (starting)
        """
        return (
            f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
            f'font-size:13px;display:flex;align-items:center;gap:8px;padding:0 4px;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};box-shadow:0 0 6px {color};"></span>'
            f'<span style="color:#e8e8e8;">{esc(text)}</span></div>'
        )

    def add_shutdown_button(self, callback):
        """Insert a live control bar (shutdown button + status dot) between
        the dashboard and the terminal.

        ``callback`` is invoked when the user clicks the shutdown button.
        The button is disabled immediately to prevent double-clicks.
        """
        if not self._widgets_ok:
            return

        widgets = self.widgets

        # Shutdown button — prominent red styling.
        btn = widgets.Button(
            description="⏻ Shutdown Server",
            button_style="danger",
            icon="power-off",
            layout=widgets.Layout(
                width="auto",
                min_width="180px",
                height="38px",
            ),
        )
        btn.style.button_color = "#dc2626"
        btn.style.font_weight = "bold"

        # Live status indicator — starts as "running".
        live_status = widgets.HTML(
            value=self._live_status_dot_html("server running", "#4ade80"),
        )

        # Container box: flex row with button on left, status on right.
        control_box = widgets.HBox(
            [btn, live_status],
            layout=widgets.Layout(
                display="flex",
                justify_content="space-between",
                align_items="center",
                padding="8px 10px",
                border="1px solid #333",
                border_bottom="none",
            ),
        )

        # Store references for later updates.
        self._shutdown_btn = btn
        self._live_status_html_widget = live_status
        self._live_control_box = control_box

        # Insert the control box into the VBox between dashboard and terminal.
        # Layout order: title, status, progress_box, dashboard, *control_box*, terminal, footer
        children = list(self.ui.children)
        # Find the terminal index (should be the 5th element, index 4).
        try:
            term_idx = children.index(self.terminal)
        except ValueError:
            term_idx = len(children) - 1
        children.insert(term_idx, control_box)
        self.ui.children = tuple(children)

        # Wire up the click handler.
        def _on_click(_btn):
            _btn.disabled = True
            _btn.description = "⏳ Shutting down..."
            _btn.style.button_color = "#78716c"
            self.set_live_status("shutting down...", "#facc15")
            try:
                callback()
            except Exception as exc:
                self.append(f"shutdown error: {exc}")
                self.render()

        btn.on_click(_on_click)

    def set_live_status(self, text, color="#4ade80"):
        """Update the live status dot text and color."""
        if self._live_status_html_widget is not None:
            self._live_status_html_widget.value = self._live_status_dot_html(text, color)
        if not self._widgets_ok:
            print(f"[live] {text}")

    def set_shutdown_complete(self):
        """Mark the panel as fully shut down — disable button, red status dot."""
        if self._shutdown_btn is not None:
            self._shutdown_btn.disabled = True
            self._shutdown_btn.description = "Shutdown Complete"
            self._shutdown_btn.style.button_color = "#374151"
        self.set_live_status("server stopped", "#f87171")
        self.set_status("shutdown complete")


class SilentPanel:
    def __init__(self):
        self.lines = []
        self.show_progress = False

    def display_panel(self):
        return self

    def start_autoscroll(self):
        pass

    def stop_autoscroll(self):
        pass

    def set_footer(self, text):
        pass

    def set_status(self, text):
        pass

    def set_title(self, text):
        pass

    def set_progress_visible(self, visible):
        self.show_progress = visible

    def set_progress(self, pct, info="", style="info"):
        pass

    def set_dashboard(self, html_body):
        pass

    def set_summary(self, title, lines=None, data=None):
        pass

    def set_links(self, title, links, note=""):
        pass

    def section(self, text):
        self.append(text)

    def append(self, line, *, max_lines=None):
        if line is not None:
            self.lines.append(str(line))
            self.lines = self.lines[-(max_lines or 500):]

    def set_lines(self, lines):
        self.lines = list(lines)

    def render(self, current=""):
        pass

    def finish(self, ok=True, message="completed"):
        pass

    # Live monitoring stubs for API compatibility.
    def add_shutdown_button(self, callback):
        pass

    def set_live_status(self, text, color="#4ade80"):
        pass

    def set_shutdown_complete(self):
        pass


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


def update_download_progress(panel, line: str) -> bool:
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
    tail_lines=500,
    refresh_interval=0.25,
    hide_redirects=True,
    show=True,
    panel=None,
    finalize=True,
):
    """Run command via Popen and render realtime output into a notebook panel."""
    log_dir = Path(log_dir) if log_dir else default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _safe_filename(label)
    log_path = Path(log_dir) / (log_name or f"{int(time.time())}_{safe_label}.log")

    if panel is None:
        panel = NotebookPanel(label, show_progress=(mode == "download")) if show else SilentPanel()
        if show:
            panel.display_panel()
    else:
        panel.display_panel()

    panel.start_autoscroll()
    panel.set_progress_visible(mode == "download")
    panel.set_footer(f"log: {log_path}")
    panel.set_status("starting download..." if mode == "download" else f"running: {label}")
    if mode == "download":
        panel.set_progress(0, "0%", "")

    shown_cmd = command_text(cmd)
    panel.append("$ " + shown_cmd, max_lines=tail_lines)
    panel.render()

    shell = isinstance(cmd, str)
    env2 = os.environ.copy()
    if env:
        env2.update(env)

    start = time.time()
    stdout_chunks: list[str] = []
    current = ""
    last_refresh = 0.0

    proc = subprocess.Popen(
        cmd,
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
                    if finalize:
                        panel.finish(False, f"timeout after {timeout}s")
                    raise TimeoutError(f"{label} timeout after {timeout}s")

                ready = []
                if fd is not None:
                    ready, _, _ = select.select([fd], [], [], 0.1)

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
                        panel.set_status(f"running: {label} | elapsed {elapsed}s")
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
    panel.append(f"$ exit {rc}", max_lines=tail_lines)

    if rc == 0:
        panel.set_status("download completed" if mode == "download" else f"completed: {label}")
        if mode == "download":
            panel.set_progress(100, "100% | completed", "success")
        panel.render()
        if finalize:
            panel.finish(True, "download completed" if mode == "download" else f"completed: {label}")
    else:
        panel.append("", max_lines=tail_lines)
        panel.append("--- tail log ---", max_lines=tail_lines)
        for line in tail_text(log_path, 80).splitlines():
            clean = sanitize_log_line(line, hide_redirects=hide_redirects)
            if clean:
                panel.append(clean, max_lines=tail_lines)
        panel.render()
        if finalize:
            panel.finish(False, f"failed: {label} | exit {rc}")
        else:
            panel.set_status(f"failed: {label} | exit {rc}")

    result = subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=None)
    result.log_path = str(log_path)
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=stdout)
    return result


def show_summary(title: str, data=None, *, lines: Optional[Iterable[str]] = None, log_path=None, panel=None, finalize=True):
    own_panel = panel is None
    if panel is None:
        panel = NotebookPanel(title, show_progress=False, height=220)
        panel.display_panel()
    if lines is None:
        if isinstance(data, dict):
            lines = [f"{k}: {v}" for k, v in data.items()]
        elif data is None:
            lines = []
        else:
            lines = [str(data)]
    if log_path:
        panel.set_footer(f"log: {log_path}")
    panel.set_summary(title, lines=lines)
    if own_panel:
        panel.set_lines([str(line) for line in lines])
        panel.render()
        if finalize:
            panel.finish(True, "summary")
    return panel
