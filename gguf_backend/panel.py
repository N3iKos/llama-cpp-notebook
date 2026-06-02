"""Notebook-first realtime output panels for Kaggle/Colab.

All command output is captured to log files. The notebook receives a compact
structured dashboard plus a terminal-like log viewer.

The terminal renderer intentionally keeps one persistent DOM container and
appends log lines incrementally through JavaScript. This avoids the old behavior
where the whole HTML block was replaced, briefly jumping to the top before being
forced back to the bottom. While a process is running, follow-bottom is enabled
by default. If the user scrolls up, follow-bottom pauses. After the process
finishes, follow-bottom is disabled so users can inspect older visible logs.
"""

from __future__ import annotations

import html
import json
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


def sanitize_log_line(line: str, *, max_len=300, hide_redirects=True) -> Optional[str]:
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
    """A single structured output cell: title, status, optional progress, dashboard, terminal, footer."""

    def __init__(self, title="Notebook CLI", *, show_progress=False, height=360, max_lines=700):
        self.title_text = title
        self.show_progress = show_progress
        self.height = height
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.term_id = "cli_term_" + uuid.uuid4().hex
        self.copy_ns = "copy_ns_" + uuid.uuid4().hex
        self.js_api = "__smooth_cli_" + self.term_id
        self._displayed = False
        self._widgets_ok = False
        self._autoscroll = False
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
            self.terminal = widgets.HTML(value=self._terminal_shell_html())
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
        self._init_terminal_js()
        return self

    def _js_call(self, method: str, *args):
        if not self._widgets_ok or not self._displayed:
            return
        payload = ", ".join(json.dumps(arg) for arg in args)
        api_name = json.dumps(self.js_api)
        method_name = json.dumps(method)
        self.display(self.Javascript(f"""
(function callSmoothCli() {{
  const api = window[{api_name}];
  if (!api || !api[{method_name}]) {{
    setTimeout(callSmoothCli, 60);
    return;
  }}
  api[{method_name}]({payload});
}})();
"""))

    def _init_terminal_js(self):
        if not self._widgets_ok:
            return
        self._autoscroll = True
        term_id = json.dumps(self.term_id)
        api_name = json.dumps(self.js_api)
        max_lines = int(self.max_lines)
        self.display(self.Javascript(f"""
(function initSmoothCli() {{
  const id = {term_id};
  const apiName = {api_name};
  const term = document.getElementById(id);
  const linesEl = document.getElementById(id + "_lines");
  const currentEl = document.getElementById(id + "_current");
  const followBtn = document.getElementById(id + "_follow_btn");
  const clearBtn = document.getElementById(id + "_clear_btn");
  const marker = document.getElementById(id + "_marker");

  if (!term || !linesEl || !currentEl) {{
    setTimeout(initSmoothCli, 80);
    return;
  }}

  const model = {{
    follow: true,
    running: true,
    maxLines: {max_lines},
    pending: [],
    scheduled: false
  }};

  function nearBottom() {{
    return (term.scrollHeight - term.scrollTop - term.clientHeight) < 48;
  }}

  function setFollow(value) {{
    model.follow = !!value;
    if (followBtn) followBtn.textContent = "Follow: " + (model.follow ? "on" : "off");
    if (marker) marker.textContent = model.follow ? "following latest log" : "manual scroll";
  }}

  function prune() {{
    while (linesEl.childNodes.length > model.maxLines) {{
      linesEl.removeChild(linesEl.firstChild);
    }}
  }}

  function flush() {{
    model.scheduled = false;
    if (model.pending.length) {{
      const frag = document.createDocumentFragment();
      for (const line of model.pending) {{
        const div = document.createElement("div");
        div.textContent = String(line);
        frag.appendChild(div);
      }}
      model.pending = [];
      linesEl.appendChild(frag);
      prune();
    }}
    if (model.follow) {{
      term.scrollTop = term.scrollHeight;
    }}
  }}

  term.addEventListener("scroll", function() {{
    if (!model.running) return;
    if (!nearBottom()) setFollow(false);
  }});

  if (followBtn) {{
    followBtn.onclick = function() {{
      setFollow(true);
      requestAnimationFrame(function() {{ term.scrollTop = term.scrollHeight; }});
    }};
  }}

  if (clearBtn) {{
    clearBtn.onclick = function() {{
      linesEl.textContent = "";
      currentEl.textContent = "";
    }};
  }}

  window[apiName] = {{
    append: function(lines) {{
      if (!Array.isArray(lines)) lines = [String(lines)];
      for (const line of lines) model.pending.push(String(line));
      if (!model.scheduled) {{
        model.scheduled = true;
        requestAnimationFrame(flush);
      }}
    }},
    current: function(text) {{
      currentEl.textContent = text ? String(text) : "";
      if (model.follow) requestAnimationFrame(function() {{ term.scrollTop = term.scrollHeight; }});
    }},
    setText: function(lines) {{
      if (!Array.isArray(lines)) lines = String(lines).split("\\n");
      linesEl.textContent = "";
      currentEl.textContent = "";
      const frag = document.createDocumentFragment();
      for (const line of lines.slice(-model.maxLines)) {{
        const div = document.createElement("div");
        div.textContent = String(line);
        frag.appendChild(div);
      }}
      linesEl.appendChild(frag);
      if (model.follow) requestAnimationFrame(function() {{ term.scrollTop = term.scrollHeight; }});
    }},
    follow: function(value) {{
      setFollow(value);
      if (value) requestAnimationFrame(function() {{ term.scrollTop = term.scrollHeight; }});
    }},
    done: function() {{
      model.running = false;
      setFollow(false);
      requestAnimationFrame(function() {{ term.scrollTop = term.scrollHeight; }});
    }}
  }};

  setFollow(true);
  window[apiName].setText(["terminal ready"]);
}})();
"""))

    def start_autoscroll(self):
        self._autoscroll = True
        self._js_call("follow", True)

    def stop_autoscroll(self):
        self._autoscroll = False
        self._js_call("done")

    def _title_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:15px;font-weight:700;padding:8px 10px;border:1px solid #333;border-bottom:0;background:#111;color:#f5f5f5;">{esc(text)}</div>'

    def _status_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:8px 10px;border-left:1px solid #333;border-right:1px solid #333;background:#181818;color:#ddd;">{esc(text)}</div>'

    def _progress_info_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#bbb;padding-top:4px;">{esc(text)}</div>'

    def _dashboard_html(self, body):
        if not body:
            return '<div></div>'
        return f'<div style="border-left:1px solid #333;border-right:1px solid #333;background:#0b0b0b;color:#e8e8e8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.35;padding:10px;">{body}</div>'

    def _terminal_shell_html(self):
        return f'''<div style="border:1px solid #333;background:#050505;color:#e8e8e8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;line-height:1.35;">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:7px 10px;background:#101010;border-bottom:1px solid #333;color:#bbb;">
    <div id="{self.term_id}_marker">following latest log</div>
    <div style="display:flex;gap:6px;align-items:center;">
      <button id="{self.term_id}_follow_btn" style="background:#222;color:#eee;border:1px solid #555;border-radius:14px;padding:4px 9px;cursor:pointer;font-family:inherit;font-size:12px;">Follow: on</button>
      <button id="{self.term_id}_clear_btn" style="background:#222;color:#eee;border:1px solid #555;border-radius:14px;padding:4px 9px;cursor:pointer;font-family:inherit;font-size:12px;">Clear view</button>
    </div>
  </div>
  <div id="{self.term_id}" style="height:{self.height}px;overflow-y:auto;padding:10px;white-space:pre-wrap;word-break:break-word;background:#050505;color:#e8e8e8;scroll-behavior:auto;">
    <div id="{self.term_id}_lines"></div>
    <div id="{self.term_id}_current"></div>
  </div>
</div>'''

    def _terminal_html(self, body):
        # Compatibility shim: normal runtime no longer replaces terminal HTML.
        return self._terminal_shell_html()

    def _footer_html(self, text):
        return f'<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;padding:7px 10px;border:1px solid #333;border-top:0;background:#111;color:#aaa;">{esc(text)}</div>'

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
        rows = "".join(f'<div style="padding:2px 0;">{esc(line)}</div>' for line in lines)
        body = f'<div style="font-weight:700;margin-bottom:6px;">{esc(title)}</div>{rows}'
        self.set_dashboard(body)

    def set_links(self, title: str, links: dict[str, str], *, note: str = ""):
        """Render endpoint rows with open/copy actions."""
        cards = [f'<div style="font-weight:700;margin-bottom:8px;">{esc(title)}</div>']
        bindings = []
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

            btn_id = f"btn_{uuid.uuid4().hex}"
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
                    f'<button id="{btn_id}" style="background:#222;color:#eee;border:1px solid #555;'
                    'border-radius:16px;padding:6px 12px;cursor:pointer;">copy</button>'
                )
                bindings.append((btn_id, input_id))

            cards.append(f"""
<div style=\"display:flex;gap:8px;align-items:center;margin:7px 0;\">
  <div style=\"min-width:145px;color:#bbb;\">{esc(label)}</div>
  <input id=\"{input_id}\" value=\"{safe_url_attr}\" readonly style=\"flex:1;background:#050505;color:#e8e8e8;border:1px solid #444;padding:7px;font-family:inherit;font-size:12px;\" />
  <div style=\"display:flex;gap:6px;align-items:center;\">{''.join(actions)}</div>
</div>
""")

        self.set_dashboard("".join(cards))
        if self._widgets_ok:
            for btn_id, input_id in bindings:
                self.display(self.Javascript(f"""
(function attachCopyButton() {{
  const btn = document.getElementById({json.dumps(btn_id)});
  const inp = document.getElementById({json.dumps(input_id)});
  if (!btn || !inp) {{ setTimeout(attachCopyButton, 100); return; }}
  btn.onclick = function() {{
    inp.select();
    const value = inp.value;
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(value);
    }} else {{
      document.execCommand('copy');
    }}
    btn.textContent = 'copied';
    setTimeout(function() {{ btn.textContent = 'copy'; }}, 1200);
  }};
}})();
"""))

    def section(self, text):
        self.append("")
        self.append("═" * 80)
        self.append(text)
        self.append("═" * 80)
        self.render()

    def append(self, line: str, *, max_lines=None):
        if line:
            line = str(line)
            self.lines.append(line)
            self.lines = self.lines[-(max_lines or self.max_lines):]
            if self._widgets_ok and self._displayed:
                self._js_call("append", [line])

    def set_lines(self, lines: Iterable[str]):
        self.lines = [str(x) for x in lines][-self.max_lines:]
        if self._widgets_ok and self._displayed:
            self._js_call("setText", self.lines)

    def render(self, current=""):
        # Smooth terminal mode: do not replace terminal HTML. Updating only the
        # partial/current line keeps the DOM stable and avoids jumpy scrolling.
        if self._widgets_ok and self._displayed:
            self._js_call("current", current.strip() if current else "")
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
        if line:
            self.lines.append(str(line))
            self.lines = self.lines[-(max_lines or 500):]

    def set_lines(self, lines):
        self.lines = list(lines)

    def render(self, current=""):
        pass

    def finish(self, ok=True, message="completed"):
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
    tail_lines=120,
    refresh_interval=0.12,
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
        for line in tail_text(log_path, 40).splitlines():
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
