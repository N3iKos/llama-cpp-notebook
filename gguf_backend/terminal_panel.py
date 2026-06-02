import html
import os
import re
import select
import shlex
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass
class TerminalPanelConfig:
    label: str = "process"
    command_display: str | None = None
    log_path: str | Path | None = None
    refresh_interval: float = 0.08
    tail_lines: int = 80
    failure_tail_lines: int = 30
    cwd: str | Path | None = None
    env: dict[str, str] | None = None


@dataclass
class TerminalPanelResult:
    returncode: int
    status: str
    log_path: Path
    tail_lines: list[str]
    elapsed_seconds: float


class CommandPanelError(RuntimeError):
    def __init__(
        self,
        *,
        label: str,
        returncode: int,
        log_path: Path,
        tail_lines: list[str],
        elapsed_seconds: float,
    ):
        self.label = label
        self.returncode = returncode
        self.log_path = log_path
        self.tail_lines = tail_lines
        self.elapsed_seconds = elapsed_seconds

        details = "\n".join(tail_lines)
        super().__init__(
            "\n".join(
                [
                    f"{label} failed with exit code {returncode}.",
                    f"log: {log_path}",
                    f"elapsed: {elapsed_seconds:.1f}s",
                    "tail:",
                    details,
                ]
            )
        )


class NotebookTerminalPanel:
    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._handle = None

    def update(
        self,
        *,
        config: TerminalPanelConfig,
        status: str,
        elapsed_seconds: float,
        exit_code: int | None,
        lines: Sequence[str],
    ) -> None:
        if not self.enabled:
            return

        try:
            from IPython.display import HTML, display
        except Exception:
            if status in {"DONE", "FAILED"}:
                print(f"{config.label}: {status} exit={exit_code} log={_resolve_log_path(config)}")
            return

        html_text = render_terminal_panel_html(
            config=config,
            status=status,
            elapsed_seconds=elapsed_seconds,
            exit_code=exit_code,
            lines=lines,
        )

        if self._handle is None:
            self._handle = display(HTML(html_text), display_id=True)
        else:
            self._handle.update(HTML(html_text))


def command_to_display(cmd: str | Sequence[str]) -> str:
    if isinstance(cmd, str):
        return cmd
    return " ".join(shlex.quote(str(part)) for part in cmd)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def tail_from_lines(lines: Sequence[str], count: int) -> list[str]:
    if count <= 0:
        return []
    return list(lines[-count:])


def render_terminal_panel_html(
    *,
    config: TerminalPanelConfig,
    status: str,
    elapsed_seconds: float,
    exit_code: int | None,
    lines: Sequence[str],
) -> str:
    status_class = status.lower()
    exit_text = "exit -" if exit_code is None else f"exit {exit_code}"
    log_path = _resolve_log_path(config)
    command_display = config.command_display or ""

    escaped_label = html.escape(config.label)
    escaped_command = html.escape(command_display)
    escaped_log = html.escape(str(log_path))
    escaped_lines = "\n".join(html.escape(strip_ansi(line)) for line in lines)

    return f"""
<style>
.ios-terminal-panel {{
  overflow: hidden;
  margin: 12px 0;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  background:
    linear-gradient(145deg, rgba(42,47,58,0.92), rgba(16,18,24,0.94));
  box-shadow: 0 18px 48px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.08);
  color: #e8edf7;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}}
.ios-terminal-titlebar {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: rgba(255,255,255,0.055);
  border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.ios-terminal-lights {{
  display: flex;
  gap: 7px;
}}
.traffic-light {{
  width: 12px;
  height: 12px;
  border-radius: 999px;
  box-shadow: inset 0 0 0 1px rgba(0,0,0,0.16);
}}
.traffic-light.red {{ background: #ff5f57; }}
.traffic-light.yellow {{ background: #febc2e; }}
.traffic-light.green {{ background: #28c840; }}
.ios-terminal-title {{
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #f6f8fc;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0;
}}
.ios-terminal-badge {{
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0;
}}
.ios-terminal-badge.running {{ color: #c7f7ff; background: rgba(65,196,255,0.16); }}
.ios-terminal-badge.done {{ color: #c9ffd8; background: rgba(40,200,64,0.18); }}
.ios-terminal-badge.failed {{ color: #ffd0d0; background: rgba(255,95,87,0.20); }}
.ios-terminal-meta {{
  display: grid;
  grid-template-columns: repeat(3, max-content) minmax(0, 1fr);
  gap: 10px;
  padding: 10px 16px 0;
  color: #aeb8c8;
  font-size: 12px;
}}
.ios-terminal-meta span {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.ios-terminal-command {{
  padding: 8px 16px 0;
  color: #d7e0ef;
  font-size: 12px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.ios-terminal-output {{
  margin: 10px 0 0;
  padding: 0 16px 16px;
  max-height: 460px;
  overflow: auto;
  color: #e9eef8;
  font-size: 12px;
  line-height: 1.45;
  white-space: pre-wrap;
}}
</style>
<div class="ios-terminal-panel">
  <div class="ios-terminal-titlebar">
    <div class="ios-terminal-lights">
      <span class="traffic-light red"></span>
      <span class="traffic-light yellow"></span>
      <span class="traffic-light green"></span>
    </div>
    <div class="ios-terminal-title">{escaped_label}</div>
    <span class="ios-terminal-badge {status_class}">{html.escape(status)}</span>
  </div>
  <div class="ios-terminal-meta">
    <span>{elapsed_seconds:.1f}s</span>
    <span>{html.escape(exit_text)}</span>
    <span>log</span>
    <span title="{escaped_log}">{escaped_log}</span>
  </div>
  <div class="ios-terminal-command">$ {escaped_command}</div>
  <pre class="ios-terminal-output">{escaped_lines}</pre>
</div>
""".strip()


def run_terminal_panel(
    cmd: str | Sequence[str],
    config: TerminalPanelConfig | None = None,
    *,
    display: bool = True,
    timeout: float | None = None,
    check: bool = True,
    prefer_pty: bool = True,
) -> TerminalPanelResult:
    config = config or TerminalPanelConfig()
    if config.command_display is None:
        config.command_display = command_to_display(cmd)

    log_path = _resolve_log_path(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if _can_use_pty(prefer_pty):
        return _run_with_pty(cmd, config, log_path, display=display, timeout=timeout, check=check)
    return _run_with_pipe(cmd, config, log_path, display=display, timeout=timeout, check=check)


def update_terminal_panel_from_log(
    *,
    config: TerminalPanelConfig,
    status: str,
    start_time: float,
    exit_code: int | None,
    display: bool = True,
) -> None:
    log_path = _resolve_log_path(config)
    lines = _read_log_tail(log_path, config.failure_tail_lines if status == "FAILED" else config.tail_lines)
    NotebookTerminalPanel(enabled=display).update(
        config=config,
        status=status,
        elapsed_seconds=time.time() - start_time,
        exit_code=exit_code,
        lines=lines,
    )


def _run_with_pipe(
    cmd: str | Sequence[str],
    config: TerminalPanelConfig,
    log_path: Path,
    *,
    display: bool,
    timeout: float | None,
    check: bool,
) -> TerminalPanelResult:
    start = time.time()
    panel = NotebookTerminalPanel(enabled=display)
    visible = deque(maxlen=max(1, config.tail_lines))
    all_lines: list[str] = []
    last_update = 0.0

    proc = subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(config.cwd) if config.cwd is not None else None,
        env=config.env,
        text=True,
        bufsize=1,
    )

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            while True:
                if timeout and time.time() - start > timeout:
                    proc.kill()
                    proc.wait()
                    _raise_timeout(config, proc.returncode, log_path, all_lines, start)

                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    _record_text(line, log, visible, all_lines)

                now = time.time()
                if now - last_update >= config.refresh_interval:
                    panel.update(
                        config=config,
                        status="RUNNING",
                        elapsed_seconds=now - start,
                        exit_code=None,
                        lines=list(visible),
                    )
                    last_update = now

                if proc.poll() is not None:
                    rest = proc.stdout.read() if proc.stdout else ""
                    if rest:
                        _record_text(rest, log, visible, all_lines)
                    break

                if not line:
                    time.sleep(min(0.05, config.refresh_interval))
        finally:
            if proc.poll() is None:
                proc.kill()

    return _finish_process(proc.returncode, config, log_path, all_lines, start, panel, check)


def _run_with_pty(
    cmd: str | Sequence[str],
    config: TerminalPanelConfig,
    log_path: Path,
    *,
    display: bool,
    timeout: float | None,
    check: bool,
) -> TerminalPanelResult:
    import pty

    start = time.time()
    panel = NotebookTerminalPanel(enabled=display)
    visible = deque(maxlen=max(1, config.tail_lines))
    all_lines: list[str] = []
    last_update = 0.0
    master_fd, slave_fd = pty.openpty()

    proc = subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=str(config.cwd) if config.cwd is not None else None,
        env=config.env,
        close_fds=True,
    )
    os.close(slave_fd)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            while True:
                if timeout and time.time() - start > timeout:
                    proc.kill()
                    proc.wait()
                    _raise_timeout(config, proc.returncode, log_path, all_lines, start)

                readable, _, _ = select.select([master_fd], [], [], min(0.05, config.refresh_interval))
                if readable:
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        chunk = b""
                    if chunk:
                        _record_text(chunk.decode("utf-8", "replace"), log, visible, all_lines)

                now = time.time()
                if now - last_update >= config.refresh_interval:
                    panel.update(
                        config=config,
                        status="RUNNING",
                        elapsed_seconds=now - start,
                        exit_code=None,
                        lines=list(visible),
                    )
                    last_update = now

                if proc.poll() is not None:
                    while True:
                        readable, _, _ = select.select([master_fd], [], [], 0)
                        if not readable:
                            break
                        try:
                            chunk = os.read(master_fd, 4096)
                        except OSError:
                            break
                        if not chunk:
                            break
                        _record_text(chunk.decode("utf-8", "replace"), log, visible, all_lines)
                    break
        finally:
            try:
                os.close(master_fd)
            except OSError:
                pass
            if proc.poll() is None:
                proc.kill()

    return _finish_process(proc.returncode, config, log_path, all_lines, start, panel, check)


def _finish_process(
    returncode: int | None,
    config: TerminalPanelConfig,
    log_path: Path,
    all_lines: Sequence[str],
    start: float,
    panel: NotebookTerminalPanel,
    check: bool,
) -> TerminalPanelResult:
    code = int(returncode if returncode is not None else -1)
    elapsed = time.time() - start
    status = "DONE" if code == 0 else "FAILED"
    visible_count = config.tail_lines if code == 0 else config.failure_tail_lines
    tail = tail_from_lines(all_lines, visible_count)

    panel.update(
        config=config,
        status=status,
        elapsed_seconds=elapsed,
        exit_code=code,
        lines=tail,
    )

    if check and code != 0:
        raise CommandPanelError(
            label=config.label,
            returncode=code,
            log_path=log_path,
            tail_lines=tail,
            elapsed_seconds=elapsed,
        )

    return TerminalPanelResult(
        returncode=code,
        status=status,
        log_path=log_path,
        tail_lines=tail,
        elapsed_seconds=elapsed,
    )


def _record_text(text: str, log, visible: deque[str], all_lines: list[str]) -> None:
    log.write(text)
    log.flush()
    for line in text.splitlines():
        clean = line.rstrip("\r")
        visible.append(clean)
        all_lines.append(clean)


def _read_log_tail(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    return tail_from_lines(path.read_text(encoding="utf-8", errors="replace").splitlines(), count)


def _resolve_log_path(config: TerminalPanelConfig) -> Path:
    if config.log_path is not None:
        return Path(config.log_path)

    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config.label.strip().lower()).strip("-") or "process"
    return Path.cwd() / f"{safe_label}.log"


def _can_use_pty(prefer_pty: bool) -> bool:
    if not prefer_pty or os.name == "nt":
        return False
    try:
        import pty  # noqa: F401

        return True
    except Exception:
        return False


def _raise_timeout(
    config: TerminalPanelConfig,
    returncode: int | None,
    log_path: Path,
    all_lines: Sequence[str],
    start: float,
) -> None:
    raise CommandPanelError(
        label=f"{config.label} timeout",
        returncode=int(returncode if returncode is not None else -1),
        log_path=log_path,
        tail_lines=tail_from_lines(all_lines, config.failure_tail_lines),
        elapsed_seconds=time.time() - start,
    )
