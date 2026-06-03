import sys
import time
from pathlib import Path

from .panel import NotebookPanel, show_summary, tail_text

_LAST_PANEL = None


def clear_live():
    # Kept for backward compatibility. New rendering is widget-based and does not
    # clear whole notebook cells.
    pass


def live_print(text: str, *, clear: bool = True):
    """Compatibility wrapper: render text inside a compact widget panel."""
    global _LAST_PANEL
    if _LAST_PANEL is None or clear:
        _LAST_PANEL = NotebookPanel("Notebook output", show_progress=False, height=260)
        _LAST_PANEL.display_panel()
    _LAST_PANEL.lines = []
    for line in str(text).splitlines():
        _LAST_PANEL.append(line)
    _LAST_PANEL.set_status("updated")
    _LAST_PANEL.render()
    sys.stdout.flush()


def now_stamp():
    return time.strftime("%H:%M:%S")
