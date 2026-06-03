import sys
import time
from pathlib import Path


def clear_live():
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except Exception:
        pass


def live_print(text: str, *, clear: bool = True):
    if clear:
        clear_live()
    print(text)
    sys.stdout.flush()


def tail_text(path, n=80):
    path = Path(path)
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def now_stamp():
    return time.strftime("%H:%M:%S")
