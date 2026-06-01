import sys
import time
from typing import List, Optional

# Track last output refresh to allow optional throttling
_last_print_time = 0.0

def clear() -> None:
    """Clears the current cell output cleanly."""
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except ImportError:
        # Fallback for plain terminal execution
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()


def live_print(lines: List[str], title: Optional[str] = None, force: bool = False, min_interval: float = 0.2) -> None:
    """Prints a clean status card in the notebook.
    
    Avoids scroll-bar spam by clearing the output and flushing stdout.
    Throttles updates based on min_interval unless force=True.
    """
    global _last_print_time
    now = time.time()
    if not force and (now - _last_print_time) < min_interval:
        return
        
    _last_print_time = now
    clear()
    
    output = []
    if title:
        output.append(f"=== {title} ===")
    output.extend(lines)
    
    print("\n".join(output))
    sys.stdout.flush()
