import subprocess
import os
import sys
from typing import List, Optional, Dict, Any, Callable

class ShellError(RuntimeError):
    """Raised when shell command execution fails."""
    def __init__(self, cmd: List[str], returncode: int, stdout: str, stderr: str):
        super().__init__(f"Command '{' '.join(cmd)}' failed with exit code {returncode}.\nStderr: {stderr}")
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run(
    cmd: List[str], 
    check: bool = True, 
    timeout: Optional[float] = None, 
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None
) -> subprocess.CompletedProcess:
    """Executes a command and returns the CompletedProcess object."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
        
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=merged_env,
            cwd=cwd
        )
        if check and res.returncode != 0:
            raise ShellError(cmd, res.returncode, res.stdout, res.stderr)
        return res
    except subprocess.TimeoutExpired as e:
        stdout_str = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr_str = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        raise ShellError(cmd, -1, stdout_str, f"Timeout expired after {timeout} seconds. Stderr: {stderr_str}")
    except OSError as e:
        raise ShellError(cmd, -1, "", f"OS error running command: {str(e)}")


def stream(
    cmd: List[str],
    parser: Optional[Callable[[str], None]] = None,
    live: bool = True,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None
) -> int:
    """Runs a process, streaming its stdout and stderr line-by-line in real-time.
    
    If parser is provided, each line is forwarded to the parser.
    If live is True and no parser is provided, it prints lines directly to standard output.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
        
    # Combine stdout and stderr into stdout to catch everything in order
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1, # Line buffered
        env=merged_env,
        cwd=cwd
    )
    
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # Strip the newline for parsing/printing consistency
                clean_line = line.rstrip("\r\n")
                if parser:
                    parser(clean_line)
                elif live:
                    print(clean_line)
                    sys.stdout.flush()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                
    return process.returncode
