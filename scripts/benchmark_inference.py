import ctypes
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any
from ctypes import wintypes

from src.ai.llm_client import LLMClient


def _windows_memory_bytes() -> tuple[int, int]:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    succeeded = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not succeeded:
        error = ctypes.get_last_error()
        raise OSError(error, "Could not read Windows process memory counters.")
    return counters.WorkingSetSize, counters.PeakWorkingSetSize


def _process_memory_bytes() -> tuple[int, int]:
    if sys.platform == "win32":
        return _windows_memory_bytes()

    import resource

    current = Path("/proc/self/statm")
    if current.is_file():
        resident_pages = int(current.read_text(encoding="ascii").split()[1])
        current_bytes = resident_pages * resource.getpagesize()
    else:
        current_bytes = 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    return current_bytes, peak_bytes


def _block_python_sockets(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("The local inference benchmark must not use Python sockets.")


def main() -> int:
    socket.create_connection = _block_python_sockets
    socket.socket.connect = _block_python_sockets

    client = LLMClient()
    current_before, peak_before = _process_memory_bytes()
    started = time.perf_counter()
    response = client.chat(
        [
            {
                "role": "user",
                "content": (
                    "Classify the cognitive action in this short prompt. "
                    "Return one short label only: explain a network protocol."
                ),
            }
        ],
        max_tokens=8,
        timeout_seconds=120,
    )
    elapsed_seconds = time.perf_counter() - started
    current_after, peak_after = _process_memory_bytes()
    report = {
        "model_file": client.model_path.name,
        "elapsed_seconds_including_first_load": round(elapsed_seconds, 3),
        "process_memory_before_bytes": current_before,
        "process_memory_after_bytes": current_after,
        "peak_process_memory_bytes": peak_after,
        "peak_process_memory_increase_bytes": max(0, peak_after - peak_before),
        "python_socket_connections_blocked": True,
        "response_received": bool(response.strip()),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
