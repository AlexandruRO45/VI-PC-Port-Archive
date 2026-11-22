"""Run the native TLS loader test with isolated temporary clones owned by different processes."""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def creation_time() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
    if not kernel32.GetProcessTimes(kernel32.GetCurrentProcess(), created, exited, kernel, user):
        raise ctypes.WinError(ctypes.get_last_error())
    return (created.dwHighDateTime << 32) | created.dwLowDateTime


def main() -> None:
    pid, started = os.getpid(), creation_time()
    with tempfile.TemporaryDirectory(prefix="leonida-tls-sweep-") as tmp:
        # These unmapped files stand in for a live process between CopyFileW and LoadLibraryExW.
        # The old name is also left alone so older checkouts can still start alongside the new loader.
        folder = Path(tmp)
        live = folder / f"leonida_tls_clone_{pid}_{started}_1.dll"
        legacy = folder / f"leonida_tls_host_{pid}_1.dll"
        previous_owner = folder / f"leonida_tls_clone_{pid}_{started - 1}_2.dll"
        with subprocess.Popen([sys.executable, "-I", "-c", "pass"]) as finished:
            finished.wait(timeout=20)
            exited = folder / f"leonida_tls_clone_{finished.pid}_1_1.dll"

        for file in (live, legacy, previous_owner, exited):
            file.write_bytes(b"synthetic TLS clone")

        # Windows may use TMP before TEMP. The isolated directory makes the sweep deterministic.
        env = dict(os.environ, TMP=tmp, TEMP=tmp)
        subprocess.run(sys.argv[1:], env=env, check=True, timeout=30)
        remaining = sorted(p.name for p in folder.iterdir())
        expected = sorted([legacy.name, live.name])
        if remaining != expected:
            raise AssertionError(f"expected only the live clones {expected}, found {remaining}")


if __name__ == "__main__":
    main()
