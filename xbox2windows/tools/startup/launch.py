"""Run one EXE in a kill-on-close job with a timeout, a commit cap and captured stdout/stderr.

usage: launch.py <exe> <cwd> <logdir> [--timeout S] [--memory-gib N] [--max-processes N] [-- args...]
Prints the exit code (hex), how it ended and the trap log.
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys
import time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR), ("lpTitle", wt.LPWSTR),
                ("dwX", wt.DWORD), ("dwY", wt.DWORD), ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD),
                ("dwXCountChars", wt.DWORD), ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD),
                ("dwFlags", wt.DWORD), ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", wt.HANDLE), ("hStdOutput", wt.HANDLE),
                ("hStdError", wt.HANDLE)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wt.HANDLE), ("hThread", wt.HANDLE), ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [("Basic", BASIC_LIMIT), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("nLength", wt.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p), ("bInheritHandle", wt.BOOL)]


k32.CreateFileW.restype = wt.HANDLE
k32.CreateJobObjectW.restype = wt.HANDLE


def inheritable_file(path):
    sa = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, True)
    h = k32.CreateFileW(path, 0x40000000, 1 | 2, ctypes.byref(sa), 2, 0x80, None)  # GENERIC_WRITE, CREATE_ALWAYS
    if h in (None, wt.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), path)
    return h


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        extra = argv[argv.index("--") + 1:]
        argv = argv[:argv.index("--")]
    exe, cwd, logdir = (os.path.abspath(a) for a in argv[:3])
    opts = dict(zip(argv[3::2], argv[4::2]))
    timeout = float(opts.get("--timeout", 15))
    memory = int(float(opts.get("--memory-gib", 16)) * 2**30)
    max_processes = int(opts.get("--max-processes", 1))
    os.makedirs(logdir, exist_ok=True)
    trap_log = os.path.join(logdir, "trap.log")
    if os.path.exists(trap_log):
        os.remove(trap_log)

    k32.SetErrorMode(0x0001 | 0x0002 | 0x8000)  # FAILCRITICALERRORS | NOGPFAULTERRORBOX | NOOPENFILEERRORBOX

    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED_LIMIT()
    # KILL_ON_JOB_CLOSE | DIE_ON_UNHANDLED_EXCEPTION | JOB_MEMORY | ACTIVE_PROCESS (when limited)
    info.Basic.LimitFlags = 0x2000 | 0x400 | 0x200 | (0x8 if max_processes else 0)
    info.Basic.ActiveProcessLimit = max_processes
    info.JobMemoryLimit = memory
    if not k32.SetInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject")

    out = inheritable_file(os.path.join(logdir, "stdout.txt"))
    err = inheritable_file(os.path.join(logdir, "stderr.txt"))
    si = STARTUPINFOW()
    si.cb = ctypes.sizeof(si)
    si.dwFlags = 0x100  # STARTF_USESTDHANDLES
    si.hStdOutput, si.hStdError = out, err
    env = dict(os.environ, X2W_TRAP_LOG=trap_log)
    block = "".join(f"{k}={v}\0" for k, v in env.items()) + "\0"
    cmdline = ctypes.create_unicode_buffer(" ".join(f'"{a}"' for a in [exe, *extra]))
    pi = PROCESS_INFORMATION()
    flags = 0x4 | 0x400  # CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT
    if not k32.CreateProcessW(exe, cmdline, None, None, True, flags, ctypes.create_unicode_buffer(block),
                              cwd, ctypes.byref(si), ctypes.byref(pi)):
        print(f"CreateProcess failed: error {ctypes.get_last_error()}")
        return 2
    if not k32.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(pi.hProcess)):
        k32.TerminateProcess(wt.HANDLE(pi.hProcess), 1)
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
    start = time.time()
    k32.ResumeThread(wt.HANDLE(pi.hThread))
    result = k32.WaitForSingleObject(wt.HANDLE(pi.hProcess), int(timeout * 1000))
    ended = "exited"
    if result == 0x102:  # WAIT_TIMEOUT
        k32.TerminateJobObject(wt.HANDLE(job), 0xDEAD0001)
        k32.WaitForSingleObject(wt.HANDLE(pi.hProcess), 5000)
        ended = f"killed after {timeout:g} s timeout"
    code = wt.DWORD()
    k32.GetExitCodeProcess(wt.HANDLE(pi.hProcess), ctypes.byref(code))
    k32.QueryInformationJobObject(wt.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info), None)
    elapsed = time.time() - start
    for h in (out, err, pi.hThread, pi.hProcess):
        k32.CloseHandle(wt.HANDLE(h))
    print(f"pid {pi.dwProcessId} {ended} in {elapsed:.2f} s, exit code {code.value:#010x}, "
          f"peak job memory {info.PeakJobMemoryUsed / 2**20:.1f} MiB")
    for name in ("stdout.txt", "stderr.txt", "trap.log"):
        path = os.path.join(logdir, name)
        text = open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else None
        print(f"--- {name}: " + ("(absent)" if text is None else f"{len(text)} chars"))
        if text:
            print(text[-4000:])
    k32.CloseHandle(wt.HANDLE(job))
    return 0


if __name__ == "__main__":
    sys.exit(main())
