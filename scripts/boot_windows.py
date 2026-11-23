"""Windows process ownership and external MiniDumpNormal capture for run_boot.py."""
import ctypes as C
from ctypes import wintypes as W
from contextlib import ExitStack, contextmanager
import math
import msvcrt
import os
import subprocess
import sys
import _winapi as win


def bind(dll, name, result, *arguments):
    fn = getattr(dll, name)
    fn.argtypes = arguments
    fn.restype = result
    return fn


kernel = C.WinDLL("kernel32", use_last_error=True)
create_job = bind(kernel, "CreateJobObjectW", W.HANDLE, W.LPVOID, W.LPCWSTR)
set_job = bind(kernel, "SetInformationJobObject", W.BOOL, W.HANDLE, C.c_int, W.LPVOID, W.DWORD)
assign_job = bind(kernel, "AssignProcessToJobObject", W.BOOL, W.HANDLE, W.HANDLE)
terminate_job = bind(kernel, "TerminateJobObject", W.BOOL, W.HANDLE, W.UINT)
resume = bind(kernel, "ResumeThread", W.DWORD, W.HANDLE)


class BasicLimits(C.Structure):
    _fields_ = [("process_time", C.c_int64), ("job_time", C.c_int64), ("flags", W.DWORD),
                ("min_ws", C.c_size_t), ("max_ws", C.c_size_t), ("active", W.DWORD),
                ("affinity", C.c_size_t), ("priority", W.DWORD), ("scheduling", W.DWORD)]


class ExtendedLimits(C.Structure):
    _fields_ = [("basic", BasicLimits), ("io_counters", C.c_uint64 * 6),
                ("memory_limits_and_peaks", C.c_size_t * 4)]


def check(value):
    if not value:
        raise C.WinError(C.get_last_error())
    return value


@contextmanager
def launch(command, cwd, env, stdout, stderr):
    """Assign the suspended root before it can spawn children. Closing the job kills all."""
    with ExitStack() as stack:
        job = check(create_job(None, None))
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        stack.callback(win.CloseHandle, job)
        check(set_job(job, 9, C.byref(limits), C.sizeof(limits)))
        with ExitStack() as inherited:
            stdin = inherited.enter_context(open(os.devnull, "rb"))
            handles = []
            current = win.GetCurrentProcess()
            for stream in (stdin, stdout, stderr):
                handle = win.DuplicateHandle(current, msvcrt.get_osfhandle(stream.fileno()),
                                             current, 0, True, win.DUPLICATE_SAME_ACCESS)
                inherited.callback(win.CloseHandle, handle)
                handles.append(handle)
            info = subprocess.STARTUPINFO(dwFlags=win.STARTF_USESTDHANDLES,
                                          hStdInput=handles[0], hStdOutput=handles[1],
                                          hStdError=handles[2],
                                          lpAttributeList={"handle_list": handles})
            process, thread, pid, _ = win.CreateProcess(
                command[0], subprocess.list2cmdline(command), None, None, True,
                0x4 | subprocess.CREATE_NO_WINDOW, env, str(cwd), info)  # CREATE_SUSPENDED
        stack.callback(win.CloseHandle, process)
        try:
            check(assign_job(job, process))
            if resume(thread) == 0xffffffff:
                raise C.WinError(C.get_last_error())
        except BaseException:
            win.TerminateProcess(process, 1)
            raise
        finally:
            win.CloseHandle(thread)
        try:
            yield process, pid
        finally:
            check(terminate_job(job, 1))
            if wait(process, 5) is None:
                raise TimeoutError("owned process termination unconfirmed")


def wait(process, seconds):
    if win.WaitForSingleObject(process, math.ceil(seconds * 1000)) == win.WAIT_TIMEOUT:
        return None
    return win.GetExitCodeProcess(process)


def write_dump(pid, path):
    # MiniDumpNormal includes thread contexts and stacks. Run out of process so
    # the supervisor can kill this writer even if DbgHelp blocks.
    dbghelp = C.WinDLL("dbghelp", use_last_error=True)
    dump = bind(dbghelp, "MiniDumpWriteDump", W.BOOL, W.HANDLE, W.DWORD, W.HANDLE,
                W.DWORD, W.LPVOID, W.LPVOID, W.LPVOID)
    proc = win.OpenProcess(0x410, False, pid)  # QUERY_INFORMATION | VM_READ
    try:
        with open(path, "xb") as out:
            check(dump(proc, pid, msvcrt.get_osfhandle(out.fileno()), 0, None, None, None))
    finally:
        win.CloseHandle(proc)


if __name__ == "__main__":
    write_dump(int(sys.argv[1]), sys.argv[2])
