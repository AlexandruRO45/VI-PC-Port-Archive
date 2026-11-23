"""Stop a Wine prefix's leftover wineserver before Wine runs under a new X server.

`xvfb-run -a` gives every run its own X display and closes it when the run ends, but the prefix's wineserver and
explorer.exe stay a few seconds longer, still on that display. A Wine program started then, under another display,
can block in its first Vulkan call (`wine_vk_init`) and never return, and once one has, the prefix stays stuck
(docs/testing.md, "The Windows runtime on Linux"). `wineserver -k` ends those processes and `wineserver -w` waits
until they are gone. Every Wine process of that prefix ends with them, so this is for a prefix the caller owns.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

TIMEOUT = 30.0  # seconds for each of wineserver -k and -w, so the wait holds when wineserver itself hangs
# wine, wine64 and the packaged wrappers named after them (wine-stable, wine-staging, wine64-devel); not wineserver
WINE_NAME = re.compile(r"wine(64)?([-_.][\w.+-]*)?")
# Where a running wineserver keeps its socket: <root>/.wine-<uid>/server-<device>-<inode of the prefix>/socket
SERVER_ROOT = "/tmp"


def wine_binary(command, env=None):
    """The wine program in a runner command such as ["xvfb-run", "-a", "wine"]: a word named wine, wine64 or a
    wrapper named after them, Proton's wine for a `proton` script, else WINELOADER from env; None when there is none."""
    for word in command:
        name = os.path.basename(word)
        if WINE_NAME.fullmatch(name):
            return word
        if name == "proton":
            found = shutil.which(word) or word
            for wine in (("files", "bin", "wine"), ("dist", "bin", "wine")):
                beside = os.path.join(os.path.dirname(os.path.abspath(found)), *wine)
                if os.path.isfile(beside):
                    return beside
    return (os.environ if env is None else env).get("WINELOADER") or None


def starts_x_server(command):
    """Whether the command gives each run an X server of its own (xvfb-run), so runs need the prefix stopped."""
    return any(os.path.basename(word) == "xvfb-run" for word in command)


def wineserver_for(wine):
    """The wineserver beside `wine` (Proton and /opt/wine-* builds have their own), else the one on PATH."""
    found = shutil.which(wine)
    if found:
        for directory in (os.path.dirname(found), os.path.dirname(os.path.realpath(found))):
            beside = os.path.join(directory, "wineserver")
            if os.path.isfile(beside) and os.access(beside, os.X_OK):
                return beside
    return shutil.which("wineserver")


def server_socket(env=None):
    """The socket a running wineserver of the prefix in env (WINEPREFIX, else ~/.wine) listens on, or None when the
    prefix does not exist. wineserver removes it when it exits."""
    environment = os.environ if env is None else env
    prefix = environment.get("WINEPREFIX") or os.path.join(os.path.expanduser("~"), ".wine")
    try:
        status = os.stat(prefix)
    except OSError:
        return None
    return os.path.join(SERVER_ROOT, f".wine-{os.getuid()}", f"server-{status.st_dev:x}-{status.st_ino:x}", "socket")


def stop_wineserver(command, env=None, timeout=TIMEOUT):
    """Ends the wineserver of the prefix in env (WINEPREFIX, else ~/.wine) for the Wine in `command`, and waits for
    it to exit, each step bounded by timeout. Returns a line saying what happened; one that starts with WARNING when
    the prefix may still have a wineserver (none found to stop, a step that failed, or its socket still there)."""
    wine = wine_binary(command, env)
    if wine is None:
        return "WARNING: no wine in the command (nor WINELOADER), so no wineserver was stopped"
    server = wineserver_for(wine)
    if server is None:
        return f"WARNING: no wineserver found for {wine}, so none was stopped"
    for flag in ("-k", "-w"):
        try:
            subprocess.run([server, flag], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"WARNING: {server} {flag} did not return within {timeout:g} s; going on"
        except OSError as error:
            return f"WARNING: could not run {server} {flag}: {error.strerror or error}"
    # -k exits 1 when no server runs, the usual case, so its exit code says nothing: the socket does.
    socket = server_socket(env) if os.name != "nt" else None
    if socket and os.path.exists(socket):
        return (f"WARNING: {server} -k and -w returned, but the prefix's wineserver socket is still there ({socket}): "
                "a wineserver of another Wine build, or one that did not exit")
    return f"{server} -k and -w: the prefix's earlier Wine processes are gone"


def stop_and_say(command, env=None):
    """stop_wineserver, printing its line on stderr when it is a WARNING."""
    note = stop_wineserver(command, env)
    if note.startswith("WARNING"):
        print(note, file=sys.stderr, flush=True)
    return note


def kill_group(process):
    """SIGKILL for the process group the run started (everything it started that is still in it); on Windows only the
    process itself."""
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:  # nothing of it is left
        pass


def run(command, timeout, cwd=None, env=None):
    """Runs `command` and returns (exit code, or None when it timed out, and its output with stderr after stdout
    lines as they came). The output goes to a file, not a pipe, and the wait is on the process itself: a wineserver
    or explorer.exe the run leaves behind keeps a pipe open long after the program exits, so reading a pipe to its end
    would wait for them. A command that starts its own X server (xvfb-run) gets the prefix stopped before and after
    (stop_wineserver); the program and whatever it started that stays in its process group (not a wineserver, which has
    a session of its own) are killed when it times out, on Ctrl-C, and when the program has exited."""
    fresh_display = starts_x_server(command)
    if fresh_display:
        stop_and_say(command, env)
    with tempfile.TemporaryFile() as output:
        group = {"start_new_session": True} if os.name != "nt" else {}
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, **group)
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as stopped:
                kill_group(process)
                try:
                    process.wait(timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass  # a process SIGKILL does not end (stuck in the kernel) is left, not waited for
                if isinstance(stopped, KeyboardInterrupt):
                    raise
                code = None
            else:
                kill_group(process)  # a helper the program left running does not outlive its test (wineserver has its own session)
        finally:
            if fresh_display:  # also on Ctrl-C: the prefix's wineserver has its own session and would outlive the run
                stop_and_say(command, env)
        output.seek(0)
        return code, output.read().decode("utf-8", "replace")
