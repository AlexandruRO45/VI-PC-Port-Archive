"""Stop a Wine prefix's leftover wineserver before Wine runs under a new X server.

`xvfb-run -a` gives every run its own X display and closes it when the run ends, but the prefix's wineserver and
explorer.exe stay a few seconds longer, still on that display. A Wine program started then, under another display,
can block in its first Vulkan call (`wine_vk_init`) and never return, and once one has, the prefix stays stuck
(docs/testing.md, "The Windows runtime on Linux"). `wineserver -k` ends those processes and `wineserver -w` waits
until they are gone. Every Wine process of that prefix ends with them, so this is for a prefix the caller owns.
"""
import os
import shutil
import signal
import subprocess
import tempfile

TIMEOUT = 30.0  # seconds for each of wineserver -k and -w, so the wait holds when wineserver itself hangs


def wine_binary(command):
    """The wine program in a runner command such as ["xvfb-run", "-a", "wine"], or None when it has none."""
    for word in command:
        if os.path.basename(word) in ("wine", "wine64"):
            return word
    return None


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


def stop_wineserver(command, env=None, timeout=TIMEOUT):
    """Ends the wineserver of the prefix in env (WINEPREFIX, else ~/.wine) for the Wine in `command`, and waits for
    it to exit, each step bounded by timeout. Returns a line saying what happened."""
    wine = wine_binary(command)
    if wine is None:
        return "no wine in the command, so no wineserver to stop"
    server = wineserver_for(wine)
    if server is None:
        return f"no wineserver found for {wine}, so none stopped"
    for flag in ("-k", "-w"):
        try:
            subprocess.run([server, flag], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"{server} {flag} did not return within {timeout:g} s; going on"
        except OSError as error:
            return f"could not run {server} {flag}: {error.strerror or error}"
    return f"{server} -k and -w: the prefix's earlier Wine processes are gone"


def run(command, timeout, cwd=None, env=None):
    """Runs `command` and returns (exit code, or None when it timed out, and its output with stderr after stdout
    lines as they came). The output goes to a file, not a pipe, and the wait is on the process itself: a wineserver
    or explorer.exe the run leaves behind keeps a pipe open long after the program exits, so reading a pipe to its end
    would wait for them. A command that starts its own X server (xvfb-run) gets the prefix stopped before and after
    (stop_wineserver); a timed-out command is killed with everything it started."""
    fresh_display = starts_x_server(command)
    if fresh_display:
        stop_wineserver(command, env)
    with tempfile.TemporaryFile() as output:
        group = {"start_new_session": True} if os.name != "nt" else {}
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, **group)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            code = None
        if fresh_display:
            stop_wineserver(command, env)
        output.seek(0)
        return code, output.read().decode("utf-8", "replace")
