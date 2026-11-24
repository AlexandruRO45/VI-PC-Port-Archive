"""Launch through the host PE loader, or through a selected userspace Wine provider."""
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from contextlib import nullcontext

from .backend_log import game_diagnostics
from .console_cbv import configure_console_cbv
from .mangohud import performance_capture
from .pipeline_profile import pipeline_capture
from .runtime_shaders import runtime_shader_compiler


def proton_installation(selection):
    if selection:
        path = Path(selection).expanduser().resolve()
        return path.parent if path.is_file() and path.name == "proton" else path
    steam = Path(os.environ.get("STEAM_COMPAT_CLIENT_INSTALL_PATH") or Path.home() / ".steam/root")
    libraries = [steam, Path.home() / ".local/share/Steam"]
    vdf = steam / "steamapps/libraryfolders.vdf"
    if vdf.is_file():
        libraries += [Path(p) for p in re.findall(r'"path"\s*"([^"]+)"', vdf.read_text(errors="replace"))]
    for library in libraries:
        candidate = library / "steamapps/common/Proton - Experimental"
        if (candidate / "proton").is_file():
            return candidate.resolve()
    local = steam / "compatibilitytools.d/Proton - Experimental"
    if (local / "proton").is_file():
        return local.resolve()
    raise ValueError("Select an installed Proton distribution with --proton or PROTONPATH")


class Host:
    def __init__(self, backend, proton, prefix, offline=False):
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            raise ValueError("This native CPU baseline requires an x86-64 host")
        self.backend = backend or ("native" if os.name == "nt" else "proton")
        self.env = os.environ.copy()
        self.command = []
        self.version = platform.platform()
        self.offline = False
        if self.backend == "native":
            if offline:
                raise ValueError("--offline requires the Wine or Proton backend on Linux")
            if os.name != "nt" or proton or prefix:
                raise ValueError("Native execution requires Windows and takes no Proton or prefix arguments")
            return
        if sys.platform != "linux":
            raise ValueError("The Wine and Proton backends require Linux")
        if self.backend == "proton":
            installation = proton_installation(proton or os.getenv("PROTONPATH"))
            if not (installation / "proton").is_file():
                raise ValueError(f"Missing Proton distribution launcher in {installation}")
            tool = "umu-run"
            self.env.update(PROTONPATH=str(installation), GAMEID="umu-default")
            version = installation / "version"
            self.version = version.read_text().strip() if version.is_file() else installation.name
            prefix_name = installation.name
        else:
            if proton:
                raise ValueError("--proton selects a distribution for the proton backend")
            tool = "wine"
            prefix_name = "wine"
        executable = shutil.which(tool)
        if not executable:
            raise ValueError(f"Install {tool} to use the {self.backend} backend")
        if self.backend == "wine":
            self.version = subprocess.check_output([executable, "--version"], text=True).strip()
        data_home = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local/share")
        prefix = (prefix or data_home / "project-tuxbox/prefixes" / prefix_name).expanduser().resolve()
        prefix.mkdir(parents=True, exist_ok=True)
        self.env["WINEPREFIX"] = str(prefix)
        self.command = [executable]
        if offline:
            self.isolate_network()

    def isolate_network(self):
        # A new network namespace has only a loopback device that is down, so the
        # game and its crash handler cannot upload minidumps or reach any server.
        unshare = shutil.which("unshare")
        if not unshare:
            raise ValueError("Install util-linux unshare to launch with --offline")
        wrapper = [unshare, "--user", "--map-current-user", "--net", "--"]
        probe = subprocess.run([*wrapper, "true"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if probe.returncode:
            reason = probe.stderr.strip() or f"exit code {probe.returncode}"
            raise ValueError(f"Cannot create an offline network namespace: {reason}")
        self.command = [*wrapper, *self.command]
        # umu-run otherwise tries to update the Steam Runtime before starting.
        self.env["UMU_RUNTIME_UPDATE"] = "0"
        self.offline = True

    def run(self, executable, arguments, *, timeout=None, capture=False):
        options = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT} if capture else {}
        return subprocess.run([*self.command, str(executable), *arguments], cwd=executable.parent,
                              env=self.env, timeout=timeout, **options)

    def run_game(self, executable, arguments, log, *, mangohud=False, profile_pipelines=False):
        env = self.env.copy()
        env.setdefault("TUXBOX_LOG", "stderr" if self.backend == "native" else "1")
        env.pop("TUXBOX_WINE_TITLE_TEMP", None)
        if self.backend in ("proton", "wine") and env.get("WINEPREFIX"):
            prefix = Path(env["WINEPREFIX"])
            drive = prefix / "dosdevices/t:"
            # Proton can remove a user-configured T: during startup. Give the GDK
            # provider its original target so it can reattach it after that point.
            if drive.is_dir():
                env["TUXBOX_WINE_TITLE_TEMP"] = str(drive.resolve())
            elif not os.path.lexists(drive) and not os.path.lexists(drive.with_name("t::")):
                storage = prefix / "drive_t"
                try:
                    storage.mkdir(mode=0o700, exist_ok=True)
                except OSError as exc:
                    log.status(f"Could not prepare the title scratch directory {storage}: {exc}")
                else:
                    env["TUXBOX_WINE_TITLE_TEMP"] = str(storage)
        if "-nullrhi" not in (arg.lower() for arg in arguments):
            configure_console_cbv(self.backend, env, executable, log.status)
        capture = performance_capture(self.backend, env, executable, log) if mangohud else nullcontext()
        pipelines = pipeline_capture(self.backend, env, log) if profile_pipelines else nullcontext()
        with capture, pipelines, runtime_shader_compiler(self.backend, env, executable, log.status), \
             game_diagnostics(self.backend, env, executable.parent, log.stderr_bytes, log_path=log.path) as stderr:
            return log.run([*self.command, str(executable), *arguments], cwd=executable.parent, env=env,
                           stderr_bytes=stderr)
