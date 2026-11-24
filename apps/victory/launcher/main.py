from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import threading
from urllib.request import Request, urlopen
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtQml import QQmlApplicationEngine


class Launcher(QObject):
    selectedPathChanged = Signal()
    gameTitleChanged = Signal()
    gameExistsChanged = Signal()
    statusChanged = Signal()
    runningChanged = Signal()
    consoleModelChanged = Signal()
    gameDirectoryChanged = Signal()
    logsChanged = Signal()
    wallpaperChanged = Signal()
    outputReceived = Signal(str)
    processFinished = Signal(int)
    wallpaperDownloaded = Signal(bool)

    def __init__(self, victory_exe: Path, repo_root: Path, data_dir: Path) -> None:
        super().__init__()
        self._victory_exe = victory_exe.resolve()
        self._repo_root = repo_root.resolve()
        self._data_dir = data_dir
        self._selection_file = data_dir / "last-executable.bin"
        self._game_directory_file = data_dir / "game-folder.txt"
        self._config_file = data_dir / "victory.toml"
        self._console_file = data_dir / "console-model.txt"
        self._log_file = data_dir / "launcher.log"
        self._wallpaper_path = data_dir / "assets" / "gta6-jason-lucia-landscape.jpg"
        self._background_image_path = data_dir / "assets" / "gta6-background-vice-city.jpg"
        self._foreground_image_path = data_dir / "assets" / "gta6-foreground-rooftop.jpg"
        self._selected_path = self._read_selection()
        self._game_directory = self._read_game_directory()
        if self._game_directory is None and self._selected_path:
            self._game_directory = self._selected_path.parent
        self._console_model = self._read_console_model()
        self._system_info = self._collect_system_info()
        self._running = False
        self._process: subprocess.Popen[str] | None = None
        try:
            self._logs = self._log_file.read_text(encoding="utf-8").splitlines()[-100:]
        except OSError:
            self._logs = []
        self.outputReceived.connect(self._append_output)
        self.processFinished.connect(self._session_finished)
        self.wallpaperDownloaded.connect(self._wallpaper_download_finished)

        if self._selected_path and not self._game_exists():
            self._append_output("Saved game location is missing. Choose its current location.")
        elif not self._selected_path:
            self._append_output("Launcher ready. Add a game executable to begin.")
        if not self._background_image_path.is_file() or not self._foreground_image_path.is_file():
            threading.Thread(target=self._download_artwork, daemon=True).start()

    @Property(str, notify=selectedPathChanged)
    def selectedPath(self) -> str:
        return str(self._selected_path or "")

    @Property(str, notify=gameTitleChanged)
    def gameTitle(self) -> str:
        if self._selected_path:
            return self._selected_path.stem
        return "NO GAME SELECTED"

    @Property(bool, notify=gameExistsChanged)
    def gameExists(self) -> bool:
        return self._game_exists()

    @Property(bool, notify=selectedPathChanged)
    def hasGame(self) -> bool:
        return self._selected_path is not None

    @Property(str, notify=statusChanged)
    def status(self) -> str:
        if self._running:
            return "GAME SESSION IN PROGRESS"
        if self._selected_path and not self._game_exists():
            return "SAVED GAME NOT FOUND"
        if self._selected_path:
            return "READY TO PLAY"
        return "YOUR LIBRARY IS READY"

    @Property(bool, notify=runningChanged)
    def running(self) -> bool:
        return self._running

    @Property(str, notify=consoleModelChanged)
    def consoleModel(self) -> str:
        return self._console_model

    @Property(list, notify=logsChanged)
    def logs(self) -> list[str]:
        return list(self._logs)

    @Property(str, notify=logsChanged)
    def logText(self) -> str:
        return "\n".join(self._logs)

    @Property(str, constant=True)
    def dataDirectory(self) -> str:
        return str(self._data_dir)

    @Property(str, notify=gameDirectoryChanged)
    def gameDirectory(self) -> str:
        return str(self._game_directory or "")

    @Property(str, constant=True)
    def userDataDirectory(self) -> str:
        return str(self._data_dir.parent / "data")

    @Property(list, constant=True)
    def systemInfo(self) -> list[dict[str, str]]:
        return list(self._system_info)

    @Property(str, notify=wallpaperChanged)
    def wallpaperSource(self) -> str:
        if self._wallpaper_path.is_file():
            return QUrl.fromLocalFile(str(self._wallpaper_path)).toString()
        fallback = self._data_dir / "qml" / "hero.svg"
        return QUrl.fromLocalFile(str(fallback)).toString()

    @Property(str, notify=wallpaperChanged)
    def backgroundSource(self) -> str:
        if self._background_image_path.is_file():
            return QUrl.fromLocalFile(str(self._background_image_path)).toString()
        return self.wallpaperSource

    @Property(str, notify=wallpaperChanged)
    def foregroundSource(self) -> str:
        if self._foreground_image_path.is_file():
            return QUrl.fromLocalFile(str(self._foreground_image_path)).toString()
        return self.wallpaperSource

    @Slot()
    def browse(self) -> None:
        initial = str(self._game_directory or "")
        selected, _ = QFileDialog.getOpenFileName(
            None,
            "Add a game to VIctory",
            initial,
            "Windows executables (*.exe);;All files (*.*)",
        )
        if not selected:
            return
        candidate = Path(selected).resolve()
        if candidate.suffix.lower() != ".exe" or not candidate.is_file():
            self._append_output("Choose an existing Windows .exe file.")
            return
        if self._inside_repository(candidate):
            self._append_output("Choose a game executable outside the VIctory source folder.")
            return
        self._selected_path = candidate
        self._game_directory = candidate.parent
        self._save_game_directory(candidate.parent)
        self._save_selection(candidate)
        self.selectedPathChanged.emit()
        self.gameTitleChanged.emit()
        self.gameExistsChanged.emit()
        self.statusChanged.emit()
        self.gameDirectoryChanged.emit()
        self._append_output(f"Added {candidate.name} to the library.")

    @Slot()
    def setGameFolder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            None,
            "Choose the Grand Theft Auto VI folder",
            str(self._game_directory or ""),
        )
        if not selected:
            return
        folder = Path(selected).resolve()
        if self._inside_repository(folder):
            self._append_output("Choose a game folder outside the VIctory source folder.")
            return
        self._game_directory = folder
        self._save_game_directory(folder)
        self.gameDirectoryChanged.emit()
        if self._selected_path and not self._is_within(self._selected_path, folder):
            self._selected_path = None
            self._selection_file.unlink(missing_ok=True)
            self.selectedPathChanged.emit()
            self.gameTitleChanged.emit()
            self.gameExistsChanged.emit()
            self.statusChanged.emit()
            self._append_output("Game executable cleared because it is outside the new folder.")
        else:
            self._append_output(f"Game folder set to {folder}.")

    @Slot()
    def openDataFolder(self) -> None:
        folder = self._data_dir.parent / "data"
        folder.mkdir(parents=True, exist_ok=True)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._append_output(f"Could not open the data folder: {folder}")

    @Slot()
    def launch(self) -> None:
        if self._running:
            return
        if not self._selected_path or not self._game_exists():
            self._append_output("Choose an existing game executable before pressing Play.")
            return
        if self._inside_repository(self._selected_path):
            self._append_output("Choose a game executable outside the VIctory source folder.")
            return
        if not self._victory_exe.is_file():
            self._append_output(f"VIctory runner not found: {self._victory_exe}")
            return

        command = [
            str(self._victory_exe),
            "run",
            str(self._selected_path),
            "--execute",
        ]
        if self._console_model == "Series S":
            command.extend(["--console", "series-s"])
        environment = os.environ.copy()
        environment["VICTORY_GAME_DIR"] = str(self._game_directory or self._selected_path.parent)
        if self._config_file.is_file():
            environment["VICTORY_CONFIG"] = str(self._config_file)
        self._append_output(f"Launching {self._selected_path.name} with {self._console_model}.")
        self._running = True
        self.runningChanged.emit()
        self.statusChanged.emit()

        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self._selected_path.parent),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            self._running = False
            self.runningChanged.emit()
            self.statusChanged.emit()
            self._append_output(f"Could not start the VIctory runner: {error}")
            return

        threading.Thread(target=self._read_process, args=(self._process,), daemon=True).start()

    @Slot(str)
    def setConsoleModel(self, model: str) -> None:
        if model not in {"Series X", "Series S"} or model == self._console_model:
            return
        self._console_model = model
        self._console_file.write_text(model, encoding="utf-8")
        self.consoleModelChanged.emit()
        self._append_output(f"Console profile set to {model}.")

    @Slot()
    def clearActivity(self) -> None:
        self._logs.clear()
        try:
            self._log_file.write_text("", encoding="utf-8")
        except OSError:
            pass
        self.logsChanged.emit()

    def _download_artwork(self) -> None:
        images = (
            (
                self._background_image_path,
                "https://www.rockstargames.com/VI/_next/static/media/"
                "Vice_City_Postcard_landscape.0v2njmlk2n-qm.jpg"
                "?akim=1&imdensity=1&imwidth=3840",
            ),
            (
                self._foreground_image_path,
                "https://www.rockstargames.com/VI/_next/static/media/"
                "Jason_Lucia_Rooftop_landscape.0c6p5do~-8~7..jpg"
                "?akim=1&imdensity=1&imwidth=3840",
            ),
        )
        downloaded = False
        for path, url in images:
            if path.is_file():
                continue
            try:
                request = Request(url, headers={"User-Agent": "VIctory Launcher/1.0"})
                with urlopen(request, timeout=20) as response:
                    content_type = response.headers.get_content_type()
                    image = response.read()
                if content_type != "image/jpeg" or len(image) < 100_000:
                    raise ValueError("Rockstar returned an unexpected artwork response")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".download")
                temporary.write_bytes(image)
                temporary.replace(path)
                downloaded = True
            except Exception:
                continue
        self.wallpaperDownloaded.emit(downloaded)

    @Slot(bool)
    def _wallpaper_download_finished(self, succeeded: bool) -> None:
        if succeeded:
            self.wallpaperChanged.emit()

    def _inside_repository(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._repo_root)
            return True
        except ValueError:
            return False

    def _game_exists(self) -> bool:
        return bool(self._selected_path and self._selected_path.is_file())

    def _read_selection(self) -> Path | None:
        try:
            raw = self._selection_file.read_bytes()
            value = raw.decode("utf-16-le").rstrip("\0")
            return Path(value) if value else None
        except (OSError, UnicodeError):
            return None

    def _save_selection(self, path: Path) -> None:
        self._selection_file.write_bytes(str(path).encode("utf-16-le"))

    def _read_game_directory(self) -> Path | None:
        try:
            value = self._game_directory_file.read_text(encoding="utf-8").strip()
            return Path(value).resolve() if value else None
        except OSError:
            return None

    def _save_game_directory(self, folder: Path) -> None:
        self._game_directory_file.write_text(str(folder), encoding="utf-8")
        self._config_file.write_text(
            "[paths]\n" f"game_dir = {json.dumps(folder.as_posix())}\n",
            encoding="utf-8",
        )

    @staticmethod
    def _is_within(path: Path, folder: Path) -> bool:
        try:
            path.resolve().relative_to(folder.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _collect_system_info() -> list[dict[str, str]]:
        system = platform.system()
        release = platform.release()
        build = platform.version()
        architecture = platform.machine() or "Unknown"
        processor = platform.processor() or "Processor name unavailable"
        avx2: bool | None = None
        adapters: list[str] = []
        memory_gb = "Not reported"

        if os.name == "nt":
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                ) as key:
                    processor = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            except OSError:
                pass
            try:
                avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
            except (AttributeError, OSError):
                pass
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Video",
                ) as root:
                    for index in range(32):
                        try:
                            adapter = winreg.EnumKey(root, index)
                            with winreg.OpenKey(root, adapter + r"\0000") as key:
                                for name in ("DriverDesc", "HardwareInformation.AdapterString"):
                                    try:
                                        value = winreg.QueryValueEx(key, name)[0]
                                        if isinstance(value, str) and value.strip():
                                            if value not in adapters:
                                                adapters.append(value.strip())
                                            break
                                    except OSError:
                                        continue
                        except OSError:
                            break
            except OSError:
                pass
            try:
                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                memory = MemoryStatus()
                memory.dwLength = ctypes.sizeof(memory)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                    memory_gb = f"{memory.ullTotalPhys / (1024**3):.0f} GB"
            except (AttributeError, OSError):
                pass

        avx2_text = "Available" if avx2 is True else "Not available" if avx2 is False else "Not probed"
        gpu_text = ", ".join(adapters) if adapters else "Adapter name unavailable"
        return [
            {
                "label": "OPERATING SYSTEM",
                "value": f"{system} {release}",
                "detail": f"Build {build}",
                "status": "READY" if system == "Windows" else "CHECK",
            },
            {
                "label": "PROCESSOR",
                "value": processor,
                "detail": f"{architecture} · {os.cpu_count() or 'Unknown'} logical processors · AVX2 {avx2_text}",
                "status": "READY" if avx2 is True else "CHECK",
            },
            {
                "label": "DISPLAY ADAPTER",
                "value": gpu_text,
                "detail": "Game rendering requires Direct3D 12, Shader Model 6.6, and mesh shaders.",
                "status": "CHECK AT LAUNCH",
            },
            {
                "label": "SYSTEM MEMORY",
                "value": memory_gb,
                "detail": "Detected from Windows system memory.",
                "status": "INFO",
            },
        ]

    def _read_console_model(self) -> str:
        try:
            value = self._console_file.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        return value if value in {"Series X", "Series S"} else "Series X"

    def _read_process(self, process: subprocess.Popen[str]) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self.outputReceived.emit(line.rstrip())
            self.processFinished.emit(process.wait())
        except Exception as error:
            self.outputReceived.emit(f"Game output reader failed: {error}")
            self.processFinished.emit(-1)

    @Slot(str)
    def _append_output(self, message: str) -> None:
        clean = message.strip()
        if not clean:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"{stamp}   {clean}"
        self._logs.append(line)
        self._logs = self._logs[-200:]
        try:
            with self._log_file.open("a", encoding="utf-8") as log:
                log.write(line + "\n")
        except OSError:
            pass
        self.logsChanged.emit()

    @Slot(int)
    def _session_finished(self, exit_code: int) -> None:
        self._process = None
        self._running = False
        self.runningChanged.emit()
        self.statusChanged.emit()
        if exit_code == 0:
            self._append_output("Game session ended normally.")
        else:
            self._append_output(f"Game session ended with exit code {exit_code}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="VIctory Qt Quick launcher")
    parser.add_argument("--victory-exe", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    arguments = parser.parse_args()

    data_dir = Path(os.environ["VICTORY_HOME"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    application = QApplication(sys.argv)
    application.setApplicationName("VIctory Launcher")
    application.setOrganizationName("VIctory")
    engine = QQmlApplicationEngine()
    engine.warnings.connect(
        lambda warnings: sys.stderr.write(
            "\n".join(error.toString() for error in warnings) + "\n"
        )
    )
    bridge = Launcher(arguments.victory_exe, arguments.repo_root, data_dir)
    engine.rootContext().setContextProperty("launcher", bridge)
    qml_file = data_dir / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_file)))
    if not engine.rootObjects():
        print(f"Could not load launcher QML: {qml_file}", file=sys.stderr)
        return 1
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
