// Embedded launcher sources. They are extracted to local application data at runtime.

pub const MAIN_PY: &str = r###"from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import threading
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
        self._wallpaper_path = data_dir / "qml" / "hero.svg"
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

        if self._selected_path and not self._game_exists():
            self._append_output("Saved game location is missing. Choose its current location.")
        elif not self._selected_path:
            self._append_output("Launcher ready. Add a game executable to begin.")

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
        return QUrl.fromLocalFile(str(self._wallpaper_path)).toString()

    @Property(str, notify=wallpaperChanged)
    def backgroundSource(self) -> str:
        return self.wallpaperSource

    @Property(str, notify=wallpaperChanged)
    def foregroundSource(self) -> str:
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
            "Choose the game folder",
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
"###;

pub const MAIN_QML: &str = r###"import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 1920
    height: 1080
    minimumWidth: 1120
    minimumHeight: 700
    visible: true
    visibility: Window.Maximized
    title: "VIctory — Game Library"
    color: "#0b0d10"
    font.family: "Segoe UI"

    property int pageIndex: 0
    readonly property color paper: "#f2eee8"
    readonly property color muted: "#a09b94"
    readonly property color accent: "#d83b4f"
    readonly property color iris: "#6257e8"
    readonly property color orchid: "#a445c4"
    readonly property color coral: "#e05c68"
    readonly property color sky: "#2b91ad"
    readonly property color surface: "#c715171b"
    readonly property color edge: "#2a2d32"

    Item {
        anchors.fill: parent
        z: -1
        Image {
            anchors.fill: parent
            source: launcher.backgroundSource
            fillMode: Image.PreserveAspectCrop
            sourceSize.width: 3840
            sourceSize.height: 2160
            asynchronous: true
            smooth: true
            opacity: 0.72
        }
        Rectangle {
            anchors.fill: parent
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0.0; color: "#bf090b10" }
                GradientStop { position: 0.48; color: "#8c090b10" }
                GradientStop { position: 1.0; color: "#5d090b10" }
            }
        }
        Rectangle {
            anchors.fill: parent
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#20060910" }
                GradientStop { position: 1.0; color: "#7a060910" }
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        ColumnLayout {
            id: workspace
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 70
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop { position: 0.0; color: "#eb10121b" }
                    GradientStop { position: 0.48; color: "#e2181421" }
                    GradientStop { position: 1.0; color: "#e00e1821" }
                }
                border.color: "#25282c"
                border.width: 1
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 24
                    anchors.rightMargin: 24
                    spacing: 14
                    RowLayout {
                        Layout.preferredWidth: 190
                        spacing: 10
                        Rectangle {
                            width: 36
                            height: 36
                            radius: 9
                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop { position: 0.0; color: window.iris }
                                GradientStop { position: 1.0; color: window.coral }
                            }
                            border.color: "#8f798e"
                            border.width: 1
                            Text {
                                anchors.centerIn: parent
                                text: "VI"
                                color: "#f0d7c6"
                                font.family: "Georgia"
                                font.pixelSize: 16
                            }
                        }
                        ColumnLayout {
                            spacing: 1
                            Text {
                                text: "VICTORY"
                                color: window.paper
                                font.pixelSize: 12
                                font.weight: Font.Bold
                                font.letterSpacing: 1.5
                            }
                            Text {
                                text: "VIctory COLLECTION"
                                color: "#777b7e"
                                font.pixelSize: 8
                                font.weight: Font.DemiBold
                                font.letterSpacing: 0.7
                            }
                        }
                    }
                    Rectangle { width: 1; height: 30; color: "#303338" }
                    RowLayout {
                        spacing: 5
                        Repeater {
                            model: [
                                { label: "Library", page: 0 },
                                { label: "Activity", page: 1 },
                                { label: "Settings", page: 2 }
                            ]
                            delegate: Rectangle {
                                required property var modelData
                                Layout.preferredWidth: 94
                                Layout.preferredHeight: 40
                                radius: 7
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: window.pageIndex === modelData.page ? "#9a6257e8" : topNavMouse.containsMouse ? "#311f2534" : "transparent" }
                                    GradientStop { position: 1.0; color: window.pageIndex === modelData.page ? "#8ce05c68" : topNavMouse.containsMouse ? "#311f2534" : "transparent" }
                                }
                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.label
                                    color: window.pageIndex === modelData.page ? "#f0e6e0" : "#a0a2a4"
                                    font.pixelSize: 12
                                    font.weight: window.pageIndex === modelData.page ? Font.DemiBold : Font.Normal
                                }
                                Rectangle {
                                    visible: window.pageIndex === modelData.page
                                    width: 24
                                    height: 2
                                    radius: 1
                                    color: window.accent
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.bottom: parent.bottom
                                }
                                MouseArea {
                                    id: topNavMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: window.pageIndex = modelData.page
                                }
                            }
                        }
                    }
                    Item { Layout.fillWidth: true }
                    Rectangle {
                        Layout.preferredWidth: launcher.running ? 112 : 86
                        Layout.preferredHeight: 30
                        radius: 15
                        color: "#191b1e"
                        border.color: "#34373b"
                        border.width: 1
                        RowLayout {
                            anchors.centerIn: parent
                            spacing: 7
                            Rectangle {
                                width: 6
                                height: 6
                                radius: 3
                                color: launcher.running ? "#e2ac71" : "#7fb78a"
                            }
                            Text {
                                text: launcher.running ? "IN SESSION" : "LOCAL"
                                color: "#b9b8b4"
                                font.pixelSize: 9
                                font.weight: Font.Bold
                                font.letterSpacing: 1
                            }
                        }
                    }
                    Rectangle {
                        width: 34
                        height: 34
                        radius: 17
                        color: "#25272b"
                        border.color: "#45484c"
                        border.width: 1
                        Text {
                            anchors.centerIn: parent
                            text: "V"
                            color: "#e4c4ae"
                            font.family: "Georgia"
                            font.pixelSize: 15
                        }
                    }
                }
            }

            Item {
                id: desktopStage
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true

                StackLayout {
                    id: pages
                    anchors.fill: parent
                    currentIndex: window.pageIndex

                Item {
                    Flickable {
                        anchors.fill: parent
                        anchors.leftMargin: 28
                        anchors.rightMargin: 28
                        anchors.topMargin: 24
                        anchors.bottomMargin: 22
                        contentWidth: width
                        contentHeight: libraryContent.implicitHeight
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds

                        ColumnLayout {
                            id: libraryContent
                            width: parent.width
                            spacing: 17

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 48
                                ColumnLayout {
                                    spacing: 2
                                    Text {
                                        text: "YOUR COLLECTION"
                                        color: "#86898c"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.8
                                    }
                                    Text {
                                        text: "Game library"
                                        color: window.paper
                                        font.family: "Georgia"
                                        font.pixelSize: 25
                                    }
                                }
                                Item { Layout.fillWidth: true }
                                Text {
                                    text: "01  TITLE"
                                    color: "#85888c"
                                    font.pixelSize: 10
                                    font.weight: Font.DemiBold
                                    font.letterSpacing: 1.1
                                }
                            }

                            Rectangle {
                                id: featureCard
                                Layout.fillWidth: true
                                Layout.preferredHeight: 500
                                radius: 12
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: "#d3151b2c" }
                                    GradientStop { position: 1.0; color: "#e01b1620" }
                                }
                                border.color: "#5b414d"
                                border.width: 1
                                clip: true

                                RowLayout {
                                    anchors.fill: parent
                                    spacing: 0
                                    Item {
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        Layout.preferredWidth: featureCard.width * 0.5
                                        Layout.minimumWidth: 300
                                        clip: true
                                        Image {
                                            anchors.fill: parent
                                            source: launcher.foregroundSource
                                            fillMode: Image.PreserveAspectCrop
                                            sourceSize.width: 3840
                                            sourceSize.height: 2160
                                            asynchronous: true
                                            smooth: true
                                        }
                                        Rectangle {
                                            anchors.fill: parent
                                            gradient: Gradient {
                                                orientation: Gradient.Horizontal
                                                GradientStop { position: 0.0; color: "#12090b10" }
                                                GradientStop { position: 0.66; color: "#3b090b10" }
                                                GradientStop { position: 1.0; color: "#b90a0c10" }
                                            }
                                        }
                                        Rectangle {
                                            anchors.left: parent.left
                                            anchors.top: parent.top
                                            anchors.leftMargin: 20
                                            anchors.topMargin: 20
                                            width: 144
                                            height: 29
                                            radius: 15
                                            color: "#bb101215"
                                            border.color: "#6e74777a"
                                            border.width: 1
                                            RowLayout {
                                                anchors.centerIn: parent
                                                spacing: 7
                                                Rectangle { width: 6; height: 6; radius: 3; color: window.accent }
                                                Text {
                                                    text: "VIctory GAMES"
                                                    color: "#e2ded9"
                                                    font.pixelSize: 9
                                                    font.weight: Font.Bold
                                                    font.letterSpacing: 0.9
                                                }
                                            }
                                        }
                                        ColumnLayout {
                                            anchors.left: parent.left
                                            anchors.bottom: parent.bottom
                                            anchors.leftMargin: 24
                                            anchors.bottomMargin: 23
                                            spacing: 5
                                            Text {
                                                text: "the city  ·  the city"
                                                color: "#f1c8ad"
                                                font.pixelSize: 10
                                                font.weight: Font.Bold
                                                font.letterSpacing: 1.6
                                            }
                                            Text {
                                                text: "A new chapter begins."
                                                color: "#f2eee9"
                                                font.family: "Georgia"
                                                font.pixelSize: 21
                                            }
                                        }
                                    }

                                    Rectangle {
                                        Layout.fillHeight: true
                                        Layout.fillWidth: true
                                        Layout.preferredWidth: featureCard.width * 0.5
                                        Layout.minimumWidth: 270
                                        gradient: Gradient {
                                            GradientStop { position: 0.0; color: "#e01a1725" }
                                            GradientStop { position: 1.0; color: "#e00d1a20" }
                                        }
                                        ColumnLayout {
                                            width: Math.min(parent.width - 72, 560)
                                            anchors.centerIn: parent
                                            spacing: 0
                                            RowLayout {
                                                Layout.fillWidth: true
                                                Rectangle {
                                                    Layout.preferredWidth: launcher.gameExists ? 76 : 90
                                                    Layout.preferredHeight: 25
                                                    radius: 13
                                                    color: launcher.gameExists ? "#1c3025" : "#332521"
                                                    border.color: launcher.gameExists ? "#365846" : "#614137"
                                                    border.width: 1
                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: launcher.gameExists ? "READY" : "SETUP"
                                                        color: launcher.gameExists ? "#9dcca4" : "#e0ad8d"
                                                        font.pixelSize: 9
                                                        font.weight: Font.Bold
                                                        font.letterSpacing: 1
                                                    }
                                                }
                                                Item { Layout.fillWidth: true }
                                                Text {
                                                    text: "VIctory  /  VI"
                                                    color: "#85888b"
                                                    font.pixelSize: 9
                                                    font.weight: Font.DemiBold
                                                    font.letterSpacing: 0.9
                                                }
                                            }
                                            Text {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 17
                                                text: "Grand Theft\nAuto VI"
                                                color: window.paper
                                                font.family: "Georgia"
                                                font.pixelSize: 34
                                                lineHeight: 0.92
                                            }
                                            Text {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 9
                                                text: launcher.gameExists ? "Your game is ready. Start a session in " + launcher.consoleModel + "." : "Explore the state of the city with the selected game."
                                                color: "#a8a6a1"
                                                font.pixelSize: 12
                                                lineHeight: 1.25
                                                wrapMode: Text.WordWrap
                                            }
                                            Item { Layout.fillHeight: true }
                                            Text {
                                                Layout.fillWidth: true
                                                text: launcher.hasGame ? launcher.selectedPath : "Add the game executable from your existing install."
                                                color: "#86888a"
                                                font.pixelSize: 10
                                                elide: Text.ElideMiddle
                                            }
                                            Rectangle {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 11
                                                Layout.preferredHeight: 51
                                                radius: 8
                                                gradient: Gradient {
                                                    orientation: Gradient.Horizontal
                                                    GradientStop { position: 0.0; color: playMouse.containsMouse ? "#8b6b62ef" : window.iris }
                                                    GradientStop { position: 0.52; color: playMouse.containsMouse ? "#a74f90df" : window.orchid }
                                                    GradientStop { position: 1.0; color: playMouse.containsMouse ? "#f27b72" : window.coral }
                                                }
                                                RowLayout {
                                                    anchors.fill: parent
                                                    anchors.leftMargin: 15
                                                    anchors.rightMargin: 15
                                                    spacing: 9
                                                    Text {
                                                        text: launcher.running ? "●" : "▶"
                                                        color: "white"
                                                        font.pixelSize: 11
                                                    }
                                                    Text {
                                                        Layout.fillWidth: true
                                                        horizontalAlignment: Text.AlignHCenter
                                                        text: launcher.running ? "SESSION IN PROGRESS" : launcher.gameExists ? "PLAY the game" : launcher.hasGame ? "LOCATE GAME FILES" : "ADD GAME FILES"
                                                        color: "white"
                                                        font.pixelSize: 11
                                                        font.weight: Font.Bold
                                                        font.letterSpacing: 0.8
                                                        elide: Text.ElideRight
                                                    }
                                                    Item { width: 12; height: 1 }
                                                }
                                                MouseArea {
                                                    id: playMouse
                                                    anchors.fill: parent
                                                    hoverEnabled: true
                                                    cursorShape: Qt.PointingHandCursor
                                                    enabled: !launcher.running
                                                    onClicked: launcher.gameExists ? launcher.launch() : launcher.browse()
                                                }
                                            }
                                            Rectangle {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 8
                                                Layout.preferredHeight: 34
                                                radius: 7
                                                color: locateMouse.containsMouse ? "#22252a" : "transparent"
                                                Text {
                                                    anchors.centerIn: parent
                                                    text: launcher.hasGame ? "CHANGE GAME LOCATION" : "I'LL ADD FILES LATER"
                                                    color: "#aaa69f"
                                                    font.pixelSize: 9
                                                    font.weight: Font.DemiBold
                                                    font.letterSpacing: 0.8
                                                }
                                                MouseArea {
                                                    id: locateMouse
                                                    anchors.fill: parent
                                                    hoverEnabled: true
                                                    cursorShape: Qt.PointingHandCursor
                                                    onClicked: launcher.browse()
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 150
                                spacing: 12

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 12
                                    gradient: Gradient {
                                        GradientStop { position: 0.0; color: "#db211e30" }
                                        GradientStop { position: 1.0; color: "#de151b22" }
                                    }
                                    border.color: window.edge
                                    border.width: 1
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 9
                                        Text {
                                            text: "GAME INSTALLATION"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            text: launcher.hasGame ? launcher.gameTitle : "Game files not added"
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 16
                                            elide: Text.ElideRight
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            text: launcher.hasGame ? launcher.selectedPath : "Select the the game executable in your existing folder."
                                            color: window.muted
                                            font.pixelSize: 10
                                            elide: Text.ElideMiddle
                                            wrapMode: Text.Wrap
                                        }
                                        Rectangle {
                                            Layout.preferredWidth: 122
                                            Layout.preferredHeight: 29
                                            radius: 6
                                            color: installMouse.containsMouse ? "#282b30" : "#202328"
                                            border.color: "#373b40"
                                            border.width: 1
                                            Text {
                                                anchors.centerIn: parent
                                                text: launcher.hasGame ? "CHANGE LOCATION" : "SELECT FILES"
                                                color: "#d9d3cb"
                                                font.pixelSize: 9
                                                font.weight: Font.DemiBold
                                                font.letterSpacing: 0.5
                                            }
                                            MouseArea {
                                                id: installMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: launcher.browse()
                                            }
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 12
                                    gradient: Gradient {
                                        GradientStop { position: 0.0; color: "#db221a2d" }
                                        GradientStop { position: 1.0; color: "#de17191f" }
                                    }
                                    border.color: window.edge
                                    border.width: 1
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 10
                                        Text {
                                            text: "CONSOLE PROFILE"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            text: launcher.consoleModel
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 18
                                        }
                                        RowLayout {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            spacing: 6
                                            Repeater {
                                                model: ["Series X", "Series S"]
                                                delegate: Rectangle {
                                                    required property string modelData
                                                    Layout.fillWidth: true
                                                    Layout.preferredHeight: 32
                                                    radius: 6
                                                    gradient: Gradient {
                                                        orientation: Gradient.Horizontal
                                                        GradientStop { position: 0.0; color: launcher.consoleModel === modelData ? "#6242a1" : "#25232a" }
                                                        GradientStop { position: 1.0; color: launcher.consoleModel === modelData ? "#a84f76" : "#1b1d22" }
                                                    }
                                                    border.color: launcher.consoleModel === modelData ? "#81505c" : "#363a3f"
                                                    border.width: 1
                                                    Text {
                                                        anchors.centerIn: parent
                                                        text: modelData
                                                        color: launcher.consoleModel === modelData ? "#f0d6cf" : "#aaa9a5"
                                                        font.pixelSize: 10
                                                    }
                                                    MouseArea {
                                                        anchors.fill: parent
                                                        cursorShape: Qt.PointingHandCursor
                                                        onClicked: launcher.setConsoleModel(modelData)
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    radius: 12
                                    gradient: Gradient {
                                        GradientStop { position: 0.0; color: "#db1e202e" }
                                        GradientStop { position: 1.0; color: "#de171923" }
                                    }
                                    border.color: window.edge
                                    border.width: 1
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 15
                                        spacing: 9
                                        Text {
                                            text: "RECENT ACTIVITY"
                                            color: "#888b8f"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.15
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            text: launcher.logs.length > 0 ? launcher.logs[launcher.logs.length - 1] : "Your local launch history will appear here."
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 14
                                            wrapMode: Text.WordWrap
                                            elide: Text.ElideRight
                                        }
                                        Rectangle {
                                            Layout.preferredWidth: 114
                                            Layout.preferredHeight: 29
                                            radius: 6
                                            color: activityMouse.containsMouse ? "#282b30" : "#202328"
                                            border.color: "#373b40"
                                            border.width: 1
                                            Text {
                                                anchors.centerIn: parent
                                                text: "VIEW ACTIVITY"
                                                color: "#d9d3cb"
                                                font.pixelSize: 9
                                                font.weight: Font.DemiBold
                                                font.letterSpacing: 0.5
                                            }
                                            MouseArea {
                                                id: activityMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: window.pageIndex = 1
                                            }
                                        }
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 24
                                Layout.topMargin: 1
                                Text {
                                    text: "QUICK ACCESS"
                                    color: "#c3bdc3"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.6
                                }
                                Item { Layout.fillWidth: true }
                                Text {
                                    text: "LOCAL TOOLS  ·  THIS PC"
                                    color: "#85888e"
                                    font.pixelSize: 8
                                    font.weight: Font.DemiBold
                                    font.letterSpacing: 1
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 74
                                spacing: 10
                                Repeater {
                                    model: [
                                        { icon: "↻", label: "Run again", detail: "Last game", action: "run", c1: "#5140b7", c2: "#9c4eb8" },
                                        { icon: "⌖", label: "Check this PC", detail: "System details", action: "check", c1: "#295aaf", c2: "#5574d5" },
                                        { icon: "▤", label: "Open data folder", detail: "Saves & logs", action: "data", c1: "#246a83", c2: "#399d9d" },
                                        { icon: "⌂", label: "Set game folder", detail: "Install location", action: "folder", c1: "#73519f", c2: "#bb578f" },
                                        { icon: "⌨", label: "Keyboard controls", detail: "Play without pad", action: "controls", c1: "#883f83", c2: "#cf5675" },
                                        { icon: "◉", label: "Console model", detail: launcher.consoleModel, action: "console", c1: "#9a4c58", c2: "#dd6e66" },
                                        { icon: "×", label: "Quit launcher", detail: "Close VIctory", action: "quit", c1: "#563650", c2: "#984f71" }
                                    ]
                                    delegate: Rectangle {
                                        required property var modelData
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        Layout.minimumWidth: 110
                                        radius: 10
                                        opacity: modelData.action === "run" && (!launcher.hasGame || !launcher.gameExists || launcher.running) ? 0.55 : 1.0
                                        gradient: Gradient {
                                            orientation: Gradient.Horizontal
                                            GradientStop { position: 0.0; color: actionMouse.containsMouse ? modelData.c1 : Qt.darker(modelData.c1, 1.24) }
                                            GradientStop { position: 1.0; color: actionMouse.containsMouse ? modelData.c2 : Qt.darker(modelData.c2, 1.32) }
                                        }
                                        border.color: actionMouse.containsMouse ? "#bfc2d8" : "#555360"
                                        border.width: 1
                                        RowLayout {
                                            anchors.fill: parent
                                            anchors.leftMargin: 12
                                            anchors.rightMargin: 9
                                            spacing: 9
                                            Text {
                                                text: modelData.icon
                                                color: "#f5eaf1"
                                                font.pixelSize: 20
                                                font.weight: Font.Light
                                            }
                                            ColumnLayout {
                                                Layout.fillWidth: true
                                                spacing: 3
                                                Text {
                                                    Layout.fillWidth: true
                                                    text: modelData.label
                                                    color: "#fff8f5"
                                                    font.pixelSize: 10
                                                    font.weight: Font.DemiBold
                                                    elide: Text.ElideRight
                                                }
                                                Text {
                                                    Layout.fillWidth: true
                                                    text: modelData.action === "console" ? launcher.consoleModel : modelData.detail
                                                    color: "#dfd5e2"
                                                    font.pixelSize: 8
                                                    elide: Text.ElideRight
                                                }
                                            }
                                        }
                                        MouseArea {
                                            id: actionMouse
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            cursorShape: Qt.PointingHandCursor
                                            enabled: modelData.action !== "run" || (launcher.hasGame && launcher.gameExists && !launcher.running)
                                            onClicked: {
                                                if (modelData.action === "run") launcher.launch()
                                                else if (modelData.action === "check") window.pageIndex = 3
                                                else if (modelData.action === "data") launcher.openDataFolder()
                                                else if (modelData.action === "folder") launcher.setGameFolder()
                                                else if (modelData.action === "controls") controlsDialog.open()
                                                else if (modelData.action === "console") launcher.setConsoleModel(launcher.consoleModel === "Series X" ? "Series S" : "Series X")
                                                else if (modelData.action === "quit") window.close()
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 30
                        spacing: 16
                        RowLayout {
                            Layout.fillWidth: true
                            ColumnLayout {
                                spacing: 3
                                Text {
                                    text: "ACTIVITY"
                                    color: "#85888b"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.6
                                }
                                Text {
                                    text: "Launch history"
                                    color: window.paper
                                    font.family: "Georgia"
                                    font.pixelSize: 27
                                }
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                Layout.preferredWidth: 130
                                Layout.preferredHeight: 34
                                radius: 7
                                color: clearMouse.containsMouse ? "#25282d" : "#191b1f"
                                border.color: "#383c41"
                                border.width: 1
                                Text {
                                    anchors.centerIn: parent
                                    text: "CLEAR HISTORY"
                                    color: "#d0cbc4"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 0.8
                                }
                                MouseArea {
                                    id: clearMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: launcher.clearActivity()
                                }
                            }
                        }
                        Text {
                            text: "the game sessions and launcher messages stored on this PC."
                            color: window.muted
                            font.pixelSize: 12
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: 12
                            gradient: Gradient {
                                GradientStop { position: 0.0; color: "#d91c2033" }
                                GradientStop { position: 1.0; color: "#d9171a20" }
                            }
                            border.color: window.edge
                            border.width: 1
                            ListView {
                                id: activityList
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 10
                                clip: true
                                model: launcher.logs
                                onCountChanged: positionViewAtEnd()
                                delegate: Rectangle {
                                    required property string modelData
                                    required property int index
                                    width: activityList.width
                                    height: message.implicitHeight + 24
                                    radius: 8
                                    color: "#1c1f23"
                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.margins: 12
                                        spacing: 10
                                        Rectangle {
                                            width: 6
                                            height: 6
                                            radius: 3
                                            color: index === activityList.count - 1 ? window.accent : "#74777b"
                                            Layout.alignment: Qt.AlignTop
                                            Layout.topMargin: 4
                                        }
                                        Text {
                                            id: message
                                            Layout.fillWidth: true
                                            text: modelData
                                            color: "#d3d0ca"
                                            font.family: "Cascadia Mono"
                                            font.pixelSize: 11
                                            wrapMode: Text.Wrap
                                        }
                                    }
                                }
                                Text {
                                    anchors.centerIn: parent
                                    visible: activityList.count === 0
                                    text: "No activity yet"
                                    color: "#92918c"
                                    font.family: "Georgia"
                                    font.pixelSize: 19
                                }
                            }
                        }
                    }
                }

                Item {
                    Flickable {
                        anchors.fill: parent
                        anchors.margins: 30
                        contentWidth: width
                        contentHeight: settingsContent.implicitHeight
                        clip: true
                        ColumnLayout {
                            id: settingsContent
                            width: parent.width
                            spacing: 14
                            Text {
                                text: "SETTINGS"
                                color: "#85888b"
                                font.pixelSize: 9
                                font.weight: Font.Bold
                                font.letterSpacing: 1.6
                            }
                            Text {
                                Layout.bottomMargin: 3
                                text: "Launcher preferences"
                                color: window.paper
                                font.family: "Georgia"
                                font.pixelSize: 27
                            }
                            Text {
                                Layout.bottomMargin: 5
                                text: "Manage your the game profile, game files, and local data."
                                color: window.muted
                                font.pixelSize: 12
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 104
                                radius: 12
                                gradient: Gradient {
                                    GradientStop { position: 0.0; color: "#d8211b31" }
                                    GradientStop { position: 1.0; color: "#d71b1924" }
                                }
                                border.color: window.edge
                                border.width: 1
                                RowLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 13
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 4
                                        Text {
                                            text: "CONSOLE PROFILE"
                                            color: "#b88f88"
                                            font.pixelSize: 9
                                            font.weight: Font.Bold
                                            font.letterSpacing: 1.2
                                        }
                                        Text {
                                            text: "New sessions use " + launcher.consoleModel + "."
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 17
                                        }
                                    }
                                    Repeater {
                                        model: ["Series X", "Series S"]
                                        delegate: Rectangle {
                                            required property string modelData
                                            Layout.preferredWidth: 102
                                            Layout.preferredHeight: 34
                                            radius: 7
                                            gradient: Gradient {
                                                orientation: Gradient.Horizontal
                                                GradientStop { position: 0.0; color: launcher.consoleModel === modelData ? "#6242a1" : "#25232a" }
                                                GradientStop { position: 1.0; color: launcher.consoleModel === modelData ? "#a84f76" : "#1b1d22" }
                                            }
                                            border.color: launcher.consoleModel === modelData ? "#81505c" : "#363a3f"
                                            border.width: 1
                                            Text {
                                                anchors.centerIn: parent
                                                text: modelData
                                                color: launcher.consoleModel === modelData ? "#f0d6cf" : "#aaa9a5"
                                                font.pixelSize: 10
                                            }
                                            MouseArea {
                                                anchors.fill: parent
                                                cursorShape: Qt.PointingHandCursor
                                                onClicked: launcher.setConsoleModel(modelData)
                                            }
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 142
                                radius: 12
                                gradient: Gradient {
                                    GradientStop { position: 0.0; color: "#d71c2940" }
                                    GradientStop { position: 1.0; color: "#d9171a25" }
                                }
                                border.color: window.edge
                                border.width: 1
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 9
                                    Text {
                                        text: "GAME LOCATION"
                                        color: "#b88f88"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.hasGame ? launcher.selectedPath : "Choose the the game executable in your existing install folder."
                                        color: window.paper
                                        font.pixelSize: 11
                                        wrapMode: Text.WrapAnywhere
                                        maximumLineCount: 2
                                        elide: Text.ElideMiddle
                                    }
                                    Rectangle {
                                        Layout.preferredWidth: 164
                                        Layout.preferredHeight: 32
                                        radius: 7
                                        color: browseSettingsMouse.containsMouse ? "#282b30" : "#202328"
                                        border.color: "#373b40"
                                        border.width: 1
                                        Text {
                                            anchors.centerIn: parent
                                            text: launcher.hasGame ? "CHANGE LOCATION" : "LOCATE the game"
                                            color: "#ddd6cf"
                                            font.pixelSize: 9
                                            font.weight: Font.DemiBold
                                            font.letterSpacing: 0.7
                                        }
                                        MouseArea {
                                            id: browseSettingsMouse
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            cursorShape: Qt.PointingHandCursor
                                            onClicked: launcher.browse()
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 115
                                radius: 12
                                gradient: Gradient {
                                    GradientStop { position: 0.0; color: "#d824202b" }
                                    GradientStop { position: 1.0; color: "#d7191922" }
                                }
                                border.color: window.edge
                                border.width: 1
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 8
                                    Text {
                                        text: "LOCAL DATA"
                                        color: "#b88f88"
                                        font.pixelSize: 9
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: launcher.dataDirectory
                                        color: window.paper
                                        font.family: "Cascadia Mono"
                                        font.pixelSize: 10
                                        wrapMode: Text.Wrap
                                    }
                                    Text {
                                        text: "Preferences and launch activity stay on this PC."
                                        color: window.muted
                                        font.pixelSize: 10
                                    }
                                }
                            }
                        }
                    }
                }

                Item {
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 34
                        spacing: 18
                        RowLayout {
                            Layout.fillWidth: true
                            ColumnLayout {
                                spacing: 4
                                Text {
                                    text: "SYSTEM PROFILE  ·  LOCAL DIAGNOSTICS"
                                    color: "#a99dd1"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 1.6
                                }
                                Text {
                                    text: "This PC"
                                    color: window.paper
                                    font.family: "Georgia"
                                    font.pixelSize: 30
                                }
                            }
                            Item { Layout.fillWidth: true }
                            Rectangle {
                                Layout.preferredWidth: 132
                                Layout.preferredHeight: 38
                                radius: 9
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: "#634eae" }
                                    GradientStop { position: 1.0; color: "#ac527e" }
                                }
                                Text {
                                    anchors.centerIn: parent
                                    text: "BACK TO LIBRARY"
                                    color: "#fff7f3"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                    font.letterSpacing: 0.7
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: window.pageIndex = 0
                                }
                            }
                        }
                        Text {
                            Layout.fillWidth: true
                            text: "A quick readout of Windows, processor, graphics adapter, and memory. Rendering feature level is confirmed by the game when it starts."
                            color: window.muted
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                        }
                        GridLayout {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            columns: 2
                            rowSpacing: 12
                            columnSpacing: 12
                            Repeater {
                                model: launcher.systemInfo
                                delegate: Rectangle {
                                    required property var modelData
                                    required property int index
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.minimumHeight: 145
                                    radius: 13
                                    gradient: Gradient {
                                        orientation: Gradient.Horizontal
                                        GradientStop { position: 0.0; color: ["#29234a", "#202b4b", "#242548", "#3a2446"][index] }
                                        GradientStop { position: 1.0; color: ["#211b34", "#1c273b", "#1c2e38", "#30202e"][index] }
                                    }
                                    border.color: ["#53427b", "#3e5380", "#3b6973", "#77415e"][index]
                                    border.width: 1
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 8
                                        RowLayout {
                                            Layout.fillWidth: true
                                            Text {
                                                text: modelData.label
                                                color: "#bdb5d5"
                                                font.pixelSize: 9
                                                font.weight: Font.Bold
                                                font.letterSpacing: 1.3
                                            }
                                            Item { Layout.fillWidth: true }
                                            Text {
                                                text: modelData.status
                                                color: modelData.status === "READY" ? "#9be3bd" : modelData.status === "INFO" ? "#a6dce5" : "#efc38e"
                                                font.pixelSize: 8
                                                font.weight: Font.Bold
                                                font.letterSpacing: 0.8
                                            }
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData.value
                                            color: window.paper
                                            font.family: "Georgia"
                                            font.pixelSize: 19
                                            maximumLineCount: 2
                                            elide: Text.ElideRight
                                            wrapMode: Text.WordWrap
                                        }
                                        Item { Layout.fillHeight: true }
                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData.detail
                                            color: "#b4b0ba"
                                            font.pixelSize: 10
                                            wrapMode: Text.WordWrap
                                        }
                                    }
                                }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10
                            Text {
                                Layout.fillWidth: true
                                text: "Game folder: " + (launcher.gameDirectory || "Not set")
                                color: "#aaa4b0"
                                font.pixelSize: 10
                                elide: Text.ElideMiddle
                            }
                            Rectangle {
                                Layout.preferredWidth: 152
                                Layout.preferredHeight: 36
                                radius: 8
                                color: "#25304a"
                                border.color: "#485c87"
                                border.width: 1
                                Text { anchors.centerIn: parent; text: "SET GAME FOLDER"; color: "#e8e1f1"; font.pixelSize: 9; font.weight: Font.DemiBold }
                                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: launcher.setGameFolder() }
                            }
                            Rectangle {
                                Layout.preferredWidth: 144
                                Layout.preferredHeight: 36
                                radius: 8
                                color: "#3d2444"
                                border.color: "#75486f"
                                border.width: 1
                                Text { anchors.centerIn: parent; text: "OPEN DATA FOLDER"; color: "#f0e0ee"; font.pixelSize: 9; font.weight: Font.DemiBold }
                                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: launcher.openDataFolder() }
                            }
                        }
                    }
                }
            }
            }
        }
    }

    Dialog {
        id: controlsDialog
        modal: true
        anchors.centerIn: Overlay.overlay
        width: Math.min(window.width - 40, 780)
        height: Math.min(window.height - 40, 560)
        padding: 0
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        background: Rectangle {
            radius: 18
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0.0; color: "#f0211b35" }
                GradientStop { position: 0.55; color: "#f01d2137" }
                GradientStop { position: 1.0; color: "#f02d1c31" }
            }
            border.color: "#80576f"
            border.width: 1
        }
        contentItem: ColumnLayout {
            anchors.fill: parent
            anchors.margins: 26
            spacing: 15
            RowLayout {
                Layout.fillWidth: true
                ColumnLayout {
                    spacing: 4
                    Text { text: "KEYBOARD CONTROLS"; color: "#d1a8e0"; font.pixelSize: 9; font.weight: Font.Bold; font.letterSpacing: 1.6 }
                    Text { text: "Play your way"; color: window.paper; font.family: "Georgia"; font.pixelSize: 26 }
                }
                Item { Layout.fillWidth: true }
                Rectangle {
                    width: 36; height: 36; radius: 18
                    color: "#3b2b47"; border.color: "#775279"; border.width: 1
                    Text { anchors.centerIn: parent; text: "×"; color: "#f4e6f0"; font.pixelSize: 19 }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: controlsDialog.close() }
                }
            }
            Text {
                Layout.fillWidth: true
                text: "Use these keyboard bindings as a fallback when a controller is not connected."
                color: "#b7afbd"
                font.pixelSize: 11
                wrapMode: Text.WordWrap
            }
            GridLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                columns: 2
                rowSpacing: 9
                columnSpacing: 9
                Repeater {
                    model: [
                        { key: "W A S D  /  I J K L", label: "Left and right sticks" },
                        { key: "ARROW KEYS", label: "D-pad" },
                        { key: "ENTER", label: "A button" },
                        { key: "BACKSPACE", label: "B button" },
                        { key: "X  /  Y", label: "X and Y buttons" },
                        { key: "Q  /  E", label: "Left and right shoulders" },
                        { key: "Z  /  C", label: "Left and right triggers" },
                        { key: "F  /  H", label: "Stick clicks" },
                        { key: "ESC  /  TAB", label: "Menu and View" }
                    ]
                    delegate: Rectangle {
                        required property var modelData
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: 66
                        radius: 10
                        gradient: Gradient {
                            orientation: Gradient.Horizontal
                            GradientStop { position: 0.0; color: "#7b392d3e" }
                            GradientStop { position: 1.0; color: "#68312f47" }
                        }
                        border.color: "#654966"
                        border.width: 1
                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 15
                            anchors.rightMargin: 15
                            spacing: 12
                            Text {
                                text: modelData.key
                                color: "#f1d1e4"
                                font.family: "Cascadia Mono"
                                font.pixelSize: 10
                                font.weight: Font.Bold
                            }
                            Rectangle { width: 1; height: 28; color: "#705270" }
                            Text {
                                Layout.fillWidth: true
                                text: modelData.label
                                color: "#e3dce3"
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                            }
                        }
                    }
                }
            }
        }
    }
}
"###;

pub const HERO_SVG: &str = r###"<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="900" viewBox="0 0 1800 900">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0.06" y2="1">
      <stop offset="0" stop-color="#111821"/>
      <stop offset=".34" stop-color="#3e3340"/>
      <stop offset=".68" stop-color="#bd795c"/>
      <stop offset="1" stop-color="#e1aa77"/>
    </linearGradient>
    <radialGradient id="horizon" cx="72%" cy="66%" r="62%">
      <stop offset="0" stop-color="#e3a871" stop-opacity=".95"/>
      <stop offset=".28" stop-color="#a45650" stop-opacity=".62"/>
      <stop offset="1" stop-color="#29303b" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="haze" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#442f42" stop-opacity=".9"/>
      <stop offset=".55" stop-color="#dd9471" stop-opacity=".18"/>
      <stop offset="1" stop-color="#2d303b" stop-opacity=".7"/>
    </linearGradient>
    <linearGradient id="foreground" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#171a1e" stop-opacity="0"/>
      <stop offset=".28" stop-color="#121518" stop-opacity=".48"/>
      <stop offset="1" stop-color="#0c0f12" stop-opacity=".96"/>
    </linearGradient>
    <filter id="blur">
      <feGaussianBlur stdDeviation="36"/>
    </filter>
    <pattern id="windows" width="34" height="36" patternUnits="userSpaceOnUse">
      <rect x="8" y="8" width="9" height="13" fill="#e4b084" opacity=".25"/>
      <rect x="22" y="8" width="5" height="13" fill="#d4a47e" opacity=".14"/>
    </pattern>
  </defs>

  <rect width="1800" height="900" fill="url(#sky)"/>
  <rect width="1800" height="900" fill="url(#horizon)"/>
  <circle cx="1280" cy="505" r="132" fill="#edb581" opacity=".2" filter="url(#blur)"/>
  <circle cx="1280" cy="505" r="73" fill="#f3c79b" opacity=".3"/>
  <path d="M0 472 C260 440 384 501 594 469 S990 452 1160 478 S1484 439 1800 462 V700 H0Z" fill="url(#haze)" opacity=".65"/>

  <g fill="#252a31" opacity=".46">
    <path d="M0 526h166v-98h42v52h48v-117h68v165h55v-72h60v-91h70v161h86V500h80v-58h60v99h95v-48h58v-82h77v115h62v-78h68v-70h84v143h62v-46h64v-65h87v116h52v-85h71v139H0Z"/>
    <path d="M0 573h70v-64h62v64h39v-94h51v94h61v-51h67v51h76v-113h82v113h67v-57h63v57h59v-98h55v98h81v-74h88v74h56v-111h73v111h78v-62h72v62h74v-97h60v97h90v-63h59v63h58v-92h56v92h47v-118h71v118h62v-72h72v72h80v-45h66v45H0Z" opacity=".72"/>
  </g>

  <g fill="#171b21">
    <path d="M0 612h100v-84h62v84h54V433h104v179h58v-119h62v119h77V470h99v142h70v-71h64v71h58V420h97v192h84v-129h70v129h67V459h92v153h72v-86h68v86h90V402h106v210h99v-115h73v115h81V475h108v137h53v-95h70v95h112v-148h72v148h85v-69h82v69h98v-95h57v95h97v-140h82v140h93v-86h72v86h90v-124h78v124H0Z"/>
  </g>

  <g opacity=".46" fill="url(#windows)">
    <path d="M162 528h62v84h-62zM320 433h104v179H320zM506 470h99v142h-99zM828 420h97v192h-97zM995 483h70v129h-70zM1135 459h92v153h-92zM1299 402h106v210h-106zM1505 497h108v115h-108zM1741 464h82v148h-82z"/>
  </g>

  <g fill="#0e1216">
    <path d="M0 660h122V580h68v80h72V512h91v148h78V562h77v98h70V492h86v168h74v-96h82v96h63V543h108v117h66V470h102v190h74v-89h81v89h80V525h103v135h89V566h83v94h73V491h116v169h85V551h79v109h88V508h78v152h62v-88h84v88h137v-47h82v47h105v-70h65v70h94v-112h96v112h120v-88h78v88h99v-46h85v46H0Z"/>
  </g>

  <g fill="#0b0e11">
    <path d="M0 727h84V669h79v58h56v-37h55v37h65V635h113v92h86v-56h76v56h75V617h114v110h74v-67h87v67h72V650h91v77h104v-99h125v99h69v-48h76v48h83V623h107v104h84v-55h77v55h74V638h117v89h87v-62h90v62h104V653h87v74h83v-45h79v45h74v-102h105v102h90v-55h85v55h100v-76h74v76h87V657h97v70h111v-55h88v55h114v-82h83v82h86v-52h77v52h120V727H0Z"/>
  </g>

  <g stroke="#d8a16f" stroke-width="2" opacity=".3">
    <path d="M0 700H1800"/>
    <path d="M0 725H1800" opacity=".4"/>
  </g>
  <g fill="#efc19a" opacity=".7">
    <rect x="190" y="580" width="4" height="16"/><rect x="397" y="523" width="3" height="21"/>
    <rect x="609" y="570" width="4" height="13"/><rect x="894" y="531" width="4" height="20"/>
    <rect x="1082" y="568" width="3" height="17"/><rect x="1392" y="548" width="4" height="22"/>
    <rect x="1645" y="576" width="3" height="15"/>
  </g>

  <path d="M0 604 C340 575 572 611 780 591 S1187 599 1380 573 S1642 591 1800 565 V900 H0Z" fill="url(#foreground)"/>
  <path d="M0 754 C310 733 572 765 823 744 S1316 758 1800 727 V900 H0Z" fill="#0b0e11" opacity=".55"/>

  <g opacity=".22" stroke="#f2c59a" stroke-width="1">
    <path d="M0 780h1800M0 812h1800M0 845h1800"/>
    <path d="M180 728l-34 172M420 722l-12 178M680 739l8 161M955 733l27 167M1248 744l45 156M1544 733l65 167"/>
  </g>
  <rect width="1800" height="900" fill="#111315" opacity=".12"/>
</svg>
"###;
