"""Leonida Desktop GUI (PySide6) with Outfit typography, OS language detection, and startup setup check."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui import APP_NAME, APP_VERSION, RELEASES_URL, REPO_URL, SITE_URL
from gui.core import (
    ALL_STAGES,
    WAVE64_MODES,
    BuildConfig,
    build_make_game_args,
    ensure_output_dir,
    format_free_space,
    format_validation_check_details,
    load_config,
    locate_python,
    resolve_repo_root,
    save_config,
    tr,
)
from gui.setup_dialog import SetupDialog

STYLESHEET = """
* {
    font-family: "Outfit", "Segoe UI", sans-serif;
}
QMainWindow, QDialog, QWidget#RootWindow {
    background-color: #090b11;
    color: #eef1f8;
}
QFrame#Sidebar {
    background-color: #0e121d;
    border-right: 1px solid #1b2236;
}
QPushButton[nav="true"] {
    text-align: left;
    padding: 11px 16px;
    border-radius: 10px;
    border: none;
    background: transparent;
    color: #9ba6c1;
    font-size: 14px;
    font-weight: 500;
}
QPushButton[nav="true"]:hover {
    background-color: #161d30;
    color: #ffffff;
}
QPushButton[nav="true"][active="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff4d8d, stop:1 #7c4dff);
    color: #ffffff;
    font-weight: 600;
}
QFrame[card="true"] {
    background-color: #111624;
    border: 1px solid #1e263d;
    border-radius: 14px;
}
QLabel[heading="true"] {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
}
QLabel[subheading="true"] {
    font-size: 13px;
    color: #8f9bb8;
}
QLabel[section="true"] {
    font-size: 14px;
    font-weight: 600;
    color: #d7dff2;
}
QLineEdit, QComboBox {
    background-color: #0a0d16;
    border: 1px solid #242e4a;
    border-radius: 9px;
    padding: 9px 12px;
    color: #eef1f8;
    font-size: 13px;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #ff4d8d;
}
QPushButton[primary="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff4d8d, stop:1 #ff7a59);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton[primary="true"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff629b, stop:1 #ff8d70);
}
QPushButton[primary="true"]:disabled {
    background: #252c40;
    color: #6d7791;
}
QPushButton[danger="true"] {
    background-color: #dc2646;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 11px 20px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton[secondary="true"] {
    background-color: #192136;
    color: #dce3f5;
    border: 1px solid #2a3554;
    border-radius: 9px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton[secondary="true"]:hover {
    background-color: #232d48;
    border-color: #3d4e7a;
}
QCheckBox {
    spacing: 8px;
    color: #dce3f5;
    font-size: 13px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #354366;
    background-color: #0a0d16;
}
QCheckBox::indicator:checked {
    background-color: #ff4d8d;
    border-color: #ff4d8d;
}
QProgressBar {
    background-color: #0a0d16;
    border: 1px solid #242e4a;
    border-radius: 8px;
    height: 14px;
    text-align: center;
    color: #ffffff;
    font-size: 11px;
    font-weight: 600;
}
QProgressBar::chunk {
    border-radius: 7px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d4ff, stop:1 #ff4d8d);
}
QPlainTextEdit {
    background-color: #07090f;
    border: 1px solid #1e263d;
    border-radius: 10px;
    padding: 10px;
    color: #d6e0f5;
    font-family: "Outfit", "Consolas", monospace;
    font-size: 12px;
}
QScrollArea {
    border: none;
    background: transparent;
}
"""


def load_outfit_fonts(repo_root: str) -> str:
    """Register bundled Outfit .ttf weights with Qt and return the resolved family name."""
    fonts_dir = os.path.join(repo_root, "assets", "fonts")
    family = "Outfit"
    if os.path.isdir(fonts_dir):
        for fname in ("Outfit-Regular.ttf", "Outfit-Medium.ttf", "Outfit-SemiBold.ttf", "Outfit-Bold.ttf"):
            fpath = os.path.join(fonts_dir, fname)
            if os.path.isfile(fpath):
                fid = QFontDatabase.addApplicationFont(fpath)
                if fid >= 0:
                    fams = QFontDatabase.applicationFontFamilies(fid)
                    if fams:
                        family = fams[0]
    return family


class SubprocessWorker(QThread):
    line_out = Signal(str)
    finished_code = Signal(int)

    def __init__(self, args: list[str], cwd: str):
        super().__init__()
        self.args = args
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._stop = False

    def stop(self) -> None:
        self._stop = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def run(self) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._proc = subprocess.Popen(
                self.args,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            self.line_out.emit(f"ERROR: failed to start process: {exc}")
            self.finished_code.emit(1)
            return

        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self.line_out.emit(line.rstrip("\r\n"))
            if self._stop:
                break
        rc = self._proc.wait()
        self.finished_code.emit(rc)


class ValidationWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(dict)

    def __init__(self, mode: str, root_dir: str, config: BuildConfig, full_sha256: bool = False):
        super().__init__()
        self.mode = mode
        self.root_dir = root_dir
        self.config = config
        self.full_sha256 = full_sha256

    def run(self) -> None:
        tools_dir = os.path.join(self.root_dir, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import validate  # noqa: E402

        if self.mode == "env":
            rep = validate.check_environment(
                stages=self.config.stages,
                work_dir=self.config.work_dir or self.config.out_dir,
                wav=self.config.wav,
                root_dir=self.root_dir,
            )
        elif self.mode == "game":
            rep = validate.validate_game_folder(
                self.config.game_dir,
                full_sha256=self.full_sha256,
                jobs=self.config.jobs,
                root_dir=self.root_dir,
                progress_cb=lambda cur, tot, name: self.progress.emit(cur, tot, name),
            )
        else:
            rep = validate.validate_codebase(root_dir=self.root_dir)
        self.completed.emit(rep.to_dict())


class LeonidaWindow(QMainWindow):
    def __init__(self, repo_root: str, font_family: str):
        super().__init__()
        self.repo_root = repo_root
        self.font_family = font_family
        self.config = load_config()
        ensure_output_dir(self.config.out_dir)
        self.lang = self.config.resolved_language()
        self.build_worker: SubprocessWorker | None = None
        self.val_worker: ValidationWorker | None = None
        self.viewer_proc: subprocess.Popen | None = None
        self.last_reports: dict[str, dict] = {}
        self.nav_buttons: list[QPushButton] = []

        self.setWindowTitle(f"{APP_NAME} — Builder & Validator v{APP_VERSION}")
        self.resize(1120, 740)
        self.setMinimumSize(940, 620)
        logo_path = os.path.join(self.repo_root, "assets", "logo.png")
        if os.path.isfile(logo_path):
            self.setWindowIcon(QIcon(logo_path))

        root = QWidget()
        root.setObjectName("RootWindow")
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(238)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(18, 22, 18, 18)
        sb_layout.setSpacing(8)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        logo_lbl = QLabel()
        if os.path.isfile(logo_path):
            pix = QPixmap(logo_path).scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lbl.setPixmap(pix)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        title_lbl = QLabel(APP_NAME)
        title_lbl.setStyleSheet("font-size: 20px; font-weight: 700; color: #ffffff;")
        ver_lbl = QLabel(f"v{APP_VERSION} · {self.lang}")
        ver_lbl.setStyleSheet("font-size: 11px; color: #7e8aa8;")
        brand_text.addWidget(title_lbl)
        brand_text.addWidget(ver_lbl)
        brand_row.addWidget(logo_lbl)
        brand_row.addLayout(brand_text)
        brand_row.addStretch()
        sb_layout.addLayout(brand_row)
        sb_layout.addSpacing(16)

        nav_keys = ["nav_build", "nav_validate", "nav_options", "nav_logs", "nav_about"]
        for idx, key in enumerate(nav_keys):
            btn = QPushButton(tr(self.lang, key))
            btn.setProperty("nav", True)
            btn.setProperty("active", idx == 0)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, i=idx: self.switch_page(i))
            self.nav_buttons.append(btn)
            sb_layout.addWidget(btn)

        sb_layout.addStretch()
        fs_btn = QPushButton("F11 · Fullscreen")
        fs_btn.setProperty("secondary", True)
        fs_btn.setCursor(Qt.PointingHandCursor)
        fs_btn.clicked.connect(self.toggle_fullscreen)
        sb_layout.addWidget(fs_btn)

        legal = QLabel(tr(self.lang, "legal_notice"))
        legal.setWordWrap(True)
        legal.setStyleSheet("font-size: 11px; color: #6f7c9b; padding: 8px; background: #0a0d16; border-radius: 8px;")
        sb_layout.addWidget(legal)

        # Stacked Pages
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_page())
        self.pages.addWidget(self._validate_page())
        self.pages.addWidget(self._options_page())
        self.pages.addWidget(self._logs_page())
        self.pages.addWidget(self._about_page())

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.pages, 1)

        self._refresh_disk_label()
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, activated=self._exit_fullscreen)
        QTimer.singleShot(250, self.open_setup_wizard)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def open_setup_wizard(self) -> None:
        dlg = SetupDialog(
            self.repo_root,
            self.lang,
            work_dir=self.work_edit.text().strip() or self.out_edit.text().strip(),
            parent=self,
        )
        dlg.exec()
        if dlg.last_report:
            self.last_reports["environment"] = dlg.last_report

    def switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setProperty("active", i == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _card(self) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setProperty("card", True)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        return frame, lay

    def _build_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(16)

        h = QLabel(f"{APP_NAME} — {tr(self.lang, 'nav_build')}")
        h.setProperty("heading", True)
        sub = QLabel(tr(self.lang, "subtitle"))
        sub.setProperty("subheading", True)
        lay.addWidget(h)
        lay.addWidget(sub)

        paths_card, pc = self._card()
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.game_edit = QLineEdit(self.config.game_dir)
        self.game_edit.setPlaceholderText("D:\\Games\\GTA_VI_Xbox_1.0.2.1")
        btn_find = QPushButton(tr(self.lang, "auto_find"))
        btn_find.setProperty("secondary", True)
        btn_find.clicked.connect(self.auto_find_game)
        btn_game = QPushButton(tr(self.lang, "browse"))
        btn_game.setProperty("secondary", True)
        btn_game.clicked.connect(lambda: self._pick_dir(self.game_edit))

        self.out_edit = QLineEdit(self.config.out_dir)
        self.out_edit.textChanged.connect(self._refresh_disk_label)
        btn_out = QPushButton(tr(self.lang, "browse"))
        btn_out.setProperty("secondary", True)
        btn_out.clicked.connect(lambda: self._pick_dir(self.out_edit))

        self.work_edit = QLineEdit(self.config.work_dir)
        self.work_edit.setPlaceholderText("<out>/work")
        self.work_edit.textChanged.connect(self._refresh_disk_label)
        btn_work = QPushButton(tr(self.lang, "browse"))
        btn_work.setProperty("secondary", True)
        btn_work.clicked.connect(lambda: self._pick_dir(self.work_edit))

        grid.addWidget(QLabel(tr(self.lang, "game_folder")), 0, 0)
        grid.addWidget(self.game_edit, 0, 1)
        grid.addWidget(btn_find, 0, 2)
        grid.addWidget(btn_game, 0, 3)

        grid.addWidget(QLabel(tr(self.lang, "out_folder")), 1, 0)
        grid.addWidget(self.out_edit, 1, 1, 1, 2)
        grid.addWidget(btn_out, 1, 3)

        grid.addWidget(QLabel(tr(self.lang, "work_folder")), 2, 0)
        grid.addWidget(self.work_edit, 2, 1, 1, 2)
        grid.addWidget(btn_work, 2, 3)
        grid.setColumnStretch(1, 1)

        pc.addLayout(grid)
        self.disk_label = QLabel("")
        self.disk_label.setStyleSheet("color: #00d4ff; font-size: 12px; font-weight: 500;")
        pc.addWidget(self.disk_label)
        lay.addWidget(paths_card)

        stages_card, sc = self._card()
        st_title = QLabel(tr(self.lang, "stages_title"))
        st_title.setProperty("section", True)
        sc.addWidget(st_title)

        st_row = QHBoxLayout()
        self.stage_checks: dict[str, QCheckBox] = {}
        stage_labels = {
            "check": "check",
            "shaders": "shaders (RDNA2 → SPIR-V)",
            "assets": "assets (OBJ / DDS / Audio)",
            "runtime": "runtime (PE Loader)",
            "pipelines": "pipelines (Vulkan Cache)",
        }
        for s in ALL_STAGES:
            cb = QCheckBox(stage_labels[s])
            cb.setChecked(s in self.config.stages)
            self.stage_checks[s] = cb
            st_row.addWidget(cb)
        sc.addLayout(st_row)
        lay.addWidget(stages_card)

        act_card, ac = self._card()
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton(tr(self.lang, "start_build"))
        self.start_btn.setProperty("primary", True)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_build)

        self.stop_btn = QPushButton(tr(self.lang, "stop_build"))
        self.stop_btn.setProperty("danger", True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_build)

        self.status_lbl = QLabel(tr(self.lang, "idle"))
        self.status_lbl.setStyleSheet("color: #9ba6c1; font-size: 13px; font-weight: 500;")

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addSpacing(12)
        btn_row.addWidget(self.status_lbl, 1)
        ac.addLayout(btn_row)

        self.build_progress = QProgressBar()
        self.build_progress.setRange(0, 100)
        self.build_progress.setValue(0)
        ac.addWidget(self.build_progress)

        self.mini_log = QPlainTextEdit()
        self.mini_log.setReadOnly(True)
        self.mini_log.setMinimumHeight(140)
        self.mini_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ac.addWidget(self.mini_log, 1)
        lay.addWidget(act_card, 1)
        return page

    def _validate_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)

        h = QLabel(tr(self.lang, "nav_validate"))
        h.setProperty("heading", True)
        lay.addWidget(h)

        bar = QHBoxLayout()
        btn_setup = QPushButton(tr(self.lang, "val_setup_btn"))
        btn_setup.setProperty("primary", True)
        btn_setup.setCursor(Qt.PointingHandCursor)
        btn_setup.clicked.connect(self.open_setup_wizard)
        bar.addWidget(btn_setup)

        for label, fn in (
            (tr(self.lang, "val_env_btn"), lambda: self.run_validation("env")),
            (tr(self.lang, "val_game_quick"), lambda: self.run_validation("game", False)),
            (tr(self.lang, "val_game_full"), lambda: self.run_validation("game", True)),
            (tr(self.lang, "val_code_btn"), lambda: self.run_validation("code")),
        ):
            b = QPushButton(label)
            b.setProperty("secondary", True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(fn)
            bar.addWidget(b)
        btn_exp = QPushButton(tr(self.lang, "export_report"))
        btn_exp.setProperty("secondary", True)
        btn_exp.clicked.connect(self.export_validation_report)
        bar.addStretch()
        bar.addWidget(btn_exp)
        lay.addLayout(bar)

        self.val_progress = QProgressBar()
        self.val_progress.setRange(0, 100)
        self.val_progress.setValue(0)
        lay.addWidget(self.val_progress)

        self.val_summary = QLabel(tr(self.lang, "val_prompt"))
        self.val_summary.setStyleSheet("font-size: 14px; font-weight: 600; color: #00d4ff;")
        lay.addWidget(self.val_summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.val_container = QWidget()
        self.val_list = QVBoxLayout(self.val_container)
        self.val_list.setSpacing(8)
        self.val_list.addStretch()
        scroll.setWidget(self.val_container)
        lay.addWidget(scroll, 1)
        return page

    def _options_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)

        h = QLabel(tr(self.lang, "nav_options"))
        h.setProperty("heading", True)
        lay.addWidget(h)

        card, c = self._card()
        grid = QGridLayout()
        grid.setVerticalSpacing(12)
        grid.setHorizontalSpacing(14)

        max_cpus = max(1, os.cpu_count() or 8)
        self.jobs_slider = QSlider(Qt.Horizontal)
        self.jobs_slider.setRange(1, max_cpus)
        self.jobs_slider.setValue(min(self.config.jobs, max_cpus))
        self.jobs_lbl = QLabel(f"{self.jobs_slider.value()} threads")
        self.jobs_slider.valueChanged.connect(lambda v: self.jobs_lbl.setText(f"{v} threads"))

        self.wave64_combo = QComboBox()
        self.wave64_combo.addItems(list(WAVE64_MODES))
        self.wave64_combo.setCurrentText(self.config.wave64_emulate)

        self.runtime_build_edit = QLineEdit(self.config.runtime_build)
        self.runtime_build_edit.setPlaceholderText("Optional: prebuilt runtime-windows folder")
        btn_rt = QPushButton(tr(self.lang, "browse"))
        btn_rt.setProperty("secondary", True)
        btn_rt.clicked.connect(lambda: self._pick_dir(self.runtime_build_edit))

        self.lang_combo = QComboBox()
        self.lang_combo.addItem(tr(self.lang, "lang_auto"), "")
        self.lang_combo.addItem("Português (Brasil)", "pt-BR")
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("Español", "es")
        idx = self.lang_combo.findData(self.config.language)
        self.lang_combo.setCurrentIndex(idx if idx >= 0 else 0)

        grid.addWidget(QLabel("CPU Jobs (--jobs)"), 0, 0)
        grid.addWidget(self.jobs_slider, 0, 1)
        grid.addWidget(self.jobs_lbl, 0, 2)

        grid.addWidget(QLabel("Wave64 Emulation"), 1, 0)
        grid.addWidget(self.wave64_combo, 1, 1, 1, 2)

        grid.addWidget(QLabel("Runtime Build Dir"), 2, 0)
        grid.addWidget(self.runtime_build_edit, 2, 1)
        grid.addWidget(btn_rt, 2, 2)

        grid.addWidget(QLabel(tr(self.lang, "lang_label")), 3, 0)
        grid.addWidget(self.lang_combo, 3, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        c.addLayout(grid)

        self.cb_full_check = QCheckBox("--full-check (SHA-256)")
        self.cb_full_check.setChecked(self.config.full_check)
        self.cb_wav = QCheckBox("--wav (Ogg Opus -> PCM WAV 16-bit)")
        self.cb_wav.setChecked(self.config.wav)
        self.cb_no_fc = QCheckBox("--no-float-controls")
        self.cb_no_fc.setChecked(self.config.no_float_controls)
        self.cb_bary = QCheckBox("--bary-fallback")
        self.cb_bary.setChecked(self.config.bary_fallback)
        self.cb_tex = QCheckBox("--texel-offset-fallback")
        self.cb_tex.setChecked(self.config.texel_offset_fallback)
        self.cb_f16 = QCheckBox("--f16-interp-fallback")
        self.cb_f16.setChecked(self.config.f16_interp_fallback)
        self.cb_mesh = QCheckBox("--mesh-compute")
        self.cb_mesh.setChecked(self.config.mesh_compute)

        for cb in (
            self.cb_full_check, self.cb_wav, self.cb_no_fc,
            self.cb_bary, self.cb_tex, self.cb_f16, self.cb_mesh,
        ):
            c.addWidget(cb)

        save_btn = QPushButton(tr(self.lang, "save_settings"))
        save_btn.setProperty("primary", True)
        save_btn.clicked.connect(self._save_current_config)
        c.addWidget(save_btn, alignment=Qt.AlignLeft)
        lay.addWidget(card)
        lay.addStretch()
        return page

    def _logs_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(12)

        top = QHBoxLayout()
        h = QLabel(tr(self.lang, "nav_logs"))
        h.setProperty("heading", True)
        top.addWidget(h)
        top.addStretch()

        for text, fn in (
            (tr(self.lang, "copy_logs"), self._copy_logs),
            (tr(self.lang, "save_logs"), self._save_logs),
            (tr(self.lang, "clear_logs"), self._clear_logs),
        ):
            b = QPushButton(text)
            b.setProperty("secondary", True)
            b.clicked.connect(fn)
            top.addWidget(b)
        lay.addLayout(top)

        self.full_log = QPlainTextEdit()
        self.full_log.setReadOnly(True)
        lay.addWidget(self.full_log, 1)
        return page

    def _about_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)

        h = QLabel(tr(self.lang, "nav_about"))
        h.setProperty("heading", True)
        lay.addWidget(h)

        card, c = self._card()
        info = QLabel(
            f"<b>{APP_NAME} Desktop v{APP_VERSION}</b><br><br>"
            f"GitHub: <a style='color:#00d4ff;' href='{REPO_URL}'>{REPO_URL}</a><br>"
            f"Website: <a style='color:#ff4d8d;' href='{SITE_URL}'>{SITE_URL}</a>"
        )
        info.setWordWrap(True)
        info.setOpenExternalLinks(True)
        c.addWidget(info)

        row = QHBoxLayout()
        btn_viewer = QPushButton(tr(self.lang, "open_viewer"))
        btn_viewer.setProperty("primary", True)
        btn_viewer.clicked.connect(self.launch_viewer)

        btn_site = QPushButton(tr(self.lang, "open_site"))
        btn_site.setProperty("secondary", True)
        btn_site.clicked.connect(lambda: webbrowser.open(SITE_URL))

        btn_rel = QPushButton(tr(self.lang, "open_releases"))
        btn_rel.setProperty("secondary", True)
        btn_rel.clicked.connect(lambda: webbrowser.open(RELEASES_URL))

        row.addWidget(btn_viewer)
        row.addWidget(btn_site)
        row.addWidget(btn_rel)
        row.addStretch()
        c.addLayout(row)
        lay.addWidget(card)
        lay.addStretch()
        return page

    def _pick_dir(self, target: QLineEdit) -> None:
        chosen = QFileDialog.getExistingDirectory(self, APP_NAME, target.text() or os.path.expanduser("~"))
        if chosen:
            target.setText(os.path.normpath(chosen))

    def _refresh_disk_label(self) -> None:
        p = self.work_edit.text().strip() or self.out_edit.text().strip()
        self.disk_label.setText(f"{tr(self.lang, 'disk_avail')} {format_free_space(p, self.lang)}")

    def _collect_config(self) -> BuildConfig:
        stages = [s for s, cb in self.stage_checks.items() if cb.isChecked()]
        return BuildConfig(
            game_dir=self.game_edit.text(),
            out_dir=self.out_edit.text(),
            work_dir=self.work_edit.text(),
            stages=stages,
            full_check=self.cb_full_check.isChecked(),
            wav=self.cb_wav.isChecked(),
            jobs=self.jobs_slider.value(),
            wave64_emulate=self.wave64_combo.currentText(),
            runtime_build=self.runtime_build_edit.text(),
            no_float_controls=self.cb_no_fc.isChecked(),
            bary_fallback=self.cb_bary.isChecked(),
            texel_offset_fallback=self.cb_tex.isChecked(),
            f16_interp_fallback=self.cb_f16.isChecked(),
            mesh_compute=self.cb_mesh.isChecked(),
            language=str(self.lang_combo.currentData() or ""),
        ).normalized()

    def _save_current_config(self) -> None:
        self.config = self._collect_config()
        path = save_config(self.config)
        self.append_log(f"{tr(self.lang, 'settings_saved')} {path}")
        QMessageBox.information(self, APP_NAME, f"{tr(self.lang, 'settings_saved')}\n{path}")

    def append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        self.mini_log.appendPlainText(line)
        self.full_log.appendPlainText(line)

    def auto_find_game(self) -> None:
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        import make_game  # noqa: E402

        found = make_game.find_dumps()
        if not found:
            QMessageBox.warning(self, APP_NAME, tr(self.lang, "no_game_found"))
            return
        self.game_edit.setText(found[0][0])
        self.append_log(f"Found: {found[0][1]['title']} {found[0][1]['version']} in {found[0][0]}")

    def start_build(self) -> None:
        if self.build_worker and self.build_worker.isRunning():
            return
        cfg = self._collect_config()
        save_config(cfg)
        py = locate_python()
        if not py:
            QMessageBox.warning(self, APP_NAME, tr(self.lang, "py_missing"))
            self.open_setup_wizard()
            return
        args = build_make_game_args(cfg, repo_root=self.repo_root, python_exe=py)
        self.append_log("$ " + " ".join(args))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText(tr(self.lang, "running"))
        self.build_progress.setRange(0, 0)

        self.build_worker = SubprocessWorker(args, cwd=self.repo_root)
        self.build_worker.line_out.connect(self.append_log)
        self.build_worker.finished_code.connect(self._on_build_finished)
        self.build_worker.start()

    def stop_build(self) -> None:
        if self.build_worker and self.build_worker.isRunning():
            self.build_worker.stop()
            self.append_log("Build cancelled.")

    def _on_build_finished(self, code: int) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.build_progress.setRange(0, 100)
        self.build_progress.setValue(100 if code == 0 else 0)
        self.status_lbl.setText(
            tr(self.lang, "finished_ok") if code == 0 else f"{tr(self.lang, 'finished_err')} (exit {code})"
        )

    def run_validation(self, mode: str, full_sha256: bool = False) -> None:
        if self.val_worker and self.val_worker.isRunning():
            return
        cfg = self._collect_config()
        if mode == "game" and not cfg.game_dir:
            QMessageBox.warning(self, APP_NAME, tr(self.lang, "select_game_first"))
            return
        self.val_summary.setText(tr(self.lang, "val_running"))
        self.val_progress.setRange(0, 0 if mode != "game" else 100)
        self.val_progress.setValue(0)

        self.val_worker = ValidationWorker(mode, self.repo_root, cfg, full_sha256=full_sha256)
        self.val_worker.progress.connect(self._on_val_progress)
        self.val_worker.completed.connect(self._on_val_completed)
        self.val_worker.start()

    def _on_val_progress(self, current: int, total: int, label: str) -> None:
        self.val_progress.setRange(0, max(1, total))
        self.val_progress.setValue(current)
        self.val_summary.setText(f"({current}/{total}): {label}")

    def _on_val_completed(self, report: dict) -> None:
        self.val_progress.setRange(0, 100)
        self.val_progress.setValue(100)
        self.last_reports[report["kind"]] = report
        ok = report.get("ok", False)
        color = "#22c55e" if ok else "#f43f5e"
        self.val_summary.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {color};")
        self.val_summary.setText(f"[{'OK' if ok else 'FAIL'}] {report.get('summary', '')}")
        self.append_log(f"Validation ({report['kind']}): {report.get('summary', '')}")

        while self.val_list.count():
            item = self.val_list.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        badge_colors = {"ok": "#22c55e", "warn": "#f59e0b", "error": "#f43f5e", "skip": "#64748b"}
        for chk in report.get("checks", []):
            row_frame, rl = self._card()
            rl.setContentsMargins(14, 10, 14, 10)
            rl.setSpacing(8)
            row_frame.setCursor(Qt.PointingHandCursor)

            hrow = QHBoxLayout()
            st = chk.get("status", "skip")
            badge = QLabel(st.upper())
            badge.setFixedWidth(64)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background-color: {badge_colors.get(st, '#64748b')}; color: #090b11; "
                "border-radius: 6px; padding: 3px 6px; font-weight: 700; font-size: 11px;"
            )
            name = QLabel(f"<b>{chk.get('name', '')}</b> — {chk.get('detail', '')}")
            name.setWordWrap(True)
            hint = QLabel(tr(self.lang, "val_click_hint"))
            hint.setStyleSheet("color: #7e8aa8; font-size: 11px; font-weight: 500;")
            hrow.addWidget(badge)
            hrow.addWidget(name, 1)
            hrow.addWidget(hint)
            rl.addLayout(hrow)

            detail_box = QPlainTextEdit()
            detail_box.setReadOnly(True)
            detail_box.setPlainText(format_validation_check_details(report, chk, self.lang))
            detail_box.setMinimumHeight(140)
            detail_box.setMaximumHeight(280)
            detail_box.setVisible(False)
            rl.addWidget(detail_box)

            def _toggle_details(_event, box=detail_box) -> None:
                box.setVisible(not box.isVisible())

            row_frame.mousePressEvent = _toggle_details
            name.mousePressEvent = _toggle_details
            badge.mousePressEvent = _toggle_details
            hint.mousePressEvent = _toggle_details
            self.val_list.addWidget(row_frame)
        self.val_list.addStretch()

    def export_validation_report(self) -> None:
        if not self.last_reports:
            return
        path, _ = QFileDialog.getSaveFileName(self, APP_NAME, "leonida_validation.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.last_reports, f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, APP_NAME, path)

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self.full_log.toPlainText())

    def _clear_logs(self) -> None:
        self.full_log.clear()
        self.mini_log.clear()

    def _save_logs(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, APP_NAME, "leonida_build.log", "Log (*.log);;Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(self.full_log.toPlainText())

    def launch_viewer(self) -> None:
        py = locate_python()
        serve = os.path.join(self.repo_root, "viewer", "serve.py")
        if not py or not os.path.isfile(serve):
            webbrowser.open(SITE_URL)
            return
        cmd = [py, "-I", serve]
        work = self.work_edit.text().strip() or os.path.join(self.out_edit.text().strip(), "work")
        world_dir = os.path.join(work, "world")
        if os.path.isfile(os.path.join(world_dir, "world.json")):
            cmd.extend(["--world", world_dir])
        if self.viewer_proc is None or self.viewer_proc.poll() is not None:
            self.viewer_proc = subprocess.Popen(cmd, cwd=self.repo_root)
            time.sleep(0.6)
        webbrowser.open("http://127.0.0.1:8765/")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.build_worker and self.build_worker.isRunning():
            self.build_worker.stop()
            self.build_worker.wait(2000)
        if self.viewer_proc and self.viewer_proc.poll() is None:
            self.viewer_proc.terminate()
        super().closeEvent(event)


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    repo_root = resolve_repo_root()
    logo_path = os.path.join(repo_root, "assets", "logo.png")
    if os.path.isfile(logo_path):
        app.setWindowIcon(QIcon(logo_path))
    font_family = load_outfit_fonts(repo_root)
    app.setFont(QFont(font_family, 10))
    app.setStyleSheet(STYLESHEET)
    win = LeonidaWindow(repo_root, font_family)
    win.show()
    return app.exec()
