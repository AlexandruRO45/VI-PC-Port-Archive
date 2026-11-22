"""Startup Environment Verification & Auto-Installer Dialog (PySide6 + Outfit)."""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui import APP_NAME
from gui.core import InstallItem, pending_install_plan, tr


class EnvCheckWorker(QThread):
    completed = Signal(dict)

    def __init__(self, root_dir: str, work_dir: str | None = None):
        super().__init__()
        self.root_dir = root_dir
        self.work_dir = work_dir

    def run(self) -> None:
        tools_dir = os.path.join(self.root_dir, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import validate  # noqa: E402

        rep = validate.check_environment(work_dir=self.work_dir, root_dir=self.root_dir)
        self.completed.emit(rep.to_dict())


class InstallWorker(QThread):
    log_line = Signal(str)
    progress = Signal(int, int)
    finished_all = Signal(bool)

    def __init__(self, items: list[InstallItem]):
        super().__init__()
        self.items = items

    def run(self) -> None:
        all_ok = True
        total = max(1, len(self.items))
        for idx, item in enumerate(self.items, 1):
            self.progress.emit(idx - 1, total)
            if not item.command:
                self.log_line.emit(f"[SKIP] {item.name}: manual install -> {item.download_url}")
                continue
            self.log_line.emit(f"=== Installing {item.name} ===")
            self.log_line.emit("$ " + " ".join(item.command))
            try:
                proc = subprocess.Popen(
                    item.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_line.emit(line.rstrip("\r\n"))
                rc = proc.wait()
                if rc != 0:
                    all_ok = False
                    self.log_line.emit(f"[ERROR] {item.name} exited with code {rc}")
                else:
                    self.log_line.emit(f"[OK] {item.name} installed.")
            except OSError as exc:
                all_ok = False
                self.log_line.emit(f"[ERROR] Could not launch installer for {item.name}: {exc}")
            self.progress.emit(idx, total)
        self.finished_all.emit(all_ok)


class SetupDialog(QDialog):
    """Checks environment prerequisites on launch and installs only what is pending."""

    def __init__(self, repo_root: str, lang: str, work_dir: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.repo_root = repo_root
        self.lang = lang
        self.work_dir = work_dir
        self.check_worker: EnvCheckWorker | None = None
        self.install_worker: InstallWorker | None = None
        self.pending_checkboxes: list[tuple[QCheckBox, InstallItem]] = []
        self.last_report: dict | None = None

        self.setWindowTitle(f"{APP_NAME} — {tr(lang, 'setup_title')}")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(820, 640)
        self.setMinimumSize(640, 500)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 20)
        lay.setSpacing(14)

        title = QLabel(tr(lang, "setup_title"))
        title.setProperty("heading", True)
        sub = QLabel(tr(lang, "setup_sub"))
        sub.setProperty("subheading", True)
        sub.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(sub)

        self.banner = QLabel(tr(lang, "val_running"))
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("font-size: 14px; font-weight: 600; color: #00d4ff;")
        lay.addWidget(self.banner)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        lay.addWidget(self.progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setSpacing(8)
        self.list_layout.addStretch()
        scroll.setWidget(self.list_container)
        lay.addWidget(scroll, 1)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(110)
        self.log_box.setVisible(False)
        lay.addWidget(self.log_box, 1)

        btn_row = QHBoxLayout()
        self.btn_install = QPushButton(tr(lang, "setup_install_btn"))
        self.btn_install.setProperty("primary", True)
        self.btn_install.setCursor(Qt.PointingHandCursor)
        self.btn_install.setEnabled(False)
        self.btn_install.clicked.connect(self.install_pending)

        self.btn_recheck = QPushButton(tr(lang, "setup_recheck_btn"))
        self.btn_recheck.setProperty("secondary", True)
        self.btn_recheck.clicked.connect(self.run_check)

        self.btn_continue = QPushButton(tr(lang, "setup_continue_btn"))
        self.btn_continue.setProperty("secondary", True)
        self.btn_continue.clicked.connect(self.accept)

        btn_row.addWidget(self.btn_install)
        btn_row.addWidget(self.btn_recheck)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_continue)
        lay.addLayout(btn_row)

        self.run_check()

    def run_check(self) -> None:
        if self.check_worker and self.check_worker.isRunning():
            return
        self.btn_install.setEnabled(False)
        self.btn_recheck.setEnabled(False)
        self.progress.setRange(0, 0)
        self.banner.setStyleSheet("font-size: 14px; font-weight: 600; color: #00d4ff;")
        self.banner.setText(tr(self.lang, "val_running"))

        self.check_worker = EnvCheckWorker(self.repo_root, self.work_dir)
        self.check_worker.completed.connect(self._on_check_done)
        self.check_worker.start()

    def _clear_rows(self) -> None:
        self.pending_checkboxes.clear()
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _on_check_done(self, report: dict) -> None:
        self.last_report = report
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.btn_recheck.setEnabled(True)
        self._clear_rows()

        ok = bool(report.get("ok"))
        pending = pending_install_plan(report)
        pending_by_key = {p.package_key: p for p in pending}

        if ok:
            self.banner.setStyleSheet("font-size: 14px; font-weight: 600; color: #22c55e;")
            self.banner.setText(tr(self.lang, "setup_all_ok"))
        else:
            self.banner.setStyleSheet("font-size: 14px; font-weight: 600; color: #f59e0b;")
            self.banner.setText(tr(self.lang, "setup_missing"))

        badge_colors = {"ok": "#22c55e", "warn": "#f59e0b", "error": "#f43f5e", "skip": "#64748b"}

        # First show pending actionable items with checkboxes + download links
        for item in pending:
            frame = QFrame()
            frame.setProperty("card", True)
            rl = QHBoxLayout(frame)
            rl.setContentsMargins(14, 10, 14, 10)
            rl.setSpacing(10)

            cb = QCheckBox()
            cb.setChecked(item.required and not item.manual_only)
            cb.setEnabled(not item.manual_only)
            self.pending_checkboxes.append((cb, item))

            badge = QLabel("ERROR" if item.status == "error" else "WARN")
            badge.setFixedWidth(62)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background-color: {badge_colors.get(item.status, '#f59e0b')}; color: #090b11; "
                "border-radius: 6px; padding: 3px 6px; font-weight: 700; font-size: 11px;"
            )

            info = QLabel(f"<b>{item.name}</b><br><span style='color:#93a0be;font-size:12px;'>{item.detail}</span>")
            info.setWordWrap(True)

            dl_btn = QPushButton(tr(self.lang, "setup_download_btn"))
            dl_btn.setProperty("secondary", True)
            dl_btn.setCursor(Qt.PointingHandCursor)
            dl_btn.clicked.connect(lambda _=False, url=item.download_url: webbrowser.open(url))

            rl.addWidget(cb)
            rl.addWidget(badge)
            rl.addWidget(info, 1)
            if item.manual_only:
                m_lbl = QLabel(tr(self.lang, "setup_manual_tag"))
                m_lbl.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: 600;")
                rl.addWidget(m_lbl)
            rl.addWidget(dl_btn)
            self.list_layout.addWidget(frame)

        # Then show already-installed OK items so the user sees what is already ready
        for chk in report.get("checks", []):
            if chk.get("status") != "ok":
                continue
            frame = QFrame()
            frame.setProperty("card", True)
            rl = QHBoxLayout(frame)
            rl.setContentsMargins(14, 9, 14, 9)
            rl.setSpacing(10)

            badge = QLabel("OK")
            badge.setFixedWidth(62)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                "background-color: #22c55e; color: #090b11; "
                "border-radius: 6px; padding: 3px 6px; font-weight: 700; font-size: 11px;"
            )
            info = QLabel(f"<b>{chk.get('name', '')}</b> — <span style='color:#93a0be;'>{chk.get('detail', '')}</span>")
            info.setWordWrap(True)
            rl.addWidget(badge)
            rl.addWidget(info, 1)
            self.list_layout.addWidget(frame)

        self.list_layout.addStretch()
        has_auto = any(not it.manual_only for it in pending_by_key.values())
        self.btn_install.setEnabled(has_auto)

    def install_pending(self) -> None:
        selected = [item for cb, item in self.pending_checkboxes if cb.isChecked() and not item.manual_only]
        if not selected:
            self.banner.setText(tr(self.lang, "setup_no_pending"))
            return

        self.log_box.setVisible(True)
        self.log_box.clear()
        self.btn_install.setEnabled(False)
        self.btn_recheck.setEnabled(False)
        self.btn_continue.setEnabled(False)
        self.banner.setStyleSheet("font-size: 14px; font-weight: 600; color: #00d4ff;")
        self.banner.setText(tr(self.lang, "setup_installing"))
        self.progress.setRange(0, len(selected))
        self.progress.setValue(0)

        self.install_worker = InstallWorker(selected)
        self.install_worker.log_line.connect(self.log_box.appendPlainText)
        self.install_worker.progress.connect(lambda cur, tot: (self.progress.setRange(0, tot), self.progress.setValue(cur)))
        self.install_worker.finished_all.connect(self._on_install_done)
        self.install_worker.start()

    def _on_install_done(self, _all_ok: bool) -> None:
        self.btn_continue.setEnabled(True)
        self.banner.setText(tr(self.lang, "setup_done"))
        self.run_check()
