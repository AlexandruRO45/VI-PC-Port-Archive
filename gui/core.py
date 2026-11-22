"""Pure standard-library core logic for the Leonida GUI (config, CLI args, paths, i18n).

Kept free of Qt imports so unit tests can exercise every rule with `python -I`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

ALL_STAGES = ("check", "shaders", "assets", "runtime", "pipelines")
WAVE64_MODES = ("auto", "on", "off")


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def default_out_dir() -> str:
    if sys.platform == "darwin" and os.path.isdir(
        os.path.join(os.path.expanduser("~"), "Library", "Mobile Documents", "com~apple~CloudDocs", "Documents")
    ):
        return os.path.join(os.path.expanduser("~"), "Leonida")
    return os.path.join(os.path.expanduser("~"), "Documents", "Leonida")


def resolve_repo_root() -> str:
    """Locate the Leonida root directory both from source and from a PyInstaller frozen bundle."""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass and os.path.isdir(meipass):
            return meipass
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def locate_python() -> str | None:
    """Return a Python 3.12+ executable path for running make_game.py and tools/*.py."""
    if not getattr(sys, "frozen", False) and sys.version_info >= (3, 12):
        return sys.executable
    candidates = ["python3", "python"]
    if os.name == "nt":
        candidates = ["py", "python", "python3"]
    for name in candidates:
        path = shutil.which(name)
        if not path:
            continue
        cmd = [path, "-3.12"] if name == "py" else [path]
        try:
            r = subprocess.run(
                [*cmd, "-c", "import sys; print(sys.executable); sys.exit(0 if sys.version_info >= (3, 12) else 1)"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().splitlines()[-1]
        except (OSError, subprocess.SubprocessError):
            continue
    return None


@dataclass
class BuildConfig:
    game_dir: str = ""
    out_dir: str = field(default_factory=default_out_dir)
    work_dir: str = ""
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    full_check: bool = False
    wav: bool = False
    jobs: int = field(default_factory=default_jobs)
    wave64_emulate: str = "auto"
    runtime_build: str = ""
    no_float_controls: bool = False
    bary_fallback: bool = False
    texel_offset_fallback: bool = False
    f16_interp_fallback: bool = False
    mesh_compute: bool = False
    language: str = "pt-BR"

    def normalized(self) -> "BuildConfig":
        valid_stages = [s for s in ALL_STAGES if s in self.stages]
        if not valid_stages:
            valid_stages = ["check"]
        mode = self.wave64_emulate if self.wave64_emulate in WAVE64_MODES else "auto"
        lang = self.language if self.language in ("pt-BR", "en") else "pt-BR"
        return BuildConfig(
            game_dir=self.game_dir.strip(),
            out_dir=(self.out_dir.strip() or default_out_dir()),
            work_dir=self.work_dir.strip(),
            stages=valid_stages,
            full_check=bool(self.full_check),
            wav=bool(self.wav),
            jobs=max(1, int(self.jobs or 1)),
            wave64_emulate=mode,
            runtime_build=self.runtime_build.strip(),
            no_float_controls=bool(self.no_float_controls),
            bary_fallback=bool(self.bary_fallback),
            texel_offset_fallback=bool(self.texel_offset_fallback),
            f16_interp_fallback=bool(self.f16_interp_fallback),
            mesh_compute=bool(self.mesh_compute),
            language=lang,
        )


def build_make_game_args(config: BuildConfig, repo_root: str | None = None, python_exe: str | None = None) -> list[str]:
    """Build the exact subprocess command list for make_game.py from a BuildConfig."""
    cfg = config.normalized()
    root = repo_root or resolve_repo_root()
    py = python_exe or sys.executable
    args = [py, "-I", os.path.join(root, "make_game.py")]
    if cfg.game_dir:
        args.extend(["--game", cfg.game_dir])
    if cfg.out_dir:
        args.extend(["--out", cfg.out_dir])
    if cfg.work_dir:
        args.extend(["--work", cfg.work_dir])
    if cfg.stages:
        args.extend(["--stages", ",".join(cfg.stages)])
    if cfg.full_check:
        args.append("--full-check")
    if cfg.wav:
        args.append("--wav")
    args.extend(["--jobs", str(cfg.jobs)])
    args.extend(["--wave64-emulate", cfg.wave64_emulate])
    if cfg.runtime_build:
        args.extend(["--runtime-build", cfg.runtime_build])
    if cfg.no_float_controls:
        args.append("--no-float-controls")
    if cfg.bary_fallback:
        args.append("--bary-fallback")
    if cfg.texel_offset_fallback:
        args.append("--texel-offset-fallback")
    if cfg.f16_interp_fallback:
        args.append("--f16-interp-fallback")
    if cfg.mesh_compute:
        args.append("--mesh-compute")
    return args


def config_file_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".leonida", "gui_config.json")


def save_config(config: BuildConfig, path: str | None = None) -> str:
    target = path or config_file_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        json.dump(asdict(config.normalized()), f, indent=2, ensure_ascii=False)
    return target


def load_config(path: str | None = None) -> BuildConfig:
    target = path or config_file_path()
    if not os.path.isfile(target):
        return BuildConfig()
    try:
        with open(target, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return BuildConfig()
        allowed = {k: raw[k] for k in BuildConfig.__dataclass_fields__ if k in raw}
        return BuildConfig(**allowed).normalized()
    except (OSError, ValueError, TypeError):
        return BuildConfig()


def format_free_space(path: str) -> str:
    target = os.path.abspath(path or default_out_dir())
    while target and not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    try:
        free_gb = shutil.disk_usage(target).free / (2**30)
        return f"{free_gb:.1f} GB livres"
    except OSError:
        return "—"


TRANSLATIONS: dict[str, dict[str, str]] = {
    "pt-BR": {
        "nav_build": "Compilar",
        "nav_validate": "Validação",
        "nav_options": "Opções",
        "nav_logs": "Console",
        "nav_about": "Mapa & Sobre",
        "subtitle": "Construtor e Validador Nativo para PC",
        "game_folder": "Pasta do Jogo (Xbox Series 1.0.2.1)",
        "out_folder": "Pasta de Saída (--out)",
        "work_folder": "Pasta de Trabalho (--work, opcional)",
        "auto_find": "Detectar Automaticamente",
        "browse": "Procurar...",
        "stages_title": "Estágios do Build",
        "start_build": "Iniciar Build",
        "stop_build": "Cancelar Build",
        "idle": "Pronto para iniciar",
        "running": "Executando...",
        "finished_ok": "Concluído com sucesso!",
        "finished_err": "Finalizado com erro",
        "val_env_btn": "Validar Ambiente",
        "val_game_quick": "Validar Jogo (Rápido)",
        "val_game_full": "Validar Jogo (SHA-256)",
        "val_code_btn": "Validar Código",
        "export_report": "Exportar Relatório JSON",
        "clear_logs": "Limpar",
        "copy_logs": "Copiar",
        "save_logs": "Salvar Log...",
        "open_viewer": "Abrir Visualizador de Mapa 3D",
        "open_site": "Abrir Site Oficial",
        "open_releases": "Ver Releases no GitHub",
        "legal_notice": (
            "Aviso: o Leonida não contém arquivos do jogo, chaves nem executáveis. "
            "Tudo é construído localmente a partir da sua própria cópia."
        ),
    },
    "en": {
        "nav_build": "Build",
        "nav_validate": "Validation",
        "nav_options": "Options",
        "nav_logs": "Console",
        "nav_about": "Map & About",
        "subtitle": "Native PC Port Builder & Validator",
        "game_folder": "Game Folder (Xbox Series 1.0.2.1)",
        "out_folder": "Output Folder (--out)",
        "work_folder": "Work Folder (--work, optional)",
        "auto_find": "Auto-Detect",
        "browse": "Browse...",
        "stages_title": "Build Stages",
        "start_build": "Start Build",
        "stop_build": "Stop Build",
        "idle": "Ready",
        "running": "Running...",
        "finished_ok": "Finished successfully!",
        "finished_err": "Finished with errors",
        "val_env_btn": "Validate Environment",
        "val_game_quick": "Validate Game (Quick)",
        "val_game_full": "Validate Game (SHA-256)",
        "val_code_btn": "Validate Codebase",
        "export_report": "Export JSON Report",
        "clear_logs": "Clear",
        "copy_logs": "Copy",
        "save_logs": "Save Log...",
        "open_viewer": "Launch 3D Map Viewer",
        "open_site": "Open Official Website",
        "open_releases": "View GitHub Releases",
        "legal_notice": (
            "Notice: Leonida contains no game files, keys, or executables. "
            "Everything is built locally on your PC from your own copy."
        ),
    },
}


def tr(lang: str, key: str) -> str:
    table = TRANSLATIONS.get(lang, TRANSLATIONS["pt-BR"])
    return table.get(key, key)
