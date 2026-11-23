"""Pure standard-library core logic for the Leonida GUI (config, CLI args, i18n, auto-installer plan).

Kept free of Qt imports so unit tests can exercise every rule with `python -I`.
"""
from __future__ import annotations

import json
import locale
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

ALL_STAGES = ("check", "shaders", "assets", "runtime", "pipelines")
WAVE64_MODES = ("auto", "on", "off")
SUPPORTED_LANGUAGES = ("pt-BR", "en", "es")


def detect_system_language(env: dict[str, str] | None = None) -> str:
    """Detect the OS UI language and map it to 'pt-BR', 'es', or 'en' (default)."""
    # 1. Windows UI language via GetUserDefaultUILanguage()
    if env is None and os.name == "nt":
        try:
            import ctypes

            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            primary = lang_id & 0x3FF
            if primary == 0x16:  # LANG_PORTUGUESE
                return "pt-BR"
            if primary == 0x0A:  # LANG_SPANISH
                return "es"
            if primary == 0x09:  # LANG_ENGLISH
                return "en"
        except Exception:
            pass

    # 2. POSIX / environment variables
    src_env = os.environ if env is None else env
    candidates = [
        src_env.get("LEONIDA_LANG", ""),
        src_env.get("LC_ALL", ""),
        src_env.get("LC_MESSAGES", ""),
        src_env.get("LANG", ""),
    ]
    if env is None:
        try:
            loc = locale.getlocale()[0] or ""
            if loc:
                candidates.append(loc)
        except Exception:
            pass

    for raw in candidates:
        low = (raw or "").strip().lower().replace("-", "_")
        if not low or low in ("c", "posix"):
            continue
        if low.startswith("pt"):
            return "pt-BR"
        if low.startswith("es"):
            return "es"
        if low.startswith("en"):
            return "en"
    return "en"


def is_windows_admin() -> bool:
    """True when running elevated on Windows (or True on non-Windows)."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


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
    candidates = ["py", "python", "python3"] if os.name == "nt" else ["python3", "python"]
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
class InstallItem:
    package_key: str
    name: str
    required: bool
    status: str
    detail: str
    download_url: str
    command: list[str] | None = None
    manual_only: bool = False


# Maps validate.py check IDs to package keys, official download URLs, and per-OS auto-install commands.
PREREQ_SPECS: dict[str, dict] = {
    "python": {
        "package_key": "python",
        "name": "Python 3.12+",
        "download_url": "https://www.python.org/downloads/",
        "win32": ["winget", "install", "--id", "Python.Python.3.12", "-e", "--accept-package-agreements", "--accept-source-agreements"],
        "darwin": ["brew", "install", "python@3.12"],
        "linux": ["pkexec", "apt-get", "install", "-y", "python3", "python3-venv"],
    },
    "dotnet": {
        "package_key": "dotnet",
        "name": ".NET 8 SDK",
        "download_url": "https://dotnet.microsoft.com/download/dotnet/8.0",
        "win32": ["winget", "install", "--id", "Microsoft.DotNet.SDK.8", "-e", "--accept-package-agreements", "--accept-source-agreements"],
        "darwin": ["brew", "install", "--cask", "dotnet-sdk"],
        "linux": ["pkexec", "apt-get", "install", "-y", "dotnet-sdk-8.0"],
    },
    "spirv_as": {
        "package_key": "vulkan_sdk",
        "name": "Vulkan SDK (spirv-as / spirv-val)",
        "download_url": "https://vulkan.lunarg.com/sdk/home",
        "win32": ["winget", "install", "--id", "KhronosGroup.VulkanSDK", "-e", "--accept-package-agreements", "--accept-source-agreements"],
        "darwin": None,
        "linux": ["pkexec", "apt-get", "install", "-y", "spirv-tools", "vulkan-tools"],
    },
    "spirv_val": {
        "package_key": "vulkan_sdk",
        "name": "Vulkan SDK (spirv-as / spirv-val)",
        "download_url": "https://vulkan.lunarg.com/sdk/home",
        "win32": ["winget", "install", "--id", "KhronosGroup.VulkanSDK", "-e", "--accept-package-agreements", "--accept-source-agreements"],
        "darwin": None,
        "linux": ["pkexec", "apt-get", "install", "-y", "spirv-tools", "vulkan-tools"],
    },
    "msvc": {
        "package_key": "msvc",
        "name": "Visual Studio 2022 C++ Build Tools",
        "download_url": "https://visualstudio.microsoft.com/visual-cpp-build-tools/",
        "win32": [
            "winget", "install", "--id", "Microsoft.VisualStudio.2022.BuildTools", "-e",
            "--accept-package-agreements", "--accept-source-agreements",
            "--override", "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended",
        ],
        "darwin": None,
        "linux": None,
    },
    "ffmpeg": {
        "package_key": "ffmpeg",
        "name": "FFmpeg (XMA2 / WAV)",
        "download_url": "https://ffmpeg.org/download.html",
        "win32": ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--accept-package-agreements", "--accept-source-agreements"],
        "darwin": ["brew", "install", "ffmpeg"],
        "linux": ["pkexec", "apt-get", "install", "-y", "ffmpeg"],
    },
    "llvm_mc": {
        "package_key": "llvm_mc",
        "name": "LLVM 21+ llvm-mc (AMDGPU)",
        "download_url": "https://github.com/llvm/llvm-project/releases",
        "win32": ["powershell", "-ExecutionPolicy", "Bypass", "-File", "{ROOT}/scripts/install_llvm.ps1"],
        "darwin": ["python3", "-I", "{ROOT}/scripts/install_llvm.py"],
        "linux": ["python3", "-I", "{ROOT}/scripts/install_llvm.py"],
    },
}


def pending_install_plan(env_report: dict, platform_name: str | None = None, repo_root: str | None = None) -> list[InstallItem]:
    """Return ONLY pending (non-ok, non-skip) installable items deduplicated by package_key."""
    plat = platform_name or ("win32" if sys.platform.startswith("win") else ("darwin" if sys.platform == "darwin" else "linux"))
    root = repo_root or resolve_repo_root()
    seen: set[str] = set()
    items: list[InstallItem] = []

    for chk in env_report.get("checks", []):
        st = chk.get("status", "ok")
        if st in ("ok", "skip"):
            continue
        cid = chk.get("id", "")
        spec = PREREQ_SPECS.get(cid)
        if not spec:
            continue
        pkg_key = spec["package_key"]
        if pkg_key in seen:
            continue
        seen.add(pkg_key)
        raw_cmd = spec.get(plat)
        cmd = [os.path.normpath(arg.replace("{ROOT}", root)) if "{ROOT}" in arg else arg for arg in raw_cmd] if raw_cmd else None
        items.append(InstallItem(
            package_key=pkg_key,
            name=spec["name"],
            required=bool(chk.get("required", True)),
            status=st,
            detail=chk.get("detail", ""),
            download_url=spec["download_url"],
            command=cmd,
            manual_only=(cmd is None),
        ))
    return items


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
    language: str = ""  # Empty means auto-detect OS language

    def resolved_language(self) -> str:
        if self.language in SUPPORTED_LANGUAGES:
            return self.language
        return detect_system_language()

    def normalized(self) -> "BuildConfig":
        valid_stages = [s for s in ALL_STAGES if s in self.stages]
        if not valid_stages:
            valid_stages = ["check"]
        mode = self.wave64_emulate if self.wave64_emulate in WAVE64_MODES else "auto"
        lang = self.language if self.language in SUPPORTED_LANGUAGES else ""
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


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "?"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{num_bytes} B"


def format_validation_check_details(report: dict, chk: dict, lang: str = "en") -> str:
    """Build a detailed, human-readable breakdown for a clicked validation card."""
    cid = chk.get("id", "")
    details = report.get("details") or {}
    lines: list[str] = [f"{chk.get('name', '')} [{chk.get('status', '').upper()}]", chk.get("detail", ""), ""]

    if cid == "integrity" and isinstance(details.get("issues"), list):
        issues = details["issues"]
        missing = [i for i in issues if i.get("kind") == "MISSING"]
        wrong_size = [i for i in issues if i.get("kind") == "SIZE"]
        changed = [i for i in issues if i.get("kind") == "CHANGED"]

        if not issues:
            lines.append(tr(lang, "val_all_files_ok"))
            return "\n".join(lines).strip()

        if missing:
            lines.append(f"=== {tr(lang, 'val_missing_hdr')} ({len(missing)}) ===")
            for item in missing:
                exp = format_bytes(item.get("expected_size"))
                lines.append(f"  - {item.get('path', '')}  ({tr(lang, 'val_expected')}: {exp})")
            lines.append("")

        if wrong_size:
            lines.append(f"=== {tr(lang, 'val_incomplete_hdr')} ({len(wrong_size)}) ===")
            for item in wrong_size:
                act = item.get("actual_size")
                exp = item.get("expected_size")
                pct = f" ({(act / exp) * 100:.1f}%)" if (act is not None and exp) else ""
                lines.append(f"  - {item.get('path', '')}  ({format_bytes(act)} / {format_bytes(exp)}{pct})")
            lines.append("")

        if changed:
            lines.append(f"=== {tr(lang, 'val_changed_hdr')} ({len(changed)}) ===")
            for item in changed:
                lines.append(f"  - {item.get('path', '')}  (SHA-256 mismatch)")
            lines.append("")

        return "\n".join(lines).strip()

    if cid == "probes" and isinstance(details.get("probe_notes"), list):
        for note in details["probe_notes"]:
            lines.append(f"  - {note}")
        return "\n".join(lines).strip()

    return "\n".join(lines).strip()


def format_free_space(path: str, lang: str = "en") -> str:
    target = os.path.abspath(path or default_out_dir())
    while target and not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    try:
        free_gb = shutil.disk_usage(target).free / (2**30)
        suffix = {"pt-BR": "GB livres", "es": "GB libres"}.get(lang, "GB free")
        return f"{free_gb:.1f} {suffix}"
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
        "disk_avail": "Espaço em disco disponível:",
        "val_env_btn": "Validar Ambiente",
        "val_setup_btn": "Assistente de Instalação",
        "val_game_quick": "Validar Jogo (Rápido)",
        "val_game_full": "Validar Jogo (SHA-256)",
        "val_code_btn": "Validar Código",
        "export_report": "Exportar Relatório JSON",
        "val_prompt": "Clique em um botão acima para validar ambiente, jogo ou código.",
        "val_running": "Validando...",
        "val_click_hint": "Clique para ver detalhes ▾",
        "val_missing_hdr": "Arquivos ausentes (MISSING)",
        "val_incomplete_hdr": "Arquivos incompletos / tamanho diferente (SIZE)",
        "val_changed_hdr": "Arquivos corrompidos / SHA-256 diferente (CHANGED)",
        "val_expected": "esperado",
        "val_all_files_ok": "Todos os arquivos do manifesto estão presentes e íntegros.",
        "clear_logs": "Limpar",
        "copy_logs": "Copiar",
        "save_logs": "Salvar Log...",
        "save_settings": "Salvar Configurações",
        "settings_saved": "Configurações salvas em:",
        "lang_auto": "Automático (Sistema)",
        "lang_label": "Idioma",
        "open_viewer": "Abrir Visualizador de Mapa 3D",
        "open_site": "Abrir Site Oficial",
        "open_releases": "Ver Releases no GitHub",
        "py_missing": "Python 3.12+ não encontrado no PATH.\nUse o Assistente de Instalação na aba Validação para instalar os pré-requisitos pendentes.",
        "select_game_first": "Informe primeiro a pasta do jogo na aba Compilar.",
        "no_game_found": "Nenhuma cópia suportada (Xbox Series 1.0.2.1) foi encontrada automaticamente.\nSelecione a pasta que contém MicrosoftGame.Config e os arquivos .rpf.",
        "legal_notice": (
            "Aviso: o Leonida não contém arquivos do jogo, chaves nem executáveis. "
            "Tudo é construído localmente a partir da sua própria cópia."
        ),
        "setup_title": "Verificação do Ambiente",
        "setup_sub": "Verificando os pré-requisitos necessários no seu sistema. Somente itens pendentes serão instalados.",
        "setup_all_ok": "Todos os pré-requisitos obrigatórios estão instalados!",
        "setup_missing": "Alguns pré-requisitos estão pendentes. Instale automaticamente ou baixe nos sites oficiais:",
        "setup_install_btn": "Instalar Pendentes Automaticamente",
        "setup_recheck_btn": "Verificar Novamente",
        "setup_continue_btn": "Continuar para o Leonida",
        "setup_download_btn": "Site Oficial",
        "setup_manual_tag": "Manual / Script",
        "setup_installing": "Instalando dependências pendentes...",
        "setup_done": "Instalação concluída. Revalidando ambiente...",
        "setup_no_pending": "Nenhum item pendente selecionado para instalação automática.",
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
        "disk_avail": "Available disk space:",
        "val_env_btn": "Validate Environment",
        "val_setup_btn": "Setup Wizard",
        "val_game_quick": "Validate Game (Quick)",
        "val_game_full": "Validate Game (SHA-256)",
        "val_code_btn": "Validate Codebase",
        "export_report": "Export JSON Report",
        "val_prompt": "Click a button above to validate your environment, game copy, or codebase.",
        "val_running": "Validating...",
        "val_click_hint": "Click for details ▾",
        "val_missing_hdr": "Missing files (MISSING)",
        "val_incomplete_hdr": "Incomplete / wrong size files (SIZE)",
        "val_changed_hdr": "Corrupted / SHA-256 mismatch files (CHANGED)",
        "val_expected": "expected",
        "val_all_files_ok": "All manifest files are present and verified.",
        "clear_logs": "Clear",
        "copy_logs": "Copy",
        "save_logs": "Save Log...",
        "save_settings": "Save Settings",
        "settings_saved": "Settings saved to:",
        "lang_auto": "Automatic (System)",
        "lang_label": "Language",
        "open_viewer": "Launch 3D Map Viewer",
        "open_site": "Open Official Website",
        "open_releases": "View GitHub Releases",
        "py_missing": "Python 3.12+ was not found on PATH.\nUse the Setup Wizard in the Validation tab to install pending prerequisites.",
        "select_game_first": "Please select the game folder in the Build tab first.",
        "no_game_found": "No supported copy (Xbox Series 1.0.2.1) was found automatically.\nSelect the folder containing MicrosoftGame.Config and the .rpf files.",
        "legal_notice": (
            "Notice: Leonida contains no game files, keys, or executables. "
            "Everything is built locally on your PC from your own copy."
        ),
        "setup_title": "Environment Verification",
        "setup_sub": "Checking required prerequisites on your system. Only pending items will be installed.",
        "setup_all_ok": "All required prerequisites are installed!",
        "setup_missing": "Some prerequisites are missing. Install them automatically or download from the official websites:",
        "setup_install_btn": "Auto-Install Pending",
        "setup_recheck_btn": "Re-check",
        "setup_continue_btn": "Continue to Leonida",
        "setup_download_btn": "Official Site",
        "setup_manual_tag": "Manual / Script",
        "setup_installing": "Installing pending dependencies...",
        "setup_done": "Installation finished. Re-checking environment...",
        "setup_no_pending": "No pending items selected for automatic installation.",
    },
    "es": {
        "nav_build": "Compilar",
        "nav_validate": "Validación",
        "nav_options": "Opciones",
        "nav_logs": "Consola",
        "nav_about": "Mapa & Acerca",
        "subtitle": "Constructor y Validador Nativo para PC",
        "game_folder": "Carpeta del Juego (Xbox Series 1.0.2.1)",
        "out_folder": "Carpeta de Salida (--out)",
        "work_folder": "Carpeta de Trabajo (--work, opcional)",
        "auto_find": "Detectar Automáticamente",
        "browse": "Examinar...",
        "stages_title": "Etapas del Build",
        "start_build": "Iniciar Build",
        "stop_build": "Cancelar Build",
        "idle": "Listo",
        "running": "Ejecutando...",
        "finished_ok": "¡Completado con éxito!",
        "finished_err": "Finalizado con errores",
        "disk_avail": "Espacio libre en disco:",
        "val_env_btn": "Validar Entorno",
        "val_setup_btn": "Asistente de Instalación",
        "val_game_quick": "Validar Juego (Rápido)",
        "val_game_full": "Validar Juego (SHA-256)",
        "val_code_btn": "Validar Código",
        "export_report": "Exportar Informe JSON",
        "val_prompt": "Haz clic arriba para validar el entorno, tu copia del juego o el código.",
        "val_running": "Validando...",
        "val_click_hint": "Haz clic para ver detalles ▾",
        "val_missing_hdr": "Archivos ausentes (MISSING)",
        "val_incomplete_hdr": "Archivos incompletos / tamaño distinto (SIZE)",
        "val_changed_hdr": "Archivos corruptos / SHA-256 distinto (CHANGED)",
        "val_expected": "esperado",
        "val_all_files_ok": "Todos los archivos del manifiesto están presentes e íntegros.",
        "clear_logs": "Limpiar",
        "copy_logs": "Copiar",
        "save_logs": "Guardar Log...",
        "save_settings": "Guardar Configuración",
        "settings_saved": "Configuración guardada en:",
        "lang_auto": "Automático (Sistema)",
        "lang_label": "Idioma",
        "open_viewer": "Abrir Visor de Mapa 3D",
        "open_site": "Abrir Sitio Oficial",
        "open_releases": "Ver Releases en GitHub",
        "py_missing": "No se encontró Python 3.12+ en PATH.\nUsa el Asistente de Instalación en Validación para instalar lo pendiente.",
        "select_game_first": "Selecciona primero la carpeta del juego en la pestaña Compilar.",
        "no_game_found": "No se encontró ninguna copia compatible (Xbox Series 1.0.2.1).\nSelecciona la carpeta con MicrosoftGame.Config y los archivos .rpf.",
        "legal_notice": (
            "Aviso: Leonida no contiene archivos del juego, claves ni ejecutables. "
            "Todo se compila localmente desde tu propia copia."
        ),
        "setup_title": "Verificación del Entorno",
        "setup_sub": "Comprobando los requisitos previos del sistema. Solo se instalará lo pendiente.",
        "setup_all_ok": "¡Todos los requisitos obligatorios están instalados!",
        "setup_missing": "Faltan algunos requisitos. Instálalos automáticamente o descárgalos desde los sitios oficiales:",
        "setup_install_btn": "Instalar Pendientes Automáticamente",
        "setup_recheck_btn": "Verificar de Nuevo",
        "setup_continue_btn": "Continuar a Leonida",
        "setup_download_btn": "Sitio Oficial",
        "setup_manual_tag": "Manual / Script",
        "setup_installing": "Instalando dependencias pendientes...",
        "setup_done": "Instalación finalizada. Verificando de nuevo...",
        "setup_no_pending": "No hay elementos pendientes seleccionados para instalación automática.",
    },
}


def tr(lang: str, key: str) -> str:
    table = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return table.get(key, TRANSLATIONS["en"].get(key, key))
