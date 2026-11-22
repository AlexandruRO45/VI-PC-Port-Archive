"""Validation engine for Leonida: environment checks, game package verification, and codebase integrity.

Usage:
    python -I tools/validate.py env [--stages check,shaders,assets,runtime,pipelines] [--work DIR] [--json]
    python -I tools/validate.py game <game_dir> [--full] [--jobs N] [--json]
    python -I tools/validate.py code [--json]
    python -I tools/validate.py report [--game DIR] [--out FILE]

Standard library only. Never reads or exports game content; reports only sizes, hashes, and tool status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Callable

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TOOLS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import manifest as manifest_mod  # noqa: E402
from mc_command import mc_command  # noqa: E402

BLOCKED_DATA_EXTENSIONS = {
    ".rpf", ".ttpf", ".xvd", ".msixvc", ".dxbc", ".fssm", ".spv", ".spvasm",
    ".dds", ".bk2", ".awc", ".r9modeldata", ".obj", ".glb", ".ogg", ".wav",
    ".exe", ".dll",
}

HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PROBES = ("MicrosoftGame.Config", "Catalog000.bin", "appdata0.rpf")
MIN_FREE_GB = {"shaders": 30, "assets": 300}
ALL_STAGES = ("check", "shaders", "assets", "runtime", "pipelines")


@dataclass
class CheckItem:
    id: str
    name: str
    status: str  # "ok", "warn", "error", "skip"
    detail: str
    required: bool = True


@dataclass
class GameFileIssue:
    path: str
    kind: str  # "MISSING", "SIZE", "CHANGED", "NEW"
    expected_size: int | None = None
    actual_size: int | None = None


@dataclass
class ValidationReport:
    kind: str
    ok: bool
    timestamp: str
    platform: str
    summary: str
    checks: list[dict] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _platform_label() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def _tool_available(cmd: str) -> bool:
    cmd = cmd.strip().strip('"')
    if os.path.isfile(cmd):
        return True
    parts = cmd.split()
    head = parts[0].strip('"') if parts else ""
    return bool(head and (shutil.which(head) or os.path.isfile(head)))


def _dotnet_sdk_version() -> str | None:
    if not shutil.which("dotnet"):
        return None
    try:
        r = subprocess.run(["dotnet", "--list-sdks"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    versions = re.findall(r"^(\d+\.\d+\.\S+)", r.stdout, re.M)
    supported = [v for v in versions if int(v.split(".")[0]) >= 8]
    return supported[-1] if supported else None


def _visual_cpp_ok() -> bool:
    if os.name != "nt":
        return False
    vswhere = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        "Microsoft Visual Studio", "Installer", "vswhere.exe",
    )
    if not os.path.isfile(vswhere):
        return False
    try:
        r = subprocess.run(
            [
                vswhere, "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def _llvm_mc_status() -> tuple[str, str]:
    cmd = mc_command()
    head = cmd[0]
    if not _tool_available(head):
        return "error", f"Not found ({head}). Set LLVM_MC or build with scripts/build_llvm_mc.ps1."
    try:
        r = subprocess.run([*cmd, "--version"], capture_output=True, text=True, errors="replace", timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return "error", f"Failed to run {head}: {exc}"
    out = r.stdout or ""
    if r.returncode != 0:
        return "error", f"{head} exited with {r.returncode}"
    has_amdgpu = "amdgcn" in out.lower() or "amdgpu" in out.lower()
    m = re.search(r"LLVM version (\d+\.\d+\.\d+)", out)
    ver = m.group(1) if m else "unknown"
    if not has_amdgpu:
        return "error", f"LLVM {ver} found at {head}, but AMDGPU target is missing."
    major = int(ver.split(".")[0]) if m else 0
    if major and major < 21:
        return "warn", f"LLVM {ver} with AMDGPU found (LLVM 21+ recommended)."
    return "ok", f"LLVM {ver} with AMDGPU target ({head})"


def _disk_free_gb(path: str) -> float | None:
    target = os.path.abspath(path)
    while target and not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    try:
        return shutil.disk_usage(target).free / (2**30)
    except OSError:
        return None


def _external_python_312() -> tuple[str | None, str | None]:
    """Find an external Python 3.12+ interpreter on PATH (needed when running from a frozen executable)."""
    candidates = ["py", "python", "python3"] if os.name == "nt" else ["python3", "python"]
    for name in candidates:
        path = shutil.which(name)
        if not path:
            continue
        cmd = [path, "-3.12"] if name == "py" else [path]
        try:
            r = subprocess.run(
                [
                    *cmd, "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); "
                    "print(sys.executable); sys.exit(0 if sys.version_info >= (3, 12) else 1)",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
                if len(lines) >= 2:
                    return lines[0], lines[1]
        except (OSError, subprocess.SubprocessError):
            continue
    return None, None


def check_environment(
    stages: list[str] | tuple[str, ...] = ALL_STAGES,
    work_dir: str | None = None,
    wav: bool = False,
    root_dir: str = ROOT_DIR,
) -> ValidationReport:
    """Inspect system prerequisites without starting a build."""
    checks: list[CheckItem] = []

    if getattr(sys, "frozen", False):
        ext_ver, ext_path = _external_python_312()
        py_ok = ext_ver is not None
        py_detail = (
            f"Python {ext_ver} ({ext_path})"
            if py_ok
            else "Not found on PATH (install Python 3.12+ and add to PATH)"
        )
    else:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        py_ok = sys.version_info >= (3, 12)
        py_detail = f"Python {py_ver} ({sys.executable})"
    checks.append(CheckItem(
        id="python",
        name="Python 3.12+",
        status="ok" if py_ok else "error",
        detail=py_detail,
        required=True,
    ))

    # LLVM (shaders)
    need_shaders = "shaders" in stages
    mc_status, mc_detail = _llvm_mc_status()
    checks.append(CheckItem(
        id="llvm_mc",
        name="LLVM llvm-mc (AMDGPU)",
        status=mc_status if need_shaders else ("ok" if mc_status == "ok" else "skip"),
        detail=mc_detail,
        required=need_shaders,
    ))

    # Vulkan SDK tools
    for tool in ("spirv-as", "spirv-val", "vulkaninfo"):
        path = shutil.which(tool)
        req = need_shaders and tool != "vulkaninfo"
        checks.append(CheckItem(
            id=tool.replace("-", "_"),
            name=f"Vulkan SDK ({tool})",
            status="ok" if path else ("error" if req else "warn"),
            detail=path or "Not found on PATH (install Vulkan SDK)",
            required=req,
        ))

    # .NET 8 SDK
    need_assets = "assets" in stages
    dotnet_ver = _dotnet_sdk_version()
    checks.append(CheckItem(
        id="dotnet",
        name=".NET 8+ SDK",
        status="ok" if dotnet_ver else ("error" if need_assets else "warn"),
        detail=f".NET SDK {dotnet_ver}" if dotnet_ver else "Not found (.NET 8 SDK required for rpf9dump)",
        required=need_assets,
    ))

    # rpf9-explorer submodule
    core_dir = os.path.join(root_dir, "third_party", "rpf9-explorer", "Core")
    sub_ok = os.path.isdir(core_dir) and bool(os.listdir(core_dir))
    checks.append(CheckItem(
        id="submodule",
        name="rpf9-explorer submodule",
        status="ok" if sub_ok else ("warn" if need_assets else "skip"),
        detail="Initialized" if sub_ok else "Not initialized (run: git submodule update --init --recursive)",
        required=False,
    ))

    # C++ toolchain
    need_runtime = "runtime" in stages
    if os.name == "nt":
        vs_ok = _visual_cpp_ok()
        checks.append(CheckItem(
            id="msvc",
            name="Visual Studio C++ (x64)",
            status="ok" if vs_ok else ("error" if need_runtime else "warn"),
            detail="Found via vswhere" if vs_ok else "Visual Studio 2022 Desktop C++ tools not detected",
            required=need_runtime,
        ))
    else:
        cmake = shutil.which("cmake")
        clang_cl = shutil.which("clang-cl")
        checks.append(CheckItem(
            id="cmake",
            name="CMake & Cross-build tools",
            status="ok" if (cmake and clang_cl) else ("warn" if need_runtime else "skip"),
            detail=f"cmake={cmake or 'missing'}, clang-cl={clang_cl or 'missing'}",
            required=False,
        ))

    # ffmpeg (optional unless --wav)
    ff = shutil.which("ffmpeg")
    checks.append(CheckItem(
        id="ffmpeg",
        name="FFmpeg (XMA2 / WAV)",
        status="ok" if ff else ("error" if (wav and need_assets) else "warn"),
        detail=ff or "Not found on PATH (optional unless --wav is enabled)",
        required=bool(wav and need_assets),
    ))

    # Disk space
    check_path = work_dir or os.path.expanduser("~")
    free_gb = _disk_free_gb(check_path)
    need_gb = sum(MIN_FREE_GB.get(s, 0) for s in stages)
    if free_gb is None:
        disk_status, disk_detail = "warn", f"Could not measure free space at {check_path}"
    elif need_gb > 0 and free_gb < need_gb:
        disk_status = "warn"
        disk_detail = f"{free_gb:.1f} GB free at {check_path} (~{need_gb} GB recommended for selected stages)"
    else:
        disk_status = "ok"
        disk_detail = f"{free_gb:.1f} GB free at {check_path}" + (f" (need ~{need_gb} GB)" if need_gb else "")
    checks.append(CheckItem(
        id="disk_space",
        name="Disk Space",
        status=disk_status,
        detail=disk_detail,
        required=False,
    ))

    errors = [c for c in checks if c.status == "error" and c.required]
    warns = [c for c in checks if c.status == "warn"]
    ok = len(errors) == 0
    summary = (
        f"Environment ready ({len(warns)} warning(s))"
        if ok else f"{len(errors)} required prerequisite(s) missing, {len(warns)} warning(s)"
    )
    return ValidationReport(
        kind="environment",
        ok=ok,
        timestamp=_now_iso(),
        platform=_platform_label(),
        summary=summary,
        checks=[asdict(c) for c in checks],
        details={"stages": list(stages), "work_dir": check_path, "free_gb": free_gb, "recommended_gb": need_gb},
    )


def load_game_manifests(root_dir: str = ROOT_DIR) -> list[tuple[str, dict]]:
    """Return [(manifest_path, parsed_dict), ...] from game/*.json."""
    game_dir = os.path.join(root_dir, "game")
    if not os.path.isdir(game_dir):
        return []
    out = []
    for name in sorted(os.listdir(game_dir)):
        if name.endswith(".json"):
            p = os.path.join(game_dir, name)
            with open(p, encoding="utf-8") as f:
                out.append((p, json.load(f)))
    return out


def validate_game_folder(
    game_dir: str,
    *,
    full_sha256: bool = False,
    jobs: int | None = None,
    root_dir: str = ROOT_DIR,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> ValidationReport:
    """Validate a game package folder against game/<version>.json (sizes + optional SHA-256)."""
    game_dir = os.path.abspath(game_dir)
    if not os.path.isdir(game_dir):
        return ValidationReport(
            kind="game_package",
            ok=False,
            timestamp=_now_iso(),
            platform=_platform_label(),
            summary=f"Folder does not exist: {game_dir}",
            checks=[asdict(CheckItem("folder", "Game folder", "error", f"Not found: {game_dir}"))],
        )

    manifests = load_game_manifests(root_dir)
    if not manifests:
        return ValidationReport(
            kind="game_package",
            ok=False,
            timestamp=_now_iso(),
            platform=_platform_label(),
            summary="No version manifests found in game/.",
        )

    matched_manifest = None
    probe_notes = []
    for _path, doc in manifests:
        files = doc.get("files", {})
        probes = [p for p in PROBES if p in files]
        bad = []
        for probe in probes:
            target = os.path.join(game_dir, probe)
            try:
                st = os.stat(target)
                if not stat.S_ISREG(st.st_mode):
                    bad.append(f"{probe}: not a file")
                elif st.st_size != files[probe]["size"]:
                    bad.append(f"{probe}: size {st.st_size} != {files[probe]['size']}")
            except OSError:
                bad.append(f"{probe}: missing")
        if not bad:
            matched_manifest = doc
            break
        probe_notes.append(f"{doc.get('title', '?')} {doc.get('version', '?')}: {', '.join(bad)}")

    if matched_manifest is None:
        return ValidationReport(
            kind="game_package",
            ok=False,
            timestamp=_now_iso(),
            platform=_platform_label(),
            summary="Folder does not match any supported game version.",
            checks=[asdict(CheckItem("probes", "Version probe files", "error", "; ".join(probe_notes)))],
            details={"game_dir": game_dir, "probe_notes": probe_notes},
        )

    want: dict[str, dict] = matched_manifest["files"]
    issues: list[GameFileIssue] = []
    to_hash: list[tuple[str, str, str]] = []
    total_files = len(want)
    checked = 0

    for rel, info in want.items():
        p = os.path.join(game_dir, *rel.split("/"))
        expected_size = info["size"]
        if not os.path.isfile(p):
            issues.append(GameFileIssue(rel, "MISSING", expected_size=expected_size))
        else:
            actual_size = os.path.getsize(p)
            if actual_size != expected_size:
                issues.append(GameFileIssue(rel, "SIZE", expected_size=expected_size, actual_size=actual_size))
            elif full_sha256:
                to_hash.append((rel, p, info["sha256"]))
        checked += 1
        if progress_cb and not full_sha256:
            progress_cb(checked, total_files, rel)

    if full_sha256 and to_hash:
        workers = jobs or max(1, (os.cpu_count() or 2) // 2)
        hashed = 0
        total_hash = len(to_hash)
        if workers == 1:
            for rel, p, digest in to_hash:
                if manifest_mod.sha256(p) != digest:
                    issues.append(GameFileIssue(rel, "CHANGED"))
                hashed += 1
                if progress_cb:
                    progress_cb(hashed, total_hash, rel)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(manifest_mod.sha256, p): (rel, digest) for rel, p, digest in to_hash}
                for fut in as_completed(futures):
                    rel, digest = futures[fut]
                    if fut.result() != digest:
                        issues.append(GameFileIssue(rel, "CHANGED"))
                    hashed += 1
                    if progress_cb:
                        progress_cb(hashed, total_hash, rel)

    exe_name = matched_manifest.get("expected_exe", "")
    exe_present = bool(exe_name and os.path.isfile(os.path.join(game_dir, exe_name)))

    checks = [
        CheckItem(
            id="version",
            name="Supported Version",
            status="ok",
            detail=f"{matched_manifest['title']} {matched_manifest['version']} ({total_files} files in manifest)",
        ),
        CheckItem(
            id="integrity",
            name="File Integrity" + (" (SHA-256)" if full_sha256 else " (Size & Presence)"),
            status="ok" if not issues else "error",
            detail="All manifest files verified cleanly." if not issues else f"{len(issues)} issue(s) found.",
        ),
        CheckItem(
            id="executable",
            name=f"Game Executable ({exe_name or 'N/A'})",
            status="ok" if exe_present else "warn",
            detail="Present in folder." if exe_present else f"{exe_name} not found (runtime can build, not run).",
            required=False,
        ),
    ]

    ok = len(issues) == 0
    summary = (
        f"Valid {matched_manifest['title']} {matched_manifest['version']} copy ({total_files} files"
        + (", SHA-256 verified)" if full_sha256 else ", sizes verified)")
        if ok else f"Validation failed: {len(issues)} file issue(s) detected"
    )
    return ValidationReport(
        kind="game_package",
        ok=ok,
        timestamp=_now_iso(),
        platform=_platform_label(),
        summary=summary,
        checks=[asdict(c) for c in checks],
        details={
            "game_dir": game_dir,
            "version": matched_manifest["version"],
            "title": matched_manifest["title"],
            "total_files": total_files,
            "full_sha256": full_sha256,
            "executable_present": exe_present,
            "issues": [asdict(i) for i in issues],
        },
    )


def validate_manifest_schema(path: str) -> list[str]:
    """Validate a game/<version>.json manifest file structure and SHA-256 formatting."""
    errors: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as exc:
        return [f"{os.path.basename(path)}: invalid JSON ({exc})"]
    if not isinstance(doc, dict):
        return [f"{os.path.basename(path)}: root must be a JSON object"]
    for key in ("title", "version", "files"):
        if key not in doc:
            errors.append(f"{os.path.basename(path)}: missing required key '{key}'")
    files = doc.get("files")
    if not isinstance(files, dict) or not files:
        errors.append(f"{os.path.basename(path)}: 'files' must be a non-empty object")
    else:
        for rel, entry in files.items():
            if not isinstance(entry, dict):
                errors.append(f"{rel}: entry must be an object")
                continue
            size = entry.get("size")
            digest = entry.get("sha256")
            if type(size) is not int or size < 0:
                errors.append(f"{rel}: 'size' must be a non-negative integer")
            if not isinstance(digest, str) or not HEX64_RE.match(digest):
                errors.append(f"{rel}: 'sha256' must be a 64-char hex digest")
    return errors


def validate_codebase(root_dir: str = ROOT_DIR) -> ValidationReport:
    """Self-validate the Leonida codebase: manifests, Python compilation, key files, and no-game-data safety."""
    checks: list[CheckItem] = []

    # 1. Key files
    required_files = [
        "make_game.py",
        "tools/manifest.py",
        "tools/validate.py",
        "gui/core.py",
        "game/xbox-1.0.2.1.json",
    ]
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(root_dir, f))]
    checks.append(CheckItem(
        id="core_files",
        name="Core Project Files",
        status="ok" if not missing_files else "error",
        detail="All core files present." if not missing_files else f"Missing: {', '.join(missing_files)}",
    ))

    # 2. Manifest schema validation
    manifest_dir = os.path.join(root_dir, "game")
    schema_errors: list[str] = []
    manifest_count = 0
    if os.path.isdir(manifest_dir):
        for name in sorted(os.listdir(manifest_dir)):
            if name.endswith(".json"):
                manifest_count += 1
                schema_errors.extend(validate_manifest_schema(os.path.join(manifest_dir, name)))
    checks.append(CheckItem(
        id="manifest_schema",
        name="Game Manifest Schemas",
        status="ok" if (manifest_count > 0 and not schema_errors) else "error",
        detail=(
            f"{manifest_count} manifest(s) valid."
            if (manifest_count > 0 and not schema_errors)
            else ("; ".join(schema_errors[:5]) or "No manifests found in game/")
        ),
    ))

    # 3. Python syntax check (in-memory compile so it works in read-only bundles too)
    compile_targets = [
        os.path.join(root_dir, "make_game.py"),
        os.path.join(root_dir, "tools"),
        os.path.join(root_dir, "gui"),
    ]
    compile_ok = True
    syntax_errors: list[str] = []

    def _check_py_file(fpath: str) -> None:
        nonlocal compile_ok
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                compile(f.read(), fpath, "exec")
        except (SyntaxError, ValueError, OSError) as exc:
            compile_ok = False
            syntax_errors.append(f"{os.path.relpath(fpath, root_dir)}: {exc}")

    for target in compile_targets:
        if os.path.isfile(target):
            _check_py_file(target)
        elif os.path.isdir(target):
            for base, dirs, files in os.walk(target):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for fname in sorted(files):
                    if fname.endswith(".py"):
                        _check_py_file(os.path.join(base, fname))
    checks.append(CheckItem(
        id="python_syntax",
        name="Python Syntax Validation",
        status="ok" if compile_ok else "error",
        detail=(
            "make_game.py, tools/, and gui/ parsed cleanly."
            if compile_ok else "; ".join(syntax_errors[:5])
        ),
    ))

    # 4. No-game-data check (via git ls-files when inside a git checkout)
    blocked_found: list[str] = []
    if shutil.which("git") and os.path.exists(os.path.join(root_dir, ".git")):
        try:
            r = subprocess.run(["git", "ls-files"], cwd=root_dir, capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    ext = os.path.splitext(line.strip())[1].lower()
                    if ext in BLOCKED_DATA_EXTENSIONS:
                        blocked_found.append(line.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    checks.append(CheckItem(
        id="no_game_data",
        name="No Game Data / Binaries Tracked",
        status="ok" if not blocked_found else "error",
        detail="Clean repository (0 restricted files tracked)." if not blocked_found else f"Found: {', '.join(blocked_found[:5])}",
    ))

    # 5. Bundled Outfit fonts
    font_dir = os.path.join(root_dir, "assets", "fonts")
    fonts_present = [
        f for f in ("Outfit-Regular.ttf", "Outfit-Medium.ttf", "Outfit-SemiBold.ttf", "Outfit-Bold.ttf")
        if os.path.isfile(os.path.join(font_dir, f))
    ]
    checks.append(CheckItem(
        id="outfit_fonts",
        name="Outfit UI Fonts",
        status="ok" if len(fonts_present) == 4 else "warn",
        detail=f"{len(fonts_present)}/4 Outfit weights present in assets/fonts/.",
        required=False,
    ))

    ok = all(c.status != "error" for c in checks if c.required)
    summary = "Codebase validation passed." if ok else "Codebase validation found errors."
    return ValidationReport(
        kind="codebase",
        ok=ok,
        timestamp=_now_iso(),
        platform=_platform_label(),
        summary=summary,
        checks=[asdict(c) for c in checks],
        details={"root_dir": root_dir, "manifest_count": manifest_count},
    )


def _print_report(report: ValidationReport, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return
    badge = "PASS" if report.ok else "FAIL"
    print(f"[{badge}] {report.kind}: {report.summary}")
    for c in report.checks:
        tag = c["status"].upper().ljust(5)
        print(f"  [{tag}] {c['name']}: {c['detail']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Leonida validation tool (environment, game copy, codebase).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_env = sub.add_parser("env", help="Validate local build prerequisites")
    p_env.add_argument("--stages", default=",".join(ALL_STAGES))
    p_env.add_argument("--work", default=None)
    p_env.add_argument("--wav", action="store_true")
    p_env.add_argument("--json", action="store_true")

    p_game = sub.add_parser("game", help="Validate a game package folder")
    p_game.add_argument("game_dir")
    p_game.add_argument("--full", action="store_true", help="Verify SHA-256 hashes of every file")
    p_game.add_argument("--jobs", type=int, default=None)
    p_game.add_argument("--json", action="store_true")

    p_code = sub.add_parser("code", help="Validate Leonida codebase integrity")
    p_code.add_argument("--json", action="store_true")

    p_rep = sub.add_parser("report", help="Generate a combined JSON validation report")
    p_rep.add_argument("--game", default=None)
    p_rep.add_argument("--full", action="store_true")
    p_rep.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "env":
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        rep = check_environment(stages=stages, work_dir=args.work, wav=args.wav)
        _print_report(rep, args.json)
        return 0 if rep.ok else 1

    if args.cmd == "game":
        rep = validate_game_folder(args.game_dir, full_sha256=args.full, jobs=args.jobs)
        _print_report(rep, args.json)
        return 0 if rep.ok else 1

    if args.cmd == "code":
        rep = validate_codebase()
        _print_report(rep, args.json)
        return 0 if rep.ok else 1

    if args.cmd == "report":
        payload = {
            "generated_at": _now_iso(),
            "platform": _platform_label(),
            "codebase": validate_codebase().to_dict(),
            "environment": check_environment().to_dict(),
        }
        if args.game:
            payload["game_package"] = validate_game_folder(args.game, full_sha256=args.full).to_dict()
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as f:
                f.write(text + "\n")
            print(f"Report written to {args.out}")
        else:
            print(text)
        overall_ok = payload["codebase"]["ok"] and payload["environment"]["ok"] and payload.get("game_package", {}).get("ok", True)
        return 0 if overall_ok else 1

    return 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
