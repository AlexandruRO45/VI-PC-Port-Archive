"""Convert a PE32+ x64 EXE into a recompiled Windows EXE with one command, or report the first refusal.

The stages run in order: header, imports, translate, build. Each profile names the imports the shared
lifter can bind for a whole program. The first profile that binds every import is translated. A refusal
names its stage, the first refused RVA when the lifter gives one, and every import without a binding.
OUT_DIR/report.json is written for every outcome. Generated files are named <stem>_aot.cpp and
<stem>_aot.exe; stale copies from an earlier run are removed first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pefile

import cpu_stdio_exe_aot


# Profile name -> (import bindings, host source, translator). Add new profiles here.
PROFILES = {
    "console-stdio": (cpu_stdio_exe_aot.IMPORTS, cpu_stdio_exe_aot.HOST, cpu_stdio_exe_aot.translate),
}
OPTIONAL_IMPORTS = {cpu_stdio_exe_aot.STARTUP_ONLY}
FIRST_RVA = re.compile(r"\bRVA (0x[0-9a-f]+)")
CXX_FLAGS = ("/nologo", "/std:c++17", "/O2", "/MT", "/EHsc")


class Refusal(Exception):
    def __init__(self, stage, message, **details):
        super().__init__(message)
        self.stage = stage
        self.details = details


def read_header(data):
    try:
        pe = pefile.PE(data=data, fast_load=True)
    except pefile.PEFormatError as exc:
        raise Refusal("header", f"not a PE image: {exc}") from exc
    with pe:
        header, opt = pe.FILE_HEADER, pe.OPTIONAL_HEADER
        if header.Machine != 0x8664 or opt.Magic != 0x20b:
            raise Refusal("header", f"expected an AMD64 PE32+ image, found machine {header.Machine:#x} "
                                    f"and optional-header magic {opt.Magic:#x}")
        if header.Characteristics & 0x2000 or not header.Characteristics & 2:
            raise Refusal("header", "expected an executable image, found a DLL or an unlinked image")
        directories = opt.DATA_DIRECTORY
        pe.parse_data_directories(directories=[1])
        imports = []
        for module in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll = module.dll.decode("ascii", "replace").lower()
            for imp in module.imports:
                imports.append((dll, imp.name if imp.name else imp.ordinal))
        return {
            "subsystem": opt.Subsystem,
            "entry_rva": f"{opt.AddressOfEntryPoint:#x}",
            "tls_directory": len(directories) > 9 and bool(directories[9].VirtualAddress),
            "delay_imports": len(directories) > 13 and bool(directories[13].VirtualAddress),
        }, imports


def import_label(dll, name):
    return f"{dll}!{name.decode('ascii', 'replace')}" if isinstance(name, bytes) else f"{dll}!#{name}"


def choose_profile(imports, entry_export):
    """Return (name, unbound labels) for the profile that leaves the fewest imports unbound."""
    best = None
    for name, (bindings, _, _) in PROFILES.items():
        unbound = [import_label(*key) for key in imports if key not in bindings
                   and not (entry_export and key in OPTIONAL_IMPORTS)]
        if best is None or len(unbound) < len(best[1]):
            best = name, unbound
    return best


def build(cxx, source, exe):
    compiler = shutil.which(str(cxx)) if cxx else shutil.which("cl")
    if compiler is None:
        raise Refusal("build", f"C++ compiler {cxx or 'cl'} not found; run tools/dev.ps1 or pass --cxx")
    # The provenance export would otherwise make the linker write an import library and .exp file.
    command = [compiler, *CXX_FLAGS, source.name, f"/Fe{exe.name}", f"/Fo{source.stem}.obj",
               "/link", "/NOIMPLIB", "/NOEXP"]
    result = subprocess.run(command, cwd=source.parent, capture_output=True, text=True, errors="replace")
    if result.returncode or not exe.is_file():
        tail = (result.stdout + result.stderr).strip().splitlines()[-10:]
        raise Refusal("build", f"{Path(compiler).name} exited {result.returncode}", output=tail)
    return Path(compiler).name


def convert(image, out_dir, *, entry_export=None, cxx=None, compile_exe=True):
    image = Path(image).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / f"{image.stem}_aot.cpp"
    exe = out_dir / f"{image.stem}_aot.exe"
    report_path = out_dir / "report.json"
    artifacts = (source, exe, *(exe.with_suffix(suffix) for suffix in (".obj", ".lib", ".exp")))
    if any(path == image or (path.exists() and image.samefile(path)) for path in (*artifacts, report_path)):
        raise ValueError(f"output would overwrite the input {image}")
    if report_path.is_symlink():
        raise ValueError(f"report path is a symbolic link: {report_path}")
    data = image.read_bytes()
    for stale in artifacts:
        stale.unlink(missing_ok=True)

    report = {"input": str(image), "sha256": hashlib.sha256(data).hexdigest(), "entry_export": entry_export}
    try:
        report["header"], imports = read_header(data)
        report["imports"] = [import_label(*key) for key in imports]
        profile, unbound = choose_profile(imports, entry_export)
        report["profile"] = profile
        if unbound:
            raise Refusal("imports", f"{len(unbound)} import(s) have no binding in profile {profile}; "
                                     f"first {unbound[0]}", unbound=unbound)
        bindings, host, translate = PROFILES[profile]
        try:
            text = translate(image, bindings, host, entry_export=entry_export)
        except ValueError as exc:
            match = FIRST_RVA.search(str(exc))
            raise Refusal("translate", str(exc), rva=match and match.group(1)) from exc
        source.write_text(text, encoding="ascii")
        report["source"] = str(source)
        if compile_exe:
            report["compiler"] = build(cxx, source, exe)
            report["exe"] = str(exe)
        report["status"] = "built" if compile_exe else "translated"
    except Refusal as refusal:
        report["status"] = "refused"
        report["refusal"] = {"stage": refusal.stage, "message": str(refusal), **refusal.details}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="PE32+ x64 executable to convert")
    parser.add_argument("out_dir", type=Path, help="directory for the generated C++, EXE and report.json")
    parser.add_argument("--entry-export", help="lift this returning export instead of the PE entry point")
    parser.add_argument("--cxx", help="C++ compiler (default: cl on PATH)")
    parser.add_argument("--no-build", action="store_true", help="stop after writing the generated C++")
    args = parser.parse_args(argv)
    try:
        report = convert(args.image, args.out_dir, entry_export=args.entry_export, cxx=args.cxx,
                         compile_exe=not args.no_build)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    if report["status"] == "refused":
        refusal = report["refusal"]
        print(f"refused at {refusal['stage']}: {refusal['message']}", file=sys.stderr)
        for line in refusal.get("unbound", [])[1:]:
            print(f"  also unbound: {line}", file=sys.stderr)
        for line in refusal.get("output", []):
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"{report['status']} with profile {report['profile']}: {report.get('exe', report['source'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
