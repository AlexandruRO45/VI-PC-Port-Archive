"""Run the synthetic mask/capture matrix; requires a real Vulkan GPU and validation."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--glslang", default="glslangValidator")
    parser.add_argument("--spirv-as", default="spirv-as")
    parser.add_argument("--work", type=Path, help="Keep synthetic artifacts in this directory")
    parser.add_argument("--benchmark", action="store_true", help="Also time 1/16/64 triangles at 8/32 pixels")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("Assertions must remain enabled; do not run with -O or PYTHONOPTIMIZE")
    for tool in (args.harness, args.glslang, args.spirv_as):
        if not shutil.which(str(tool)):
            parser.error(f"Required executable unavailable: {tool}")
    root = Path(__file__).resolve().parent
    temporary = tempfile.TemporaryDirectory(prefix="leonida-inner-coverage-") if args.work is None else None
    work = Path(temporary.name) if temporary else args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(LEONIDA_INNER_WORK=str(work), LEONIDA_INNER_HARNESS=str(args.harness.resolve()),
               LEONIDA_INNER_GLSLANG=shutil.which(args.glslang), PROBE_SYNC_VALIDATION="1")
    assembler = Path(shutil.which(args.spirv_as)).resolve()
    env["PATH"] = str(assembler.parent) + os.pathsep + env.get("PATH", "")
    validator = shutil.which("spirv-val", path=env["PATH"])
    if not validator:
        parser.error("spirv-val must be next to spirv-as or on PATH")
    # ps_oracle invokes spirv-as by name. Verify the selected binary has that name.
    if assembler.stem != "spirv-as":
        parser.error("--spirv-as must name the spirv-as executable")
    try:
        for name in ("check_basic.py", "check_batched_coverage.py", "check_batch_regressions.py",
                     "check_scratch_reuse.py", "check_generated_geometry.py", "check_geometry_capture.py",
                     "check_helper_counter.py"):
            subprocess.run([sys.executable, str(root / name)], env=env, check=True)
        if args.benchmark:
            subprocess.run([sys.executable, str(root / "benchmark.py")], env=env, check=True)
        modules = list(work.rglob("*.spv"))
        for module in modules:
            subprocess.run([validator, "--target-env", "vulkan1.2", str(module)], check=True)
        print(f"SPIR-V validation: {len(modules)} generated modules passed.")
        print("PASS: synthetic mask, semantic, reuse and capture suites; core + sync validation required.")
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
