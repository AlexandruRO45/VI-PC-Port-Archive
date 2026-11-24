"""Run one SPIR-V bridge GPU test, reporting missing SPIR-V tools as a CTest skip.

Usage: run_bridge_test.py <driver.py> <spirv-as> <spirv-val> <spirv-cross> <dxc> <offscreen_check> <adapter>

CMake registers every bridge test through this wrapper, whether or not the tools were found, so a
missing toolchain shows up as a skip (77) instead of an absent test. With
X2W_REQUIRE_SPIRV_TOOLS=1, which hosted CI sets, a missing tool or a skip from the test itself
fails instead: a green CI run then means the test actually ran.
"""
import os
from pathlib import Path
import subprocess
import sys

SKIP = 77


def main(argv):
    if len(argv) != 7:
        sys.exit(__doc__)
    driver, assembler, validator, cross, *rest = argv
    strict = os.environ.get("X2W_REQUIRE_SPIRV_TOOLS") == "1"
    missing = [name for name, path in (("spirv-as", assembler), ("spirv-val", validator), ("spirv-cross", cross))
               if not Path(path).is_file()]
    if missing:
        print(f"skip: {', '.join(missing)} not found at configure time; "
              "run tools/setup-spirv-tools.ps1 and reconfigure", flush=True)
        return 1 if strict else SKIP
    code = subprocess.run([sys.executable, "-B", driver, assembler, validator, cross, *rest]).returncode
    if code == SKIP and strict:
        print("error: the test skipped, but X2W_REQUIRE_SPIRV_TOOLS=1 requires it to run", flush=True)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
