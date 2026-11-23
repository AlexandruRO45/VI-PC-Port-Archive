"""Run Windows glslang on a standalone test fixture using ASCII filenames and cwd."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def scratch_directory(output_dir):
    # A non-ASCII profile can place TEMP outside the compiler's code page too.
    for parent in (output_dir, Path(tempfile.gettempdir()), Path(os.environ["SystemRoot"]) / "Temp"):
        if str(parent).isascii():
            try:
                return tempfile.TemporaryDirectory(prefix="leonida-glsl-", dir=parent)
            except OSError:
                continue
    raise OSError("glslang needs a writable ASCII temporary directory; set TEMP to one and rebuild")


def main():
    compiler, source, output, *flags = sys.argv[1:]
    source, output = Path(source), Path(output)
    with scratch_directory(output.parent) as folder:
        # Fixture basenames are ASCII. Keep them for stage inference and diagnostics.
        shader = source.name
        shutil.copyfile(source, Path(folder, shader))
        result = subprocess.run([compiler, *flags, "-o", "result.spv", shader], cwd=folder)
        if result.returncode:
            return result.returncode
        # The scratch folder may be on another volume. Publish only a complete module.
        with tempfile.TemporaryDirectory(dir=output.parent) as staging:
            staged = Path(staging, "result.spv")
            shutil.copyfile(Path(folder, "result.spv"), staged)
            os.replace(staged, output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as exc:
        sys.exit(f"ERROR: {exc}")
