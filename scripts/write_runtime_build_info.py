"""Package public source/build metadata with the runtime-windows CI artifact.

    python -I scripts/write_runtime_build_info.py runtime/build

Only the named CI variables and CMake cache entries below are recorded. The source
commit comes from git, including the checked-out merge commit on pull requests.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from report_paths import same_report_path

CACHE_FIELDS = ("CMAKE_BUILD_TYPE", "CMAKE_GENERATOR", "LEONIDA_D3D12_SHIM_TESTS", "LEONIDA_REQUIRE_VULKAN")
CI_FIELDS = ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")


def write_build_info(build_dir, *, repo_root=ROOT, environ=None):
    environ = os.environ if environ is None else environ
    cache = Path(build_dir) / "CMakeCache.txt"
    output = Path(build_dir) / "build-info.json"
    if same_report_path(cache, output):
        raise ValueError("build-info.json must not overwrite CMakeCache.txt")

    configuration = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        name, typed, _ = key.partition(":")
        if separator and typed and name in CACHE_FIELDS:
            configuration[name] = value.strip()
    missing = [name for name in CACHE_FIELDS if not configuration.get(name)]
    if missing:
        raise ValueError("missing CMake cache fields: " + ", ".join(missing))
    missing = [name for name in CI_FIELDS if not environ.get(name)]
    if missing:
        raise ValueError("missing CI metadata: " + ", ".join(missing))

    commit = subprocess.run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=repo_root,
                            check=True, capture_output=True, text=True).stdout.strip()
    run_url = (f"{environ['GITHUB_SERVER_URL'].rstrip('/')}/{environ['GITHUB_REPOSITORY']}"
               f"/actions/runs/{environ['GITHUB_RUN_ID']}/attempts/{environ['GITHUB_RUN_ATTEMPT']}")
    info = {"schema_version": 1, "source_commit": commit, "workflow_run_url": run_url,
            "build_configuration": {name: configuration[name] for name in CACHE_FIELDS}}
    with output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path, help="configured Windows runtime build directory")
    args = parser.parse_args(argv)
    try:
        write_build_info(args.build_dir)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
