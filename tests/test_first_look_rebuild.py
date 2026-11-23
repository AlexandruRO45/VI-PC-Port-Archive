"""scripts/first_look.ps1 checks for a current and complete runtime before a dry run.

    python -I tests/test_first_look_rebuild.py

A gta6rt.exe from before a pull still runs, but its dry run reports the old shim table: the day-one import to-do list
would be wrong without any error. The script used to build the loader only when it was missing. No game files.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "first_look.ps1"
PE_INSPECT_STUB = (
    "from pathlib import Path\n"
    "import sys\n"
    "report = Path(sys.argv[-1]) / (Path(sys.argv[1]).name + '.json')\n"
    "report.write_text('{\"imports\":[],\"sections\":[]}', encoding='utf-8', newline='\\n')\n"
)
# The rebuild test is about the loader, not about the other tools the script runs. Each of those must be a python
# call that exits 0: first_look.ps1 records the dry run's code from $LASTEXITCODE, and the loader is stood in for by a
# Write-Output alias, which does not set it. A tool left out of the fixture leaves 1 behind, and the script then exits
# 1 even when it rebuilt and dry-ran correctly.
TOOL_STUB_TEMPLATE = "raise SystemExit(%d)\n"
# The build script leaves a marker; the loader is replaced by an alias that echoes its arguments.
BUILD_STUB = r"""
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'built.txt') -Value 'built'
if ($env:LEONIDA_TEST_INCOMPLETE_RUNTIME -ne '1') {
    $build = Join-Path (Split-Path $PSScriptRoot -Parent) 'runtime\build'
    foreach ($name in 'gta6rt.exe', 'gta6rt_tls_host.dll') {
        New-Item -ItemType File -Path (Join-Path $build $name) -Force | Out-Null
    }
    if ($env:LEONIDA_TEST_LOADER) {
        Copy-Item -LiteralPath $env:LEONIDA_TEST_LOADER -Destination (Join-Path $build 'gta6rt.exe') -Force
    }
}
"""
# The loader is a real executable: first_look.ps1 takes the dry run's exit status from it, and a PowerShell alias
# has none. It echoes its arguments and exits 0, which is what the script's report and the assertions read.
RUN = r"""
$ErrorActionPreference = 'Stop'
$stub = Join-Path $PWD 'loader-stub.exe'
Add-Type -TypeDefinition 'public static class LoaderStub5977 { public static int Main(string[] a) { System.Console.WriteLine(string.Join(" ", a)); return 0; } }' -OutputAssembly $stub -OutputType ConsoleApplication
$loader = Join-Path $PWD 'runtime\build\gta6rt.exe'
# A case that deletes the loader expects it rebuilt, so only stand in for a loader the fixture left in place.
if (Test-Path -LiteralPath $loader) {
    # Keep the fixture's timestamp: the copy's own would make a deliberately older loader look current.
    $stamp = (Get-Item -LiteralPath $loader).LastWriteTimeUtc
    Copy-Item -LiteralPath $stub -Destination $loader -Force
    (Get-Item -LiteralPath $loader).LastWriteTimeUtc = $stamp
}
$env:LEONIDA_TEST_LOADER = $stub
& .\scripts\first_look.ps1 -Exe (Join-Path $PWD 'fixture.exe')
"""
# The same run with no stub: the fixture's empty gta6rt.exe is not a valid executable, so the dry run cannot start.
RUN_LOADER_CANNOT_START = r"""
$ErrorActionPreference = 'Stop'
& .\scripts\first_look.ps1 -Exe (Join-Path $PWD 'fixture.exe')
"""


@unittest.skipUnless(os.name == "nt", "Windows PowerShell diagnostics script")
class FirstLookRebuildTest(unittest.TestCase):
    def run_case(self, source_newer=False, missing=(), incomplete_build=False, tool_exit=0, stub_loader=True,
                 cache=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("scripts", "tools", "runtime/build", "runtime/src"):
                (root / folder).mkdir(parents=True)
            shutil.copyfile(SCRIPT, root / "scripts" / SCRIPT.name)
            (root / "scripts" / "build_runtime.ps1").write_text(BUILD_STUB, encoding="utf-8")
            (root / "tools" / "pe_inspect.py").write_text(PE_INSPECT_STUB, encoding="utf-8", newline="\n")
            for name in ("gdk_runtime_calls.py", "find_hash_lookup.py", "exe_checklist.py"):
                (root / "tools" / name).write_text(TOOL_STUB_TEMPLATE % tool_exit, encoding="utf-8", newline="\n")
            (root / "fixture.exe").write_bytes(b"MZ")
            loader = root / "runtime" / "build" / "gta6rt.exe"
            if cache is not None:
                (loader.parent / "CMakeCache.txt").write_text(cache, encoding="utf-8")
            source = root / "runtime" / "src" / "pe_loader.cpp"
            tls = loader.with_name("gta6rt_tls_host.dll")
            loader.write_bytes(b"")
            tls.touch()
            source.write_text("// source\n", encoding="utf-8")
            now = time.time()
            older, newer = now - 3600, now
            os.utime(loader, (newer, newer) if not source_newer else (older, older))
            os.utime(source, (newer, newer) if source_newer else (older, older))
            for name in missing:
                loader.with_name(name).unlink()
            env = {k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"}
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
            env["LEONIDA_TEST_INCOMPLETE_RUNTIME"] = "1" if incomplete_build else "0"
            run = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                  "-Command", RUN if stub_loader else RUN_LOADER_CANNOT_START], cwd=root, env=env,
                                 capture_output=True, text=True, timeout=60)
            dry_run = (root / "reports" / "exe" / "fixture" / "dryrun.txt").is_file()
            return (root / "scripts" / "built.txt").is_file(), dry_run, run.returncode, run.stdout + run.stderr

    def test_a_failing_tool_does_not_become_the_dry_runs_exit_status(self):
        # A tool that fails only warns, and its code must not become the script's. Today the loader masks it by
        # setting $LASTEXITCODE itself; the reset in first_look.ps1 is what keeps that true when it does not (the
        # case below), and this pins the property so a changed loader call cannot quietly break it.
        built, dry_run, code, out = self.run_case(tool_exit=3)
        self.assertFalse(built, out)
        self.assertTrue(dry_run, out)
        self.assertEqual(code, 0, out)
        self.assertIn("gdk_runtime_calls could not read all of", out)

    def test_a_loader_that_cannot_start_is_not_a_tools_exit_status(self):
        # The fixture's gta6rt.exe is not a real executable. Windows PowerShell reports that as exit 1 itself, so
        # the old script also ended at 1 - but said nothing about the dry run never happening, and with a loader
        # that cannot be found at all $LASTEXITCODE is left untouched, which is the tool's code.
        built, dry_run, code, out = self.run_case(tool_exit=2, stub_loader=False)
        self.assertFalse(built, out)
        self.assertFalse(dry_run, out)
        self.assertEqual(code, 1, out)
        self.assertIn("the loader did not run", out)

    def test_a_loader_older_than_the_sources_is_rebuilt(self):
        built, dry_run, code, out = self.run_case(source_newer=True)
        self.assertTrue(built, out)
        self.assertTrue(dry_run, out)
        self.assertEqual(code, 0, out)
        self.assertIn("rebuilding it first", out)

    def test_an_up_to_date_loader_is_used_as_it_is(self):
        built, dry_run, code, out = self.run_case()
        self.assertFalse(built, out)
        self.assertTrue(dry_run, out)
        self.assertEqual(code, 0, out)

    def test_a_missing_runtime_file_is_rebuilt(self):
        for name in ("gta6rt.exe", "gta6rt_tls_host.dll"):
            with self.subTest(name=name):
                built, dry_run, code, out = self.run_case(missing=(name,))
                self.assertTrue(built, out)
                self.assertTrue(dry_run, out)
                self.assertEqual(code, 0, out)

    def test_an_incomplete_build_stops_before_the_dry_run(self):
        built, dry_run, code, out = self.run_case(missing=("gta6rt_tls_host.dll",), incomplete_build=True)
        self.assertTrue(built, out)
        self.assertFalse(dry_run, out)
        self.assertNotEqual(code, 0, out)
        self.assertIn("runtime build did not produce", out)
        self.assertIn("gta6rt_tls_host.dll", out)

    def test_incomplete_visual_studio_cache_names_the_missing_command(self):
        generator = "CMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022\n"
        for command in ("", "CMAKE_COMMAND:INTERNAL=\n"):
            with self.subTest(command=command):
                built, dry_run, code, out = self.run_case(source_newer=True, cache=generator + command)
                self.assertNotEqual(code, 0, out)
                self.assertFalse(built, out)
                self.assertIn("CMakeCache.txt", out)
                self.assertIn("Reconfigure", out)
                self.assertIn("CMAKE_COMMAND", out)
                self.assertFalse(dry_run, out)


if __name__ == "__main__":
    unittest.main()
