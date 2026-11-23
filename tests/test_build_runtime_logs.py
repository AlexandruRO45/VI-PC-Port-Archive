"""Exercise live log polling with deterministic writes to a synthetic build log."""
import base64
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_runtime.ps1"


@unittest.skipUnless(os.name == "nt", "requires Windows PowerShell")
class BuildRuntimeLogsTest(unittest.TestCase):
    def test_live_polls_and_final_drain(self):
        script = SCRIPT.read_text(encoding="utf-8-sig")
        helper = script[script.index("function Emit-NewLogLines {"):script.index("\n$outOff = 0")]
        command = "$ErrorActionPreference = 'Stop'\n" + helper + r'''
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$path = $env:LEONIDA_TEST_LOG
$offset = 0
$writer = [IO.File]::Open($path, 'Create', 'Write', 'ReadWrite')
try {
    $first = [Text.Encoding]::UTF8.GetBytes("ready`r`n`npart " + [char]0x6771)
    $writer.Write($first, 0, $first.Length - 1)  # Stop inside the UTF-8 character.
    $writer.Flush()
    Emit-NewLogLines -Offset ([ref]$offset) -Path $path
    'poll 2'
    Emit-NewLogLines -Offset ([ref]$offset) -Path $path
    $writer.WriteByte($first[-1])
    $second = [Text.Encoding]::UTF8.GetBytes(" end`r")
    $writer.Write($second, 0, $second.Length)
    $writer.Flush()
    'poll 3'
    Emit-NewLogLines -Offset ([ref]$offset) -Path $path
    $last = [Text.Encoding]::UTF8.GetBytes("`nnext`ntrailing " + [char]0x4eac)
    $writer.Write($last, 0, $last.Length)
    $writer.Flush()
    'poll 4'
    Emit-NewLogLines -Offset ([ref]$offset) -Path $path
} finally {
    $writer.Dispose()
}
'final'
Emit-NewLogLines -Offset ([ref]$offset) -Path $path -Final
'drained'
Emit-NewLogLines -Offset ([ref]$offset) -Path $path -Final
'''
        encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                env=dict(os.environ, LEONIDA_TEST_LOG=str(Path(directory) / "build.log")),
                capture_output=True, encoding="utf-8", timeout=30,
            )
        self.assertEqual(result.stdout.splitlines(), [
            "ready", "", "poll 2", "poll 3", "poll 4", "part 東 end", "next",
            "final", "trailing 京", "drained",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
