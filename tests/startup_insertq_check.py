"""Compare the normal Windows launch with the optional startup instruction fallback."""
import os
from pathlib import Path
import struct
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "tools/startup"))
from insertq import apply_insertq

exe = Path(sys.argv[1]).resolve()
logs = Path(sys.argv[2]).resolve()
# Exercise rejects and defined boundary values even on hosts which execute SSE4a natively.
rip = 0x123456789
for width, offset, expected in [(16, 16, 0x11223344eeff7788), (0, 0, 0x9900aabbccddeeff),
                                (1, 63, 0x9122334455667788)]:
    ctx = bytearray(b"\xa5" * 1232)
    struct.pack_into("<Q", ctx, 0xf8, rip)
    struct.pack_into("<Q", ctx, 0x1a0, 0x1122334455667788)
    struct.pack_into("<Q", ctx, 0x1b0, 0x9900aabbccddeeff)
    want = ctx.copy()
    struct.pack_into("<Q", want, 0xf8, rip + 6)
    struct.pack_into("<Q", want, 0x1a0, expected)
    assert apply_insertq(bytes.fromhex("f20f78c1") + bytes((width, offset)), ctx, rip)
    assert ctx == want
for bad in ["f20f78", "f20f78001010", "f20f79c11010", "f20f78c14001", "664578c11010", "f20f78c12030"]:
    ctx = bytearray(1232)
    struct.pack_into("<Q", ctx, 0xf8, rip)
    saved = ctx.copy()
    assert not apply_insertq(bytes.fromhex(bad), ctx, rip) and ctx == saved

def launch(enabled):
    run = logs / ("enabled" if enabled else "baseline")
    subprocess.run([sys.executable, "-B", str(root / "tools/startup/debug_launch.py"), str(exe),
                    str(exe.parent), str(run), "--offline", "0", "--console", "0", "--timeout", "10"],
                   check=True, env=dict(os.environ, X2W_STARTUP_INSERTQ=str(int(enabled))))
    return (run / "finished.txt").read_text(encoding="utf-8")

baseline = launch(False)
translated = launch(True)
assert "process exited with 0x00000000" in translated, translated
if "process exited with 0x00000000" in baseline:
    assert "INSERTQ emulated" not in translated
else:
    assert "stopped at illegal instruction (0xc000001d)" in baseline, baseline
    assert "immediate INSERTQ emulated: 1" in translated
