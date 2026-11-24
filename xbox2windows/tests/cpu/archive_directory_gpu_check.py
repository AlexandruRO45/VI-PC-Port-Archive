"""Run a compiler-built RDS5 directory reader, its mapped PE, and its AOT image on the same provider."""
from collections import defaultdict
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startup_storage_directory_batch import DISTANT, PAGE, SHIFT, ranges, write_compressed_fixture
from startup_xg_bc_chain_check import block


def solid_pages():
    groups = defaultdict(list)
    for red in range(2, 252, 2):
        color = bytes((red, 16, 48, 254))
        page = block(98, color) * 256
        stream = zlib.compress(page, 9)
        groups[len(stream)].append((color, page, stream))
    # Only changing the content distinguishes the shifted first record.
    group = max(groups.values(), key=len)
    different_length = next(items[0] for items in groups.values() if len(items[0][2]) != len(group[0][2]))
    return group[0], different_length, group[-1]


def check(mapper, native, lifted, provider):
    first, second, decoy = solid_pages()
    mapped = [sys.executable, "-B", Path(__file__).with_name("archive_check.py"), "--mapped", mapper, native, provider]
    commands = {"native": [native], "pe_map": mapped, "AOT": [lifted]}
    adapter = None
    env = dict(os.environ, X2W_XMEM_PERMISSIVE="1")
    with tempfile.TemporaryDirectory() as workspace:
        for records in (1, 16384):
            case = Path(workspace) / str(records)
            case.mkdir()
            path = case / "fixture.ttpf"
            write_compressed_fixture(path, records, [p[2] for p in (first, second, decoy)])
            directory = ranges(path)
            (base, length), (next_pos, next_length) = directory[:2]
            assert directory[DISTANT] == (base + SHIFT, length) and next_length != length
            for mode, bias, selection in (("correct", 0, first), ("shifted", SHIFT, decoy),
                                          ("EOF", path.stat().st_size - base, None)):
                for runner, command in commands.items():
                    folder = case / (mode + "-" + runner)
                    folder.mkdir()
                    shutil.copyfile(path, folder / path.name)
                    child = subprocess.run(command, input=struct.pack("<Q", bias), cwd=folder, env=env,
                                           capture_output=True, timeout=12)
                    if child.returncode == 77 and runner == "native" and env.get("X2W_SHIM_DRAW_ADAPTER") == "hardware":
                        return 77
                    assert child.returncode == (0 if selection else 6), (runner, mode, child.returncode, child.stderr)
                    trace = (folder / "ranges.txt").read_text().splitlines()
                    expected = [f"request {base+bias} {length}", f"request {next_pos} {next_length}",
                                "status 00000000" if selection else "status 80004005", "status 00000000"]
                    assert trace[:4] == expected, (runner, mode, trace)
                    page = selection[1] if selection else b"\xa5" * PAGE
                    assert (folder / "pages.bin").read_bytes() == page + second[1], (runner, mode, "decoded blocks")
                    rendered = folder / "pixels.bin"
                    if selection:
                        assert trace[4].startswith("adapter ")
                        if adapter is None:
                            adapter = trace[4]
                        assert trace[4:] == [adapter, "fence 1", "release 0"], (runner, mode, trace[4:])
                        pixels = rendered.read_bytes()
                        assert pixels == (selection[0] * (128 * 64) + second[0] * (128 * 64)), (runner, mode, "RGBA")
                    else:
                        assert len(trace) == 4 and not rendered.exists()
                print(f"{records=} {mode}: native, mapped and AOT matched both ranges, statuses, pages and exit")
    print(adapter, "8192/16384 pixels change when only the first offset is shifted")
    return 0


if __name__ == "__main__":
    sys.exit(check(*[Path(arg).resolve() for arg in sys.argv[1:]]))
