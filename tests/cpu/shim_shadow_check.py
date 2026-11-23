"""Compare two submitted shadow frames from one native, mapped and translated EXE."""
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh_shadow_check as shadow
from shim_startup_check import EXPECTED, SHIM_IMPORTS, check


def build(directory, dxc):
    shadow.compile_stage(dxc, directory, "mesh",
                         shadow.lower(shadow.enumerate_pcs(shadow.mesh_program()), shadow.WORDS, (3, 1), shadow.HEADER),
                         "ms_6_5", ["-DTRACE_EXPORTS"])
    shadow.compile_stage(dxc, directory, "caster", shadow.translate_caster(shadow.caster_program()))
    words = shadow.sampler_words("LESS", "point")
    shadow.compile_stage(dxc, directory, "receiver", shadow.translate_receiver(shadow.receiver_program(words), words))
    moved = tuple((x + 4, y) for x, y in shadow.CASTER)
    lights = [shadow.light_casters(shadow.LIGHT_SHEAR, True, corners) for corners in (shadow.CASTER, moved)]
    maps = [shadow.shadow_map(casters) for casters in lights]
    draws = [shadow.draw_inputs([(*point, depth) for point in triangle])
             for casters in lights for triangle, depth in casters]
    draws += shadow.camera_draws() + shadow.camera_draws(moved)
    state = shadow.comparison_sampler_state(words)
    header = struct.pack("<8I", shadow.SIDE, len(lights[0]), len(lights[1]),
                         state["d3d12_filter"], state["d3d12_comparison_func"], 0, 0, 0)
    (directory / "frames.case").write_bytes(header + b"".join(data for data, _ in draws)
                                            + b"".join(trace for _, trace in draws))
    first = shadow.scene_image(maps[0])[0]
    second = shadow.scene_image(maps[1], corners=moved)[0]
    stale = shadow.scene_image(maps[0], corners=moved)[0]
    depths = b"".join(struct.pack("<f", float(value)) for frame in maps for value in frame)
    mismatch = shadow.differences(second, stale)
    if not mismatch or maps[0] == maps[1]:
        raise AssertionError("moving the caster must change depth and reveal the stale binding")
    return first + second + depths, first + stale + depths, mismatch


def run(directory, dxc, mapper, exe, lifted, companion, io, xmem, d3d12):
    current, stale, mismatch = build(directory, dxc)
    mapped = [sys.executable, "-B", Path(__file__).with_name("shim_startup_check.py"), "--map",
              mapper, exe, companion, io, xmem, d3d12, ""]
    markers = sorted(f"bound {dll}!{symbol}" for dll, symbols in SHIM_IMPORTS.items() for symbol in symbols)
    for case, expected in (("shadow", current), ("shadow-stale", stale)):
        env = dict(os.environ, X2W_SHIM_DRAW_CASE=case, X2W_SHIM_COMPUTE_CASE="", X2W_SHIM_CHAIN_DIR=str(directory))
        for runner, command in (("native", [exe]), ("pe_map", mapped), ("AOT", [lifted])):
            child = subprocess.run(command, env=env, capture_output=True, timeout=120)
            lines = child.stderr.decode(errors="replace").splitlines()
            if child.returncode == 77 and case == "shadow" and runner == "native":
                print("skip:", child.stdout.decode(errors="replace"), lines)
                return 77
            check(f"{case} {runner} diagnostics", sorted(lines),
                  sorted([*markers, "mapped host returned"]) if runner == "pe_map" else [])
            check(f"{case} {runner} exit", child.returncode, 42)
            check(f"{case} {runner} startup", child.stdout[:len(EXPECTED.output)], EXPECTED.output)
            payload = child.stdout[len(EXPECTED.output):]
            check(f"{case} {runner} frame payload length", len(payload), len(expected) + 1)
            check(f"{case} {runner} tracked surface releases", payload[-1], 4)
            check(f"{case} {runner} readback mismatch count", shadow.differences(payload[:-1], expected), 0)
            check(f"{case} {runner} differences from fresh depth binding",
                  shadow.differences(payload[:-1], current), mismatch if case == "shadow-stale" else 0)
            print(f"{case} {runner}: two color/depth frames match; releases=4 exit=42")
    print(f"stale map: {mismatch} second-frame color pixels differ; light depth still matches")
    return 0


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        sys.exit(run(Path(folder), *map(Path, sys.argv[1:-1])))
