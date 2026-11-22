"""Run the translated compute -> mesh -> pixel chain from the shim EXE's PE entry.

The EXE's draw phase (draw_request / draw_frame) selects one of three chain cases.
For each case, the draw provider runs the #658 sequence on one direct queue:
translated gfx1013 compute producers write a 2x2 texture, and translated indexed
mesh and image_sample_lz pixel stages sample it into a 32x32 frame. The provider
publishes the frame and the mesh export trace after the startup output.

chain0 dispatches palette A and binds it. chain1 dispatches A, then B, and binds B.
chain-stale dispatches both and binds the earlier texture (A) against B's reference.
The native EXE, the pe_map mapping and the AOT build must agree on the request,
the exit status and every published byte. Every RGBA byte and the trace must
match a reference computed here without D3D12. The stale frame must differ from
B's reference at exactly the covered pixels whose palette entry changed, while
every clear pixel and the trace still match.
"""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compute_mesh_texture_check import producer  # noqa: E402
from mesh_entry_check import enumerate_pcs, fixture  # noqa: E402
from mesh_indexed_check import HEADER, WORDS, lower  # noqa: E402
from mesh_textured_check import PALETTE, POSITIONS, mesh_program, texture_case  # noqa: E402
from pixel_aot import lower as lower_pixel  # noqa: E402
from pixel_aot_check import compile_shader, lod_zero_sample  # noqa: E402
from shim_startup_check import EXPECTED, SHIM_IMPORTS, check  # noqa: E402

SIDE = 32
FRAME = SIDE * SIDE * 4
TRACE = 80
INPUT = 1536
STALE_MISMATCHES = 276
# case: (reference palette, producers dispatched, expected exit)
CASES = {"chain0": (0, 1, 42), "chain1": (1, 2, 42), "chain-stale": (1, 2, 20)}


def build(tmp, dxc):
    """Write the translated shaders and the two case files the provider reads."""
    mesh_source = tmp / "mesh.hlsl"
    mesh_source.write_text(lower(enumerate_pcs(mesh_program()), WORDS, (3, 1), HEADER), encoding="ascii")
    subprocess.run([dxc, "-T", "ms_6_5", "-E", "main", "-DTRACE_EXPORTS", "-Fo", tmp / "mesh.dxil",
                    mesh_source], check=True, capture_output=True)
    pixel_source = tmp / "pixel.hlsl"
    pixel_source.write_text(lower_pixel(lod_zero_sample(), 20, 24, [1, 2, 3, 4], texcoord_components=4),
                            encoding="ascii")
    compile_shader(dxc, pixel_source, tmp / "pixel.dxil")
    references = []
    for i, palette in enumerate((PALETTE, PALETTE[1:] + PALETTE[:1])):
        data, pixels, texels = texture_case(palette, False)
        source = tmp / f"palette-{i}.hlsl"
        source.write_text(producer(texels), encoding="ascii")
        subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", tmp / f"palette-{i}.dxil", source],
                       check=True, capture_output=True)
        case = fixture(data, POSITIONS, pixels, True)
        if len(case) != INPUT + TRACE + FRAME:
            raise AssertionError(f"palette {i} case has {len(case)} bytes")
        (tmp / f"palette-{i}.case").write_bytes(case)
        references.append(case[INPUT:])  # trace, then frame
    return references


def pixel_mismatches(frame, reference):
    return [i for i in range(SIDE * SIDE) if frame[4 * i:4 * i + 4] != reference[4 * i:4 * i + 4]]


def verify(case, published, references):
    selected = CASES[case][0]
    check(f"{case} publishes frame and trace", len(published), FRAME + TRACE)
    frame, trace = published[:FRAME], published[FRAME:]
    expected_trace, expected = references[selected][:TRACE], references[selected][TRACE:]
    check(f"{case} mesh export trace", trace, expected_trace)
    differing = pixel_mismatches(frame, expected)
    if case != "chain-stale":
        check(f"{case} RGBA bytes match the reference", differing, [])
        return 0
    # The stale texture is palette A, so the frame must be exactly A's reference.
    check("stale frame is palette A's frame", frame, references[0][TRACE:])
    clear = {i for i in range(SIDE * SIDE) if expected[4 * i:4 * i + 4] == bytes((0, 0, 0, 255))}
    check("stale clear pixels match", [i for i in differing if i in clear], [])
    check("stale covered pixels detected", len(differing), STALE_MISMATCHES)
    return len(clear)


def run(tmp, dxc, mapper, exe, lifted, companion, io, xmem, d3d12):
    references = build(tmp, dxc)
    map_command = [sys.executable, "-B", Path(__file__).with_name("shim_startup_check.py"), "--map",
                   mapper, exe, companion, io, xmem, d3d12, ""]
    markers = sorted(f"bound {dll}!{symbol}" for dll, symbols in SHIM_IMPORTS.items() for symbol in symbols)
    for case, (_, producers, code) in CASES.items():
        env = dict(os.environ, X2W_SHIM_DRAW_CASE=case, X2W_SHIM_COMPUTE_CASE="",
                   X2W_SHIM_CHAIN_DIR=str(tmp))
        outputs = {}
        for runner, command in (("native", [exe]), ("pe_map", map_command), ("AOT", [lifted])):
            child = subprocess.run(command, env=env, capture_output=True, timeout=120)
            lines = child.stderr.decode(errors="replace").splitlines()
            if child.returncode == 77:
                print(f"skip: {case} {runner}: {' '.join(lines)}")
                return 77
            if runner == "pe_map":
                check("mapper returned", lines[-1:], ["mapped host returned"])
                check("explicit shim bindings", sorted(lines[:-1]), markers)
            else:
                check(f"{case} {runner} diagnostics", lines, [])
            check(f"{case} {runner} startup output", child.stdout[:len(EXPECTED.output)], EXPECTED.output)
            check(f"{case} {runner} exit", child.returncode, code)
            outputs[runner] = child.stdout[len(EXPECTED.output):]
        check(f"{case} runners agree", (outputs["native"], outputs["AOT"]), (outputs["pe_map"], outputs["pe_map"]))
        clear = verify(case, outputs["native"], references)
        detail = (f"stale texture differs at {STALE_MISMATCHES} pixels; {clear} clear pixels and trace match"
                  if case == "chain-stale" else "1024/1024 pixels and trace match")
        print(f"{case}: producers={producers} exit={code} "
              f"native, pe_map and AOT agree on {FRAME + TRACE} bytes; {detail}")
    return 0


if __name__ == "__main__":
    # The draw provider path is passed so the test depends on its build; the EXE loads it from its directory.
    with TemporaryDirectory() as directory:
        sys.exit(run(Path(directory), *map(Path, sys.argv[1:-1])))
