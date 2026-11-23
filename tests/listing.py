"""Hand-written ``.s`` listings for translator tests that need neither llvm-mc nor spirv-tools."""
import re
import struct

import rdna2spirv

EMPTY_DXBC = b"DXBC" + bytes(16) + struct.pack("<III", 1, 32, 0)
EPILOGUE = "v_mov_b32 v8, v3\nexp mrt0 v8, v8, v8, v8 done vm\ns_endpgm"


def translate_listing(tmp_path, body, name="ps_listing"):
    """Translate pixel-shader ``body`` (assembly text, one instruction per line) and return the SPIR-V assembly.

    The first line of a listing is its header (shader_ir.parse_file skips it), so one is written here. Pixel shaders
    must not read v0/v1 (barycentrics) before writing them. v3 is the register the epilogue exports.
    """
    (tmp_path / "disasm").mkdir(exist_ok=True)
    (tmp_path / f"{name}.dxbc").write_bytes(EMPTY_DXBC)
    lines = [f"; {name} code 0x0..0x100"] + [ln.strip() for ln in (body + "\n" + EPILOGUE).splitlines() if ln.strip()]
    path = tmp_path / "disasm" / f"{name}.s"
    path.write_text(lines[0] + "\n" + "".join("\t" + ln + "\n" for ln in lines[1:]))
    return rdna2spirv.translate(str(path))


def check_written(case, spvasm, dest, value):
    """Assert that the VGPR word `dest` receives exactly `value`.

    Gen.dst() emits the write in one of two shapes, and both leave that same word in the register:

    * ``OpSelect %uint <mask> value <old>`` followed by ``OpStore dest <select>``, when a mask is live and the
      other lanes' bits of the register have to survive the write (``<old>`` is ``OpLoad``ed from `dest`), and
    * a direct ``OpStore dest value``, when the write already sits under a guard: an enclosing ``if (active)``
      run, or a shader that never writes EXEC, where %active is constant true and merging it is a no-op.

    `value` is the id the caller computed from the instruction's modifier chain, so omod (``OpFMul``) and clamp
    (``OpExtInst ... FClamp``) stay pinned by the caller; this accepts either way of landing that value in the
    register, and a merge is still required to consume the old word loaded from the same register. The per-lane
    result of both shapes is checked by the shader tests under tests/semantic/.
    """
    dest, value = re.escape(dest), re.escape(value)
    old_word = re.search(rf"(?m)^ *(%\w+) = OpLoad %uint {dest}$", spvasm)
    merged = (re.search(rf"(?m)^ *(%\w+) = OpSelect %uint %\w+ {value} {old_word.group(1)}$", spvasm)
              if old_word else None)
    if merged is not None:
        case.assertRegex(spvasm, rf"(?m)^ *OpStore {dest} {merged.group(1)}$")
    else:
        case.assertRegex(spvasm, rf"(?m)^ *OpStore {dest} {value}$", f"{dest} never receives {value}")
