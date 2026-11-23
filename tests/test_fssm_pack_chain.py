"""runtime/test/lanterns/fssm_pack.py --translate runs the project's offline chain (llvm-mc, fssm_unpack.py,
shader_disasm.py, rdna2spirv.py) on what it packs, as make_game.py's shaders stage does for the title's containers, and
the translated index is keyed by the packed program. Needs llvm-mc (AMDGPU target) and spirv-tools, like
tests/test_translate.py; skips itself without them. No GPU, no game files: each program is a few hand-written
instructions.

    python -m pytest tests/test_fssm_pack_chain.py
"""
import hashlib
import json
import os
import struct
import subprocess
import sys

import pytest

from conftest import LLVM_MC, ROOT, requires_tools

pytestmark = requires_tools

PACK = os.path.join(ROOT, "runtime", "test", "lanterns", "fssm_pack.py")


def pack_and_translate(tmp_path, name, asm, *flags):
    """Pack <name>.asm with --translate; returns the out folder and its spirv/index.json."""
    src = tmp_path / f"{name}.asm"
    src.write_text(asm, encoding="utf-8")
    out = tmp_path / "out"
    env = dict(os.environ)
    env.setdefault("LLVM_MC", subprocess.list2cmdline(LLVM_MC) if LLVM_MC else "llvm-mc")
    r = subprocess.run([sys.executable, "-I", PACK, str(out), str(src), "--translate", *flags],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return out, json.loads((out / "spirv" / "index.json").read_text(encoding="utf-8"))


def test_translated_index_is_keyed_by_the_packed_blob(tmp_path):
    asm = ("; stage: ps\n; wave: 32\n; input: flat\n; data-at: 0x40\n; data: 0x3f000000\n"
           "v_interp_mov_f32 v2, p0, attr0.x\nv_mov_b32_e32 v3, 1.0\n"
           "exp mrt0 v2, v3, v3, v3 done vm\ns_endpgm\n")
    out, index = pack_and_translate(tmp_path, "ps_chain", asm)
    blob = (out / "fssm" / "ps_chain.fssm").read_bytes()
    row = index["ps_chain"]
    assert row["fssm_sha256"] == hashlib.sha256(blob).hexdigest()
    assert (row["stage"], row["ps_flat_interpolants"]) == ("ps", [0])



def test_slot_byte_and_first_payload_do_not_move_the_shader(tmp_path):
    # The input's slot byte (9) is not its attr number (0), the tail has a binding record, and the container's first
    # 0x0D payload is another program: the chain keys the shader by the last payload and declares its flat input at
    # table position 0. A reader that took the slot or the first payload would give another index row.
    asm = ("; stage: ps\n; input: flat slot=9\n; binding: 1 2 3 4 5\n; first-payload: ps_first\n"
           "v_interp_mov_f32 v2, p0, attr0.x\nv_mov_b32_e32 v3, 1.0\nexp mrt0 v2, v3, v3, v3 done vm\ns_endpgm\n")
    out, index = pack_and_translate(tmp_path, "ps_two", asm)
    blob = (out / "fssm" / "ps_two.fssm").read_bytes()
    assert index["ps_two"]["fssm_sha256"] == hashlib.sha256(blob).hexdigest()
    assert index["ps_two"]["ps_flat_interpolants"] == [0]
    assert "ps_first" not in index


@pytest.mark.parametrize("header, system_inputs", [("; stage: ps\n; position: xy\n", ["0", "1"]),
                                                   ("; stage: ps\n", None)])
def test_position_header_lets_a_pixel_shader_read_sv_position(tmp_path, header, system_inputs):
    # With the ISG1 part v2 and v3 are FragCoord.x and .y; without it the translator takes them for registers
    # nothing wrote (zeros), and the shader would read 0 for every pixel.
    body = "v_cvt_u32_f32_e32 v4, v2\nv_cvt_u32_f32_e32 v5, v3\nexp mrt0 v4, v5, v4, v5 done vm\ns_endpgm\n"
    _, index = pack_and_translate(tmp_path, "ps_pos", header + body)
    assert index["ps_pos"].get("ps_system_inputs") == system_inputs


def test_branches_to_labels_are_resolved(tmp_path):
    # A loop (backward s_cbranch_scc0) and a jump over one instruction (forward s_branch), both to labels. The packed
    # code must carry the offsets llvm-mc resolves, and --translate decodes it from the header with 0 invalid words.
    asm = ("; stage: ps\n"
           "s_mov_b32 s0, 2\n"                 # 0x00
           "loop:\n"
           "s_sub_u32 s0, s0, 1\n"              # 0x04
           "s_cmp_eq_u32 s0, 0\n"               # 0x08
           "s_cbranch_scc0 loop\n"              # 0x0c: back to 0x04, (0x04 - 0x10) / 4 = -3
           "s_branch done\n"                    # 0x10: on to 0x18, (0x18 - 0x14) / 4 = 1
           "v_mov_b32_e32 v0, 0\n"              # 0x14, jumped over
           "done:\n"
           "v_mov_b32_e32 v1, 1.0\n"            # 0x18
           "exp mrt0 v1, v1, v1, v1 done vm\n"
           "s_endpgm\n")
    out, index = pack_and_translate(tmp_path, "ps_branch", asm)
    blob = (out / "fssm" / "ps_branch.fssm").read_bytes()
    code_at = struct.unpack_from("<Q", blob, 0xC0)[0] + 0x28
    words = struct.unpack_from("<6I", blob, code_at)
    assert (words[3], words[4]) == (0xBF84FFFD, 0xBF820001)    # s_cbranch_scc0 -3, s_branch +1
    assert index["ps_branch"]["fssm_sha256"] == hashlib.sha256(blob).hexdigest()


def test_native_variant_is_translated_without_emulation(tmp_path):
    asm = ("; stage: cs\n; wave: 64\n; group: 64,1,1\n; rsrc2: 0x8c\n; native-variant: cs_w_native\n"
           "v_cmp_eq_u32_e32 vcc, 0, v0\ns_bcnt1_i32_b64 s8, vcc\ns_endpgm\n")
    _, index = pack_and_translate(tmp_path, "cs_w", asm, "--wave64-emulate")
    assert index["cs_w"]["fssm_sha256"] != index["cs_w_native"]["fssm_sha256"]
    assert index["cs_w"]["wave64_exchange"] == "workgroup"
    assert (index["cs_w_native"].get("wave64_exchange"), index["cs_w_native"]["subgroup_size"]) == (None, 64)
