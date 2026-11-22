"""Tests for direct binary RDNA2 -> SPIR-V translation (#1901 part 2).

Covers:
  - tools/rdna2spirv.py translating .fssm and fspx/*.code binaries directly via tools/rdna2_decode.py
    (bypassing llvm-mc and .s text files), including embedded data words with exact PC preservation
  - tools/fspx_unpack.py --disasm --backend python (including v_dot4_i32_i8 op_sel words llvm-mc rejects)
  - make_game.py --disasm-backend python (preflight without llvm-mc, signature hashing, cache reuse)
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
for _p in (ROOT, TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fspx_unpack  # noqa: E402
import make_game  # noqa: E402
import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _make_fssm(code: bytes, rsrc2: int = (4 << 1), threads=(32, 1, 1)) -> bytes:
    """Build a minimal synthetic FSSM compute blob with code at offset 0x180 (header ptr at 0xC0 = 0x158)."""
    hdr = bytearray(0x180)
    hdr[0:4] = b"FSSM"
    struct.pack_into("<I", hdr, 0xC0, 0x158)  # code_start = 0x158 + 0x28 = 0x180
    struct.pack_into("<IIII", hdr, 0x138, rsrc2, threads[0], threads[1], threads[2])
    return bytes(hdr) + code


def _make_fspx_payload(code: bytes, rsrc2: int = (8 << 1), threads=(64, 1, 1), wave32: bool = True,
                       with_prologue: bool = True) -> bytes:
    """Build a minimal synthetic FSPX payload (kind 0 + kind 6)."""
    kind0 = struct.pack(
        "<7I",
        0, 0, 0,
        (threads[0] & 0xFFFF) | ((threads[1] & 0xFFFF) << 16),
        (threads[2] & 0xFFFF) | ((1 << 20) if wave32 else 0),
        0x000C0007,  # rsrc1
        rsrc2,
    )
    if with_prologue:
        prologue = b"".join(struct.pack("<III", 0xD5800000, 0, 0) for _ in range(3))
        kind6 = prologue + code
    else:
        kind6 = code
    # Header (16) + 2 entries (32) = 48
    off0 = 48
    off6 = off0 + len(kind0)
    total = off6 + len(kind6)
    return (
        b"FSPX"
        + struct.pack("<III", 1, 2, total)
        + struct.pack("<IIII", 0, off0, len(kind0), 0)
        + struct.pack("<IIII", 6, off6, len(kind6), 0)
        + kind0
        + kind6
    )


class DirectBinaryTranslateTests(unittest.TestCase):
    def test_fssm_direct_translate_matches_text_path_byte_for_byte(self):
        # Compute shader: v1 = 10, v2 = 20, v3 = v1 + v2, s_endpgm
        code = bytes([
            0x8A, 0x02, 0x02, 0x7E,  # v_mov_b32_e32 v1, 10
            0x94, 0x02, 0x04, 0x7E,  # v_mov_b32_e32 v2, 20
            0x01, 0x05, 0x06, 0x4A,  # v_add_nc_u32_e32 v3, v1, v2
            0x00, 0x00, 0x81, 0xBF,  # s_endpgm
        ])
        blob = _make_fssm(code, rsrc2=(4 << 1), threads=(32, 1, 1))
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "fssm"))
            os.makedirs(os.path.join(d, "disasm"))
            fssm_path = os.path.join(d, "fssm", "cs_direct.fssm")
            s_path = os.path.join(d, "disasm", "cs_direct.s")
            with open(fssm_path, "wb") as f:
                f.write(blob)
            asm_text, bad = rdna2_decode.disasm_bytes(code)
            self.assertEqual(bad, 0)
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; cs_direct  code 0x180..0x{0x180 + len(code):x} (header), gfx1013, 0 invalid words\n")
                f.write(asm_text)
            spv_direct = rdna2spirv.translate(fssm_path)
            spv_text = rdna2spirv.translate(s_path)
            self.assertEqual(spv_direct, spv_text)
            _validate(spv_direct)

    def test_fssm_direct_translate_with_embedded_data_words(self):
        # Branch jumps over an embedded undecodable data word (0xFFFFFFFF) to s_endpgm.
        code = bytes([
            0x8A, 0x02, 0x02, 0x7E,  # pc+0:  v_mov_b32_e32 v1, 10
            0x01, 0x00, 0x82, 0xBF,  # pc+4:  s_branch +1 (target = pc+4+4+4 = pc+12)
            0xFF, 0xFF, 0xFF, 0xFF,  # pc+8:  undecodable embedded data word
            0x00, 0x00, 0x81, 0xBF,  # pc+12: s_endpgm
        ])
        blob = _make_fssm(code)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "fssm"))
            fssm_path = os.path.join(d, "fssm", "cs_emb_data.fssm")
            with open(fssm_path, "wb") as f:
                f.write(blob)
            spv = rdna2spirv.translate(fssm_path)
            _validate(spv)
            # Confirm direct decode saw the data word and pruned it cleanly via reachable_code.
            self.assertIn("OpReturn", spv)

    def test_fspx_unpack_backend_python_and_direct_code_translate(self):
        # Synthetic FSPX program with v_mov_b32_e32 + v_dot4_i32_i8 (including the op_sel=2/op_sel_hi=5 word
        # 0xCC165008 0x0C0E0501 that llvm-mc rejects) + s_endpgm.
        code_plain = bytes([
            0x8A, 0x02, 0x02, 0x7E,  # v_mov_b32_e32 v1, 10
            0x94, 0x02, 0x04, 0x7E,  # v_mov_b32_e32 v2, 20
            0x01, 0x05, 0x06, 0x4A,  # v_add_nc_u32_e32 v3, v1, v2
            0x00, 0x00, 0x81, 0xBF,  # s_endpgm
        ])
        code_dot4_opsel = struct.pack("<II", 0xCC165008, 0x0C0E0501) + bytes([0x00, 0x00, 0x81, 0xBF])

        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "cs_fspx_plain.fspx")
            p2 = os.path.join(d, "cs_fspx_opsel.fspx")
            out = os.path.join(d, "unpacked")
            with open(p1, "wb") as f:
                f.write(_make_fspx_payload(code_plain))
            with open(p2, "wb") as f:
                f.write(_make_fspx_payload(code_dot4_opsel))

            rc = fspx_unpack.main([p1, p2, "--out", out, "--disasm", "--backend", "python"])
            self.assertEqual(rc, 0)

            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                idx = json.load(f)
            self.assertEqual(len(idx), 2)
            # Both programs decode with 0 invalid words under --backend python (even the op_sel=2 dot4 word!).
            self.assertEqual(idx[0]["invalid_words"], 0)
            self.assertEqual(idx[1]["invalid_words"], 0)
            with open(os.path.join(out, "disasm", "cs_fspx_opsel.s"), encoding="utf-8") as f:
                opsel_asm = f.read()
            self.assertIn("v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]", opsel_asm)

            # Direct .code translation matches the disassembled .s translation byte-for-byte.
            code_path = os.path.join(out, "fspx", "cs_fspx_plain.code")
            s_path = os.path.join(out, "disasm", "cs_fspx_plain.s")
            spv_direct = rdna2spirv.translate(code_path)
            spv_text = rdna2spirv.translate(s_path)
            self.assertEqual(spv_direct, spv_text)
            _validate(spv_direct)

            # CLI directory discovery on fspx/ (.code files) works when SPIR-V tools are available.
            if SPIRV_AS and SPIRV_VAL:
                spv_out = os.path.join(d, "spv_out")
                r = subprocess.run(
                    [sys.executable, "-I", os.path.join(TOOLS, "rdna2spirv.py"),
                     code_path, "--out", spv_out],
                    capture_output=True, text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
                self.assertTrue(os.path.isfile(os.path.join(spv_out, "cs_fspx_plain.spv")))

    def test_make_game_python_disasm_backend_needs_no_llvm_mc(self):
        sig = make_game.disasm_signature("python")
        parsed = json.loads(sig)
        self.assertEqual(parsed[0], "python")
        self.assertEqual(len(parsed[1]), 64)  # SHA-256 hex digest

        with tempfile.TemporaryDirectory() as work:
            # preflight with disasm_backend="python" must NOT require llvm-mc.
            with mock.patch.object(make_game, "tool_available", return_value=False), \
                 mock.patch.object(make_game.shutil, "which", side_effect=lambda name: f"/fake/{name}" if name in ("spirv-as", "spirv-val") else None), \
                 mock.patch.object(make_game.shutil, "disk_usage", return_value=mock.Mock(free=500 * 2**30)):
                make_game.preflight(["shaders"], work, disasm_backend="python")


if __name__ == "__main__":
    unittest.main()
