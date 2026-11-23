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
                     os.path.join(out, "fspx"), "--out", spv_out],
                    capture_output=True, text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
                self.assertTrue(os.path.isfile(os.path.join(spv_out, "cs_fspx_plain.spv")))

    def test_direct_mcpu_matches_text_path_for_vop3_0x140(self):
        # VOP3 opcode 0x140 is v_mad_legacy_f32 on gfx1013 (.fssm) and v_fma_legacy_f32 on gfx1030 (.code).
        w0 = (0x35 << 26) | (0x140 << 16) | 1
        w1 = (256 + 1) | ((256 + 2) << 9) | ((256 + 3) << 18)
        vop3_0x140 = struct.pack("<II", w0, w1)
        endpgm = bytes([0x00, 0x00, 0x81, 0xBF])
        # Plain stream and stream with an unreachable data word jumped over by s_branch +1 (exercises decode_reachable_direct).
        plain = vop3_0x140 + endpgm
        with_data = bytes([0x01, 0x00, 0x82, 0xBF]) + bytes([0xFF, 0xFF, 0xFF, 0xFF]) + vop3_0x140 + endpgm

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "fssm"))
            for label, code in (("plain", plain), ("reachable", with_data)):
                p = os.path.join(d, "fssm", f"cs_0x140_{label}.fssm")
                with open(p, "wb") as f:
                    f.write(_make_fssm(code))
                spv = rdna2spirv.translate(p)
                _validate(spv)
                self.assertIn("OpReturn", spv)

            for label, code in (("plain", plain), ("reachable", with_data)):
                fspx_path = os.path.join(d, f"cs_fspx_0x140_{label}.fspx")
                out = os.path.join(d, f"unpacked_{label}")
                with open(fspx_path, "wb") as f:
                    f.write(_make_fspx_payload(code))
                self.assertEqual(fspx_unpack.main([fspx_path, "--out", out, "--disasm", "--backend", "python"]), 0)
                code_path = os.path.join(out, "fspx", f"cs_fspx_0x140_{label}.code")
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    rdna2spirv.translate(code_path)
                self.assertIn("v_fma_legacy_f32", str(ctx.exception))

    def test_code_metadata_wave32_false_and_overlap_rejection(self):
        # .code honours "wave32": false from its sidecar .json.
        code = bytes([0x8A, 0x02, 0x02, 0x7E, 0x00, 0x00, 0x81, 0xBF])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cs_w64.fspx")
            out = os.path.join(d, "unpacked")
            with open(p, "wb") as f:
                f.write(_make_fspx_payload(code, wave32=False))
            self.assertEqual(fspx_unpack.main([p, "--out", out]), 0)
            info = rdna2spirv.decode_binary_file(os.path.join(out, "fspx", "cs_w64.code"))
            self.assertFalse(info["wave32"])
            self.assertEqual(info["mcpu"], "gfx1030")

        # decode_reachable_direct rejects a branch landing inside an 8-byte instruction.
        # pc+0: s_cbranch_scc1 +1 -> targets pc+8 (middle of the 8-byte VOP3 starting at pc+4).
        vop3_nop = struct.pack("<II", (0x35 << 26) | (0x140 << 16) | 1, (256 + 1) | ((256 + 2) << 9) | ((256 + 3) << 18))
        bad_overlap = (
            bytes([0x01, 0x00, 0x85, 0xBF])  # pc+0: s_cbranch_scc1 +1 -> pc+8
            + vop3_nop                        # pc+4..11: 8-byte VOP3 (w1 at pc+8 is non-zero)
            + bytes([0x00, 0x00, 0x81, 0xBF])
        )
        with self.assertRaises(rdna2spirv.Unsupported):
            rdna2spirv.decode_reachable_direct(bad_overlap, 0, True, "gfx1013")

    def test_python_disasm_signature_tracks_file_contents(self):
        sig1 = json.loads(make_game.python_disasm_signature())
        self.assertEqual(sig1[0], "python")
        self.assertEqual(len(sig1[1]), 64)
        with tempfile.TemporaryDirectory() as d:
            for fname, body in (("rdna2_decode.py", b"v1"), ("rdna2_isa_tables.py", b"t1")):
                with open(os.path.join(d, fname), "wb") as f:
                    f.write(body)
            with mock.patch.object(make_game, "TOOLS", d):
                a = make_game.python_disasm_signature()
                with open(os.path.join(d, "rdna2_decode.py"), "wb") as f:
                    f.write(b"v2")
                b = make_game.python_disasm_signature()
        self.assertNotEqual(a, b)

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

    def test_direct_translate_fails_closed_on_reachable_unknown_vop3p(self):
        # Reachable unknown VOP3P opcode (8-byte and 12-byte with literal) must raise Unsupported rather than
        # decoding w1 as a separate instruction or silently truncating the block.
        endpgm = bytes([0x00, 0x00, 0x81, 0xBF])
        mov_v1 = bytes([0x8A, 0x02, 0x02, 0x7E])
        cases = (
            ("8byte_entry", struct.pack("<II", 0xCC7F0000, 0x7E02028A) + endpgm),
            ("8byte_mid", mov_v1 + struct.pack("<II", 0xCC7F0000, 0x7E02028A) + endpgm),
            ("12byte_lit", mov_v1 + struct.pack("<III", 0xCC7F0000, 0x000000FF, 0x7E02028A) + endpgm),
        )
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "fssm"))
            for label, code in cases:
                with self.subTest(case=label):
                    p = os.path.join(d, "fssm", f"cs_{label}.fssm")
                    with open(p, "wb") as f:
                        f.write(_make_fssm(code))
                    with self.assertRaises(rdna2spirv.Unsupported):
                        rdna2spirv.translate(p)

    def test_direct_translate_skips_unreachable_data_resembling_vop3p(self):
        # An unreachable embedded data word (0xCC7F0000, resembling an 8-byte VOP3P prefix) jumped over by
        # s_branch +1 must NOT swallow s_endpgm 4 bytes later: recursive-descent decode skips the data word cleanly.
        code = (
            bytes([0x8A, 0x02, 0x02, 0x7E])    # pc+0:  v_mov_b32_e32 v1, 10
            + bytes([0x01, 0x00, 0x82, 0xBF])  # pc+4:  s_branch +1 -> pc+12
            + struct.pack("<I", 0xCC7F0000)    # pc+8:  unreachable data word with VOP3P prefix
            + bytes([0x00, 0x00, 0x81, 0xBF])  # pc+12: s_endpgm
        )
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "fssm"))
            p = os.path.join(d, "fssm", "cs_emb_vop3p_data.fssm")
            with open(p, "wb") as f:
                f.write(_make_fssm(code))
            spv = rdna2spirv.translate(p)
            _validate(spv)
            self.assertIn("OpReturn", spv)


if __name__ == "__main__":
    unittest.main()
