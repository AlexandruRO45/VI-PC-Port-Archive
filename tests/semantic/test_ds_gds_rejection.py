"""Reject the `gds` (Global Data Share) modifier on all `ds_*` instructions.

RDNA2's DS encoding (opcode prefix 0x36) uses bit 17 (`GDS`) to target Global Data Share rather than
workgroup-local LDS. Vulkan/SPIR-V has no GDS equivalent, so `rdna2spirv` must reject `"gds" in mods`
across every `ds_*` lowering path (`lds_op`, `ds_permute`, `ds_bpermute_b32`, `swizzle`, `gs_ds`, and
`ds_nop`) instead of silently miscompiling them to workgroup-local `%lds`.

Tests:
- `gds` rejection across DS reads, writes, atomics, cross-lane ops, `ds_nop`, and GS `gs_ds`;
- direct unit checks on each individual `ds_*` handler method (`lds_op`, `ds_permute`, `swizzle`, `gs_ds`);
- raw binary `llvm-mc -triple=amdgcn -mcpu=gfx1030` -> `rdna2_decode.decode_bytes`/`disasm_bytes` ->
  `rdna2spirv` round-trip with DS bit 17 (`GDS`) set;
- positive controls confirming non-`gds` forms still translate, pass `spirv-val --target-env vulkan1.2`,
  and execute identically in `spirv_mini_interp`.
"""
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
from support import rdna2spirv  # noqa: E402
import rdna2_decode  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

SHADERS = os.path.join(support.ROOT, "runtime", "test", "shaders")


def _cs_fssm(wave=32, lds_blocks=8):
    """Minimal FSSM compute header: 4 user SGPRs, TID X enabled, `lds_blocks` * 512 B of LDS."""
    h = bytearray(0x148)
    rsrc2 = (4 << 1) | (lds_blocks << 15)
    struct.pack_into("<4I", h, 0x138, rsrc2, wave, 1, 1)
    return bytes(h)


def _gs_dxbc():
    if SHADERS not in sys.path:
        sys.path.insert(0, SHADERS)
    import build
    return build.gs_container()


def _llvm_mc_cmd():
    env = os.environ.get("LLVM_MC")
    if env:
        return shlex.split(env)
    exe = shutil.which("llvm-mc")
    return [exe] if exe else None


def _validate_spirv(spvasm):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        sa = os.path.join(tmp, "ds_gds.spvasm")
        sv = os.path.join(tmp, "ds_gds.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        as_res = subprocess.run(
            [spirv_as, "--target-env", "vulkan1.2", sa, "-o", sv],
            capture_output=True, text=True, check=False,
        )
        if as_res.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{as_res.stderr}{as_res.stdout}")
        val_res = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", sv],
            capture_output=True, text=True, check=False,
        )
        if val_res.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{val_res.stderr}{val_res.stdout}")


def _assemble_gfx1030(lines):
    """Assemble RDNA2 instruction lines with llvm-mc and return list of raw byte sequences."""
    mc = _llvm_mc_cmd()
    res = subprocess.run(
        mc + ["-triple=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
        input="\n".join(lines) + "\n",
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        raise AssertionError(f"llvm-mc failed:\n{res.stderr}{res.stdout}")
    chunks = []
    for raw in re.findall(r"encoding:\s*\[([^\]]+)\]", res.stdout):
        chunks.append(bytes(int(tok.strip(), 16) for tok in raw.split(",")))
    if len(chunks) != len(lines):
        raise AssertionError(f"expected {len(lines)} encodings, got {len(chunks)}:\n{res.stdout}")
    return chunks


class DsGdsRejectionTests(unittest.TestCase):
    CS_GDS_CASES = [
        # DS reads
        ("ds_read_b32", "ds_read_b32 v2, v1 gds"),
        ("ds_read_b32_off", "ds_read_b32 v2, v1 offset:16 gds"),
        ("ds_read_b64", "ds_read_b64 v[2:3], v1 gds"),
        ("ds_read2_b32", "ds_read2_b32 v[2:3], v1 offset0:0 offset1:1 gds"),
        ("ds_read_u8", "ds_read_u8 v2, v1 gds"),
        ("ds_read_addtid_b32", "ds_read_addtid_b32 v2 gds"),
        ("ds_read_addtid_b32_off", "ds_read_addtid_b32 v2 offset:16 gds"),
        # DS writes
        ("ds_write_b32", "ds_write_b32 v1, v2 gds"),
        ("ds_write_b32_off", "ds_write_b32 v1, v2 offset:16 gds"),
        ("ds_write_b64", "ds_write_b64 v1, v[2:3] gds"),
        ("ds_write2_b32", "ds_write2_b32 v1, v2, v3 offset0:0 offset1:1 gds"),
        ("ds_write_b8", "ds_write_b8 v1, v2 gds"),
        ("ds_write_addtid_b32", "ds_write_addtid_b32 v2 gds"),
        ("ds_write_addtid_b32_off", "ds_write_addtid_b32 v2 offset:16 gds"),
        # DS atomics
        ("ds_add_u32", "ds_add_u32 v1, v2 gds"),
        ("ds_add_rtn_u32", "ds_add_rtn_u32 v3, v1, v2 gds"),
        ("ds_cmpst_b32", "ds_cmpst_b32 v1, v2, v3 gds"),
        ("ds_max_f32", "ds_max_f32 v1, v2 gds"),
        ("ds_wrxchg_rtn_b32", "ds_wrxchg_rtn_b32 v3, v1, v2 gds"),
        # Cross-lane DS ops
        ("ds_swizzle_quad_perm", "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2) gds"),
        ("ds_swizzle_bitmask", 'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"01pi0") gds'),
        ("ds_swizzle_numeric", "ds_swizzle_b32 v2, v1 offset:0x801f gds"),
        ("ds_swizzle_bare", "ds_swizzle_b32 v2, v1 gds"),
        ("ds_permute_b32", "ds_permute_b32 v2, v1, v3 gds"),
        ("ds_permute_b32_off", "ds_permute_b32 v2, v1, v3 offset:4 gds"),
        ("ds_bpermute_b32", "ds_bpermute_b32 v2, v1, v3 gds"),
        ("ds_bpermute_b32_off", "ds_bpermute_b32 v2, v1, v3 offset:4 gds"),
        # DS no-op with gds bit set
        ("ds_nop", "ds_nop gds"),
    ]

    GS_PRO = [
        "s_bfe_i32 s0, s3, 0x80008",
        "s_bfm_b64 exec, s0, 0",
        "s_cmp_eq_u32 s0, 64",
        "s_cselect_b64 exec, -1, exec",
        "v_mov_b32_e32 v12, 0",
        "v_mov_b32_e32 v14, 0",
        "v_mov_b32_e32 v15, 0",
    ]
    GS_EPI = [
        "s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)",
        "s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)",
        "s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)",
        "s_endpgm",
    ]
    GS_GDS_CASES = [
        ("gs_ds_read_b32", "ds_read_b32 v4, v12 offset:0 gds"),
        ("gs_ds_read2_b32", "ds_read2_b32 v[4:5], v12 offset0:0 offset1:1 gds"),
        ("gs_ds_write_b32", "ds_write_b32 v12, v14 offset:0 gds"),
        ("gs_ds_write2_b32", "ds_write2_b32 v12, v14, v14 offset0:0 offset1:1 gds"),
        ("gs_ds_write_b64", "ds_write_b64 v12, v[14:15] offset:0 gds"),
    ]

    def test_cs_ds_gds_rejected(self):
        preamble = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, 7",
            "v_mov_b32_e32 v3, 11",
            "s_mov_b32 m0, 0",
        ]
        fssm = _cs_fssm(32)
        for label, line in self.CS_GDS_CASES:
            with self.subTest(case=label, line=line):
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    support.translate_lines(
                        f"cs_gds_{label}",
                        preamble + [line, "s_endpgm"],
                        fssm=fssm,
                    )
                self.assertIn("gds", str(ctx.exception))

    def test_gs_ds_gds_rejected(self):
        dxbc = _gs_dxbc()
        for label, line in self.GS_GDS_CASES:
            with self.subTest(case=label, line=line):
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    support.translate_lines(
                        f"gs_w64_{label}",
                        self.GS_PRO + [line] + self.GS_EPI,
                        dxbc=dxbc,
                    )
                self.assertIn("gds", str(ctx.exception))

    def test_direct_handlers_reject_gds(self):
        """Every individual ds_* helper rejects 'gds' even if invoked directly."""
        g_cs = rdna2spirv.Gen(stage="cs")
        v1 = {"k": "vgpr", "r": 1, "n": 1}
        v2 = {"k": "vgpr", "r": 2, "n": 1}
        v3 = {"k": "vgpr", "r": 3, "n": 1}

        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g_cs.lds_op("ds_read_b32", [v2, v1], ["gds"])
        self.assertIn("gds", str(ctx.exception))

        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g_cs.ds_permute([v2, v1, v3], ["gds"])
        self.assertIn("gds", str(ctx.exception))

        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g_cs.swizzle([v2, v1], ["offset:swizzle(QUAD_PERM,1,0,3,2)", "gds"], {"raw": ""})
        self.assertIn("gds", str(ctx.exception))

        g_gs = rdna2spirv.Gen(stage="gs")
        g_gs.gs = {"nreg": 1, "in": [], "out": [], "maxv": 3}
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g_gs.gs_ds("ds_write_b32", [v1, v2], ["gds"])
        self.assertIn("gds", str(ctx.exception))


@unittest.skipUnless(_llvm_mc_cmd() is not None, "needs llvm-mc or LLVM_MC")
class DsGdsBinaryRoundTripTests(unittest.TestCase):
    def test_bit17_gds_decode_and_reject(self):
        """Assemble non-gds DS ops with llvm-mc, set DS bit 17 (GDS), decode via rdna2_decode, reject in rdna2spirv."""
        base_insns = [
            "ds_read_b32 v2, v1",
            "ds_read2_b32 v[2:3], v1 offset0:0 offset1:1",
            "ds_read_u8 v2, v1",
            "ds_read_addtid_b32 v2",
            "ds_write_b32 v1, v2",
            "ds_write2_b32 v1, v2, v3 offset0:0 offset1:1",
            "ds_write_b8 v1, v2",
            "ds_write_addtid_b32 v2",
            "ds_add_u32 v1, v2",
            "ds_add_rtn_u32 v3, v1, v2",
            "ds_cmpst_b32 v1, v2, v3",
            "ds_max_f32 v1, v2",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2)",
            "ds_permute_b32 v2, v1, v3",
            "ds_bpermute_b32 v2, v1, v3",
            "ds_nop",
        ]
        encoded = _assemble_gfx1030(base_insns)
        fssm = _cs_fssm(32)
        preamble = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, 7",
            "v_mov_b32_e32 v3, 11",
            "s_mov_b32 m0, 0",
        ]

        for src, raw in zip(base_insns, encoded):
            with self.subTest(src=src):
                self.assertEqual(len(raw), 8, f"expected 64-bit DS encoding for {src}")
                w0, w1 = struct.unpack("<2I", raw)
                self.assertEqual((w0 >> 26) & 0x3F, 0x36, f"expected DS encoding prefix 0x36 for {src}")
                self.assertEqual((w0 >> 17) & 1, 0, f"expected GDS bit 17 clear for {src}")

                # Flip DS bit 17 (GDS = 1)
                gds_bytes = struct.pack("<2I", w0 | (1 << 17), w1)
                dec = rdna2_decode.decode_bytes(gds_bytes, mcpu="gfx1030", wave32=True)
                self.assertEqual(dec["invalid"], 0)
                self.assertEqual(len(dec["insns"]), 1)
                self.assertIn("gds", dec["insns"][0]["mods"], f"rdna2_decode did not emit gds for {src}")

                text, invalid = rdna2_decode.disasm_bytes(gds_bytes, mcpu="gfx1030", wave32=True)
                self.assertEqual(invalid, 0)
                disasm_line = text.strip()
                self.assertIn("gds", disasm_line)

                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    support.translate_lines(
                        "cs_gds_bin_rt",
                        preamble + [disasm_line, "s_endpgm"],
                        fssm=fssm,
                    )
                self.assertIn("gds", str(ctx.exception))

    def test_llvm_mc_native_gds_round_trip(self):
        """Assemble explicit `gds` DS instructions with llvm-mc, decode with rdna2_decode, reject in rdna2spirv."""
        gds_insns = [
            "ds_read_b32 v2, v1 gds",
            "ds_write_b32 v1, v2 gds",
            "ds_add_u32 v1, v2 gds",
            "ds_add_rtn_u32 v3, v1, v2 gds",
            "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2) gds",
        ]
        encoded = _assemble_gfx1030(gds_insns)
        fssm = _cs_fssm(32)
        preamble = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, 7",
            "v_mov_b32_e32 v3, 11",
        ]
        for src, raw in zip(gds_insns, encoded):
            with self.subTest(src=src):
                w0, _ = struct.unpack("<2I", raw)
                self.assertEqual((w0 >> 17) & 1, 1, f"expected GDS bit 17 set for {src}")
                text, invalid = rdna2_decode.disasm_bytes(raw, mcpu="gfx1030", wave32=True)
                self.assertEqual(invalid, 0)
                disasm_line = text.strip()
                self.assertIn("gds", disasm_line)
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    support.translate_lines(
                        "cs_gds_native_rt",
                        preamble + [disasm_line, "s_endpgm"],
                        fssm=fssm,
                    )
                self.assertIn("gds", str(ctx.exception))


class DsNonGdsPositiveControlTests(unittest.TestCase):
    def _run_cs(self, name, body, wave=32):
        spvasm = support.translate_lines(name, body + ["s_endpgm"], fssm=_cs_fssm(wave))
        _validate_spirv(spvasm)
        mem = {4 * i: 0xEEEEEEEE for i in range(wave)}
        si.run_workgroup(
            spvasm, wave, wave,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28,
            memory=mem,
        )
        return [mem[4 * i] for i in range(wave)]

    def test_lds_write_atomic_read_and_nop(self):
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v2, 100, v0",
            "ds_write_b32 v1, v2",
            "ds_nop",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "v_mov_b32_e32 v4, 5",
            "ds_add_rtn_u32 v5, v1, v4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            # v5 = 100 + lane (old), v6 = 105 + lane (new) -> store v5 + v6
            "v_add_nc_u32_e32 v7, v5, v6",
            "buffer_store_dword v7, v1, s[0:3], 0 offen",
        ]
        got = self._run_cs("cs_ds_pos_lds", body)
        expected = [(100 + i) + (105 + i) for i in range(32)]
        self.assertEqual(got, expected)

    def test_addtid_round_trip(self):
        body = [
            "s_mov_b32 m0, 0",
            "v_add_nc_u32_e32 v2, 200, v0",
            "ds_write_addtid_b32 v2 offset:16",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_addtid_b32 v3 offset:16",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v1, 2, v0",
            "buffer_store_dword v3, v1, s[0:3], 0 offen",
        ]
        got = self._run_cs("cs_ds_pos_addtid", body)
        self.assertEqual(got, [200 + i for i in range(32)])

    def test_cross_lane_swizzle_permute_bpermute(self):
        # ds_swizzle_b32 QUAD_PERM(1,0,3,2) swaps adjacent lanes
        got_swz = self._run_cs(
            "cs_ds_pos_swz",
            [
                "v_add_nc_u32_e32 v1, 10, v0",
                "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,0,3,2)",
                "v_lshlrev_b32_e32 v3, 2, v0",
                "buffer_store_dword v2, v3, s[0:3], 0 offen",
            ],
        )
        self.assertEqual(got_swz, [10 + (i ^ 1) for i in range(32)])

        # ds_bpermute_b32 reads from lane (lane + 1) mod 32 via offset:4
        got_bp = self._run_cs(
            "cs_ds_pos_bperm",
            [
                "v_lshlrev_b32_e32 v1, 2, v0",
                "v_add_nc_u32_e32 v2, 50, v0",
                "ds_bpermute_b32 v4, v1, v2 offset:4",
                "buffer_store_dword v4, v1, s[0:3], 0 offen",
            ],
        )
        self.assertEqual(got_bp, [50 + ((i + 1) & 31) for i in range(32)])

        # ds_permute_b32 scatters lane i's value to lane (i + 1) mod 32 via offset:4
        got_p = self._run_cs(
            "cs_ds_pos_perm",
            [
                "v_lshlrev_b32_e32 v1, 2, v0",
                "v_add_nc_u32_e32 v2, 70, v0",
                "ds_permute_b32 v4, v1, v2 offset:4",
                "buffer_store_dword v4, v1, s[0:3], 0 offen",
            ],
        )
        self.assertEqual(got_p, [70 + ((i - 1) & 31) for i in range(32)])

    def test_gs_ds_non_gds_translates_and_validates(self):
        spvasm = support.translate_lines(
            "gs_w64_ds_pos",
            DsGdsRejectionTests.GS_PRO
            + ["ds_write2_b32 v12, v14, v14 offset0:0 offset1:1"]
            + DsGdsRejectionTests.GS_EPI,
            dxbc=_gs_dxbc(),
        )
        _validate_spirv(spvasm)


if __name__ == "__main__":
    unittest.main()
