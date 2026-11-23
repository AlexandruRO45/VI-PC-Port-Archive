"""Semantic, decoder, and SPIR-V validation tests for image_gather4_l / image_gather4_l_o
on SQ_RSRC_IMG_2D_ARRAY and SQ_RSRC_IMG_CUBE (including a16, d16, dmask 0x1..0x8, PS & CS).
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.dirname(os.path.abspath(__file__))
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402


def _find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for ext in ("", ".exe"):
            cand = os.path.join(sdk, "Bin", name + ext)
            if os.path.isfile(cand):
                return cand
    return None


SPIRV_AS = _find_tool("spirv-as")
SPIRV_VAL = _find_tool("spirv-val")


def _llvm_mc() -> list[str] | None:
    env = os.environ.get("LLVM_MC")
    cmd = shlex.split(env) if env else ["llvm-mc"]
    try:
        r = subprocess.run(
            cmd + ["-triple=amdgcn--", "-mcpu=gfx1030", "-show-encoding"],
            input="s_endpgm\n",
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cmd if r.returncode == 0 and "encoding" in r.stdout else None


MC = _llvm_mc()


def _validate_spirv(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa = os.path.join(d, "shader.spvasm")
        sv = os.path.join(d, "shader.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _assemble_gfx1030(lines: list[str]) -> bytes:
    assert MC is not None
    src = "\n".join(lines) + "\n"
    r = subprocess.run(
        MC + ["-triple=amdgcn--", "-mcpu=gfx1030", "-show-encoding"],
        input=src,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        raise AssertionError(f"llvm-mc failed for:\n{src}\n{r.stderr}")
    raw = bytearray()
    for line in r.stdout.splitlines():
        if "encoding: [" in line:
            body = line.split("encoding: [", 1)[1].split("]", 1)[0]
            raw.extend(int(b, 16) for b in body.split(","))
    return bytes(raw)


def _vrange(start: int, count: int) -> str:
    return f"v{start}" if count == 1 else f"v[{start}:{start + count - 1}]"


def _gather_insn(op: str, dim: str, dmask: int, a16: bool, d16: bool) -> str:
    n_dst = 2 if d16 else 4
    is_off = op.endswith("_o")
    if a16:
        # GFX10 a16 packs (u, v) in v4 and (layer/face, lod) in v5; _o prepends 1 32-bit offset VGPR.
        n_addr = 3 if is_off else 2
    else:
        n_addr = 5 if is_off else 4
    mods = [f"dmask:0x{dmask:x}", f"dim:SQ_RSRC_IMG_{dim}"]
    if a16:
        mods.append("a16")
    if d16:
        mods.append("d16")
    return f"{op} {_vrange(0, n_dst)}, {_vrange(4, n_addr)}, s[0:7], s[8:11] " + " ".join(mods)


def _cs_fssm(threads: int = 64) -> bytes:
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def _translate_stage(stage: str, insn: str) -> str:
    if stage == "ps":
        return support.translate_lines(
            "ps_gather4_l_dims",
            [insn, "s_waitcnt vmcnt(0)", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"],
        )
    assert stage == "cs"
    return support.translate_lines(
        "cs_gather4_l_dims",
        [insn, "s_waitcnt vmcnt(0)", "s_endpgm"],
        fssm=_cs_fssm(),
    )


class TestImageGather4LDims(unittest.TestCase):
    def test_matrix_translate_and_validate(self):
        for op in ("image_gather4_l", "image_gather4_l_o"):
            for dim in ("2D_ARRAY", "CUBE"):
                for a16 in (False, True):
                    for d16 in (False, True):
                        for dmask in (0x1, 0x2, 0x4, 0x8):
                            for stage in ("ps", "cs"):
                                with self.subTest(
                                    op=op, dim=dim, a16=a16, d16=d16, dmask=hex(dmask), stage=stage
                                ):
                                    insn = _gather_insn(op, dim, dmask, a16, d16)
                                    spv = _translate_stage(stage, insn)
                                    _validate_spirv(spv)

                                    # Arrayed 2D view (%images_arr / %img2da) + manual query/fetch, no OpSampledImage.
                                    self.assertIn("OpImageQueryLevels %int", spv)
                                    self.assertIn("OpImageQuerySizeLod %v3int", spv)
                                    self.assertIn("OpExtInst %float %glsl RoundEven", spv)
                                    self.assertIn("OpCompositeConstruct %v3int", spv)
                                    self.assertIn("OpImageFetch", spv)
                                    self.assertNotIn("OpSampledImage", spv)
                                    self.assertIn("%images_arr", spv)
                                    for signedness in (1, 2):
                                        self.assertIn(f"%itx{signedness}_1", spv)

                                    # Selected gather channel extracted from each OpImageFetch result.
                                    ch = dmask.bit_length() - 1
                                    self.assertRegex(
                                        spv,
                                        rf"OpCompositeExtract %(?:float|uint|int) %\w+ {ch}\b",
                                    )

                                    # Cube face/coord remapping via cube_coords (0.125 = 0x3e000000).
                                    if dim == "CUBE":
                                        self.assertIn("OpFSub %float", spv)
                                        self.assertIn("%c3e000000", spv)
                                    # a16 unpacking via UnpackHalf2x16.
                                    if a16:
                                        self.assertIn("OpExtInst %v2float %glsl UnpackHalf2x16", spv)
                                    # _o extracts signed 6-bit X and Y offsets and adds to texel coords.
                                    if op.endswith("_o"):
                                        self.assertIn("OpBitFieldSExtract %uint", spv)
                                        self.assertIn("OpIAdd %v2int", spv)
                                    # d16 packs pairs of 16-bit results into 2 VGPRs.
                                    if d16:
                                        self.assertIn("OpShiftLeftLogical %uint", spv)
                                        self.assertIn("OpBitwiseOr %uint", spv)

    def test_unsupported_dims_and_forms(self):
        cases = [
            "image_gather4_l v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D",
            "image_gather4_l_o v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D",
            "image_gather4_l v[0:3], v[4:7], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D",
            "image_gather4_l_o v[0:3], v[4:8], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D",
            "image_gather4_l v[0:3], v[4:7], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY",
            "image_gather4_l v[0:3], v[4:7], s[0:7], s[8:11] dmask:0x0 dim:SQ_RSRC_IMG_CUBE",
            # Too few address VGPRs for 32-bit 2D_ARRAY / CUBE (needs 4: u, v, layer, lod).
            "image_gather4_l v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY",
            "image_gather4_l v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE",
        ]
        for insn in cases:
            with self.subTest(insn=insn):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate_stage("ps", insn)


@unittest.skipUnless(MC, "needs llvm-mc with AMDGPU target (set LLVM_MC)")
class TestImageGather4LBinaryDecodeRoundTrip(unittest.TestCase):
    def test_decode_and_translate_round_trip(self):
        for op in ("image_gather4_l", "image_gather4_l_o"):
            for dim in ("2D_ARRAY", "CUBE"):
                for a16 in (False, True):
                    for d16 in (False, True):
                        with self.subTest(op=op, dim=dim, a16=a16, d16=d16):
                            insn = _gather_insn(op, dim, 0x2, a16, d16)
                            raw = _assemble_gfx1030(
                                [insn, "s_waitcnt vmcnt(0)", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
                            )
                            decoded = rdna2_decode.decode_bytes(raw, mcpu="gfx1030", wave32=True)
                            first = decoded["insns"][0]
                            self.assertEqual(first["op"], op)
                            self.assertIn(f"dim:SQ_RSRC_IMG_{dim}", first["mods"])
                            self.assertEqual("a16" in first["mods"], a16)
                            self.assertEqual("d16" in first["mods"], d16)

                            expected_dst = 2 if d16 else 4
                            expected_addr = (3 if op.endswith("_o") else 2) if a16 else (5 if op.endswith("_o") else 4)
                            self.assertEqual(first["args"][0]["n"], expected_dst)
                            self.assertEqual(first["args"][1]["n"], expected_addr)

                            listing, n_unknown = rdna2_decode.disasm_bytes(raw, mcpu="gfx1030", wave32=True)
                            self.assertEqual(n_unknown, 0)
                            spv = support.translate_listing("ps_gather4_l_rt", listing.encode("utf-8"))
                            _validate_spirv(spv)
                            self.assertIn("OpImageQuerySizeLod %v3int", spv)
                            self.assertIn("OpImageFetch", spv)


if __name__ == "__main__":
    unittest.main()
