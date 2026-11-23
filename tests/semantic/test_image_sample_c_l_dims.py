"""Semantic, SPIR-V validation, and llvm-mc round-trip tests for RDNA2 image_sample_c_l
across 1D, 2D, 2D_ARRAY, and CUBE dimensions (+ a16, d16, PS/CS stages).

RDNA2 MIMG 0x02c (`image_sample_c_l`) address register layouts (verified against
`llvm-mc -arch=amdgcn -mcpu=gfx1030 -show-encoding` and `tools/rdna2_decode.py`):
  - 1D:       [ref, u, lod]              (3 VGPRs; a16: [ref, f16(u, lod)] = 2 VGPRs)
  - 2D:       [ref, u, v, lod]           (4 VGPRs; a16: [ref, f16(u, v), f16(lod, _)] = 3 VGPRs)
  - 2D_ARRAY: [ref, u, v, slice, lod]    (5 VGPRs; a16: [ref, f16(u, v), f16(slice, lod)] = 3 VGPRs)
  - CUBE:     [ref, u, v, face, lod]     (5 VGPRs; a16: [ref, f16(u, v), f16(face, lod)] = 3 VGPRs)

3D (forbidden by Vulkan/SPIR-V for OpImageSampleDref*), non-0x1 dmask, wrong vaddr
widths, and tfe/lwe/glc modifiers must be rejected with rdna2spirv.Unsupported.
"""
import os
import re
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


def _find_vulkan_tool(name: str) -> str | None:
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


SPIRV_AS = _find_vulkan_tool("spirv-as")
SPIRV_VAL = _find_vulkan_tool("spirv-val")


def _find_llvm_mc_cmd() -> list[str] | None:
    env = os.environ.get("LLVM_MC")
    if env:
        return env.split()
    direct = shutil.which("llvm-mc")
    if direct:
        return [direct]
    wsl = shutil.which("wsl")
    if wsl:
        probe = subprocess.run(
            [wsl, "-d", "Ubuntu-24.04", "-e", "test", "-x", "/usr/lib/llvm-21/bin/llvm-mc"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return [wsl, "-d", "Ubuntu-24.04", "-e", "/usr/lib/llvm-21/bin/llvm-mc"]
    return None


LLVM_MC_CMD = _find_llvm_mc_cmd()


def _cs_fssm(threads_x: int = 32) -> bytes:
    """Minimal 0x148-byte CS FSSM header with 12 user SGPRs (s[0:11]) and group size."""
    buf = bytearray(0x148)
    struct.pack_into("<4I", buf, 0x138, 12 << 1, threads_x, 1, 1)
    return bytes(buf)


CS_FSSM = _cs_fssm()


def _validate_spirv(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        raise unittest.SkipTest("spirv-as / spirv-val not available")
    with tempfile.TemporaryDirectory() as tmp:
        sa = os.path.join(tmp, "mod.spvasm")
        sv = os.path.join(tmp, "mod.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _assemble_gfx1030(line: str) -> bytes:
    assert LLVM_MC_CMD is not None
    r = subprocess.run(
        LLVM_MC_CMD + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
        input=line + "\n",
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise AssertionError(f"llvm-mc failed for {line!r}:\n{r.stderr}")
    m = re.search(r"encoding:\s*\[([^\]]+)\]", r.stdout)
    if not m:
        raise AssertionError(f"missing encoding in llvm-mc output:\n{r.stdout}")
    return bytes(int(tok.strip(), 16) for tok in m.group(1).split(","))


def _build_shader(dim: str, a16: bool, d16: bool, stage: str) -> tuple[str, list[str]]:
    if dim == "SQ_RSRC_IMG_1D":
        if a16:
            moves = ["v_mov_b32 v0, 0.5", "v_mov_b32 v1, 0x3c003400"]  # u=0.25, lod=1.0
            vaddr = "v[0:1]"
        else:
            moves = ["v_mov_b32 v0, 0.5", "v_mov_b32 v1, 0.25", "v_mov_b32 v2, 1.0"]
            vaddr = "v[0:2]"
    elif dim == "SQ_RSRC_IMG_2D":
        if a16:
            moves = [
                "v_mov_b32 v0, 0.5",
                "v_mov_b32 v1, 0x3a003400",  # u=0.25, v=0.75
                "v_mov_b32 v2, 0x00003c00",  # lod=1.0, pad=0.0
            ]
            vaddr = "v[0:2]"
        else:
            moves = [
                "v_mov_b32 v0, 0.5",
                "v_mov_b32 v1, 0.25",
                "v_mov_b32 v2, 0.75",
                "v_mov_b32 v3, 1.0",
            ]
            vaddr = "v[0:3]"
    else:  # SQ_RSRC_IMG_2D_ARRAY or SQ_RSRC_IMG_CUBE
        if a16:
            moves = [
                "v_mov_b32 v0, 0.5",
                "v_mov_b32 v1, 0x3e003d00",  # u=1.25, v=1.5
                "v_mov_b32 v2, 0x3c004000",  # slice/face=2.0, lod=1.0
            ]
            vaddr = "v[0:2]"
        else:
            moves = [
                "v_mov_b32 v0, 0.5",
                "v_mov_b32 v1, 1.25",
                "v_mov_b32 v2, 1.5",
                "v_mov_b32 v3, 2.0",
                "v_mov_b32 v4, 1.0",
            ]
            vaddr = "v[0:4]"

    mods = [f"dmask:0x1", f"dim:{dim}"]
    if a16:
        mods.append("a16")
    if d16:
        mods.append("d16")
    sample_line = f"image_sample_c_l v5, {vaddr}, s[0:7], s[8:11] " + " ".join(mods)

    if stage == "ps":
        tail = ["s_waitcnt vmcnt(0)", "exp mrt0 v5, v5, v5, v5 done vm", "s_endpgm"]
        name = f"ps_sample_c_l_{dim.lower()}_{'a16' if a16 else 'a32'}_{'d16' if d16 else 'f32'}"
    else:
        tail = ["s_waitcnt vmcnt(0)", "s_endpgm"]
        name = f"cs_sample_c_l_{dim.lower()}_{'a16' if a16 else 'a32'}_{'d16' if d16 else 'f32'}"
    return name, moves + [sample_line] + tail


def _trace_vgpr_slot(defs: dict[str, str], val: str) -> str:
    """Resolve an OpBitcast %float (OpLoad %uint (OpAccessChain ... %vgpr %cN)) back to '%cN'."""
    cur = val
    while cur in defs and defs[cur].startswith("OpBitcast "):
        cur = defs[cur].split()[-1]
    load = defs[cur]
    assert load.startswith("OpLoad %uint "), f"unexpected load for {val}: {load}"
    ptr = defs[load.split()[-1]]
    parts = ptr.split()
    assert parts[-2] == "%vgpr", f"expected %vgpr access chain, got {ptr}"
    return parts[-1]


def _trace_a16_half(defs: dict[str, str], val: str) -> tuple[str, int]:
    """Resolve an unpacked a16 float back to ('%cN', half_index)."""
    cur = val
    while cur in defs and defs[cur].startswith("OpBitcast "):
        cur = defs[cur].split()[-1]
    ext = defs[cur]
    parts = ext.split()
    assert parts[0] == "OpCompositeExtract", f"expected OpCompositeExtract, got {ext}"
    pair_id, half = parts[2], int(parts[3])
    unpack = defs[pair_id]
    assert "UnpackHalf2x16" in unpack, f"expected UnpackHalf2x16, got {unpack}"
    raw_load = unpack.split()[-1]
    return _trace_vgpr_slot(defs, raw_load), half


class TestImageSampleCLDims(unittest.TestCase):
    DIMS = (
        "SQ_RSRC_IMG_1D",
        "SQ_RSRC_IMG_2D",
        "SQ_RSRC_IMG_2D_ARRAY",
        "SQ_RSRC_IMG_CUBE",
    )

    def test_all_dims_a16_d16_ps_and_cs(self):
        expected_lod = {
            ("SQ_RSRC_IMG_1D", False): "%c2",
            ("SQ_RSRC_IMG_1D", True): ("%c1", 1),
            ("SQ_RSRC_IMG_2D", False): "%c3",
            ("SQ_RSRC_IMG_2D", True): ("%c2", 0),
            ("SQ_RSRC_IMG_2D_ARRAY", False): "%c4",
            ("SQ_RSRC_IMG_2D_ARRAY", True): ("%c2", 1),
            ("SQ_RSRC_IMG_CUBE", False): "%c4",
            ("SQ_RSRC_IMG_CUBE", True): ("%c2", 1),
        }
        expected_array = {
            "SQ_RSRC_IMG_1D": "%images_1d",
            "SQ_RSRC_IMG_2D": "%images",
            "SQ_RSRC_IMG_2D_ARRAY": "%images_arr",
            "SQ_RSRC_IMG_CUBE": "%images_arr",
        }
        expected_coord_ty = {
            "SQ_RSRC_IMG_1D": None,  # scalar %float
            "SQ_RSRC_IMG_2D": "%v2float",
            "SQ_RSRC_IMG_2D_ARRAY": "%v3float",
            "SQ_RSRC_IMG_CUBE": "%v3float",
        }

        for stage in ("ps", "cs"):
            for dim in self.DIMS:
                for a16 in (False, True):
                    for d16 in (False, True):
                        with self.subTest(stage=stage, dim=dim, a16=a16, d16=d16):
                            name, lines = _build_shader(dim, a16, d16, stage)
                            spv = support.translate_lines(
                                name,
                                lines,
                                fssm=CS_FSSM if stage == "cs" else None,
                            )
                            _validate_spirv(spv)

                            defs = dict(re.findall(r"(%\w+)\s*=\s*(.+)", spv))
                            m = re.search(
                                r"(%\w+) = OpImageSampleDrefExplicitLod %float (%\w+) (%\w+) (%\w+) Lod (%\w+)",
                                spv,
                            )
                            self.assertIsNotNone(m, f"missing OpImageSampleDrefExplicitLod in {name}")
                            result, sampled_img, coord, ref, lod = m.groups()

                            # Check image array binding and capability
                            img_id = defs[sampled_img].split()[2]
                            img_ptr = defs[img_id].split()[-1]
                            self.assertIn(expected_array[dim], defs[img_ptr])
                            if dim == "SQ_RSRC_IMG_1D":
                                self.assertIn("OpCapability Sampled1D", spv)

                            # Check coordinate type and cube_coords transform
                            coord_ty = expected_coord_ty[dim]
                            if coord_ty is None:
                                self.assertFalse(defs[coord].startswith("OpCompositeConstruct "))
                            else:
                                self.assertTrue(
                                    defs[coord].startswith(f"OpCompositeConstruct {coord_ty} "),
                                    f"expected {coord_ty} coord, got {defs[coord]}",
                                )
                            if dim == "SQ_RSRC_IMG_CUBE":
                                s_id, t_id, layer_id = defs[coord].split()[-3:]
                                self.assertIn("OpFSub %float", defs[s_id])
                                self.assertIn("%f1", defs[s_id])
                                self.assertIn("OpFSub %float", defs[t_id])
                                self.assertIn("%f1", defs[t_id])
                                self.assertIn("OpFSub %float", defs[layer_id])

                            # Ref is always f32 in v0 (even under a16)
                            self.assertEqual(_trace_vgpr_slot(defs, ref), "%c0")

                            # Verify LOD register / a16 half
                            if a16:
                                self.assertEqual(_trace_a16_half(defs, lod), expected_lod[(dim, True)])
                            else:
                                self.assertEqual(_trace_vgpr_slot(defs, lod), expected_lod[(dim, False)])

                            # Verify d16 f16 packing vs f32 passthrough
                            bits = next(k for k, v in defs.items() if v == f"OpBitcast %uint {result}")
                            has_f16_pack = any(
                                v == f"OpShiftRightLogical %uint {bits} %c10" for v in defs.values()
                            )
                            self.assertEqual(has_f16_pack, d16)

    def test_llvm_mc_binary_decode_roundtrip(self):
        if LLVM_MC_CMD is None:
            self.skipTest("llvm-mc not available")

        cases = [
            ("SQ_RSRC_IMG_1D", "v[0:2]", ""),
            ("SQ_RSRC_IMG_1D", "v[0:1]", " a16"),
            ("SQ_RSRC_IMG_1D", "v[0:1]", " a16 d16"),
            ("SQ_RSRC_IMG_2D", "v[0:3]", ""),
            ("SQ_RSRC_IMG_2D", "v[0:2]", " a16"),
            ("SQ_RSRC_IMG_2D", "v[0:2]", " a16 d16"),
            ("SQ_RSRC_IMG_2D_ARRAY", "v[0:4]", ""),
            ("SQ_RSRC_IMG_2D_ARRAY", "v[0:2]", " a16"),
            ("SQ_RSRC_IMG_2D_ARRAY", "v[0:2]", " a16 d16"),
            ("SQ_RSRC_IMG_CUBE", "v[0:4]", ""),
            ("SQ_RSRC_IMG_CUBE", "v[0:2]", " a16"),
            ("SQ_RSRC_IMG_CUBE", "v[0:2]", " a16 d16"),
        ]
        for dim, vaddr, extra in cases:
            asm = f"image_sample_c_l v5, {vaddr}, s[0:7], s[8:11] dmask:0x1 dim:{dim}{extra}"
            with self.subTest(asm=asm):
                raw = _assemble_gfx1030(asm)
                decoded_text, warns = rdna2_decode.disasm_bytes(raw)
                self.assertEqual(warns, 0)
                decoded_insn = decoded_text.split(";")[0].strip()
                self.assertEqual(decoded_insn, asm)

                spv = support.translate_lines(
                    f"ps_rt_{dim.lower()}",
                    [
                        decoded_insn,
                        "s_waitcnt vmcnt(0)",
                        "exp mrt0 v5, v5, v5, v5 done vm",
                        "s_endpgm",
                    ],
                )
                _validate_spirv(spv)
                self.assertIn("OpImageSampleDrefExplicitLod %float ", spv)

    def test_negative_rejected_forms(self):
        rejected = [
            # 3D forbidden by Vulkan/SPIR-V for OpImageSampleDref*
            "image_sample_c_l v5, v[0:4], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D",
            "image_sample_c_l v5, v[0:2], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_3D a16",
            # Non-scalar dmask
            "image_sample_c_l v[5:6], v[0:2], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_1D",
            "image_sample_c_l v[5:6], v[0:3], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_2D",
            "image_sample_c_l v[5:6], v[0:4], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY",
            "image_sample_c_l v[5:6], v[0:4], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_CUBE",
            # Short vaddr
            "image_sample_c_l v5, v[0:1], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D",
            "image_sample_c_l v5, v0, s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D a16",
            "image_sample_c_l v5, v[0:2], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D",
            "image_sample_c_l v5, v[0:1], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
            "image_sample_c_l v5, v[0:3], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY",
            "image_sample_c_l v5, v[0:1], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY a16",
            "image_sample_c_l v5, v[0:3], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE",
            "image_sample_c_l v5, v[0:1], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE a16",
            # Extra vaddr
            "image_sample_c_l v5, v[0:3], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D",
            "image_sample_c_l v5, v[0:2], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D a16",
            # Unsupported modifiers: tfe, lwe, glc
            "image_sample_c_l v[5:6], v[0:2], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D tfe",
            "image_sample_c_l v[5:6], v[0:4], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY lwe",
            "image_sample_c_l v5, v[0:4], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE glc",
        ]
        for insn in rejected:
            with self.subTest(insn=insn):
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines("ps_neg_sample_c_l", [insn, "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
