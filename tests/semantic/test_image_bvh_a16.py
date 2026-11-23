"""Tests for RDNA2 image_bvh_intersect_ray and image_bvh64_intersect_ray with a16.

When a16 is set on GFX10.3 (RDNA2), ray_dir.{x,y,z} and ray_inv_dir.{x,y,z} are packed as six f16
values across three VGPRs (instead of six f32 VGPRs), reducing the address VGPR count by 3:
  - image_bvh_intersect_ray a16:   8 VGPRs (1 node + 1 extent + 3 origin + 3 packed dir/inv)
  - image_bvh64_intersect_ray a16: 9 VGPRs (2 node + 1 extent + 3 origin + 3 packed dir/inv)

Packing layout (verified with LLVM 21 llc -march=amdgcn -mcpu=gfx1030):
  - rest[4]: [15:0] = dir.x,  [31:16] = dir.y
  - rest[5]: [15:0] = dir.z,  [31:16] = inv.x
  - rest[6]: [15:0] = inv.y,  [31:16] = inv.z

Covers:
  1. Decoder round-trip (known gfx1030 encodings + live llvm-mc -mcpu=gfx1030 when available)
     for 32-bit, a16, and NSA a16 forms of both image_bvh_intersect_ray and image_bvh64_intersect_ray.
  2. rdna2spirv translation + spirv-val --target-env vulkan1.2 validation for ps and cs, checking
     UnpackHalf2x16 emission and exact component extraction for rest[4:7].
  3. End-to-end box (box16/box32) and triangle intersection execution via spirv_mini_interp using
     oblique rays with distinct exact-f16 components on all axes, matching 32-bit results.
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

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)

NODE = 0x1040  # descriptor base 0x10 (256-byte units) = 0x1000; node id 0x08 | type -> +0x40
MISS = 0xFFFFFFFF


def _f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _u2f(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w))[0]


def _f16(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _pack_f16x2(lo: float, hi: float) -> int:
    return _f16(lo) | (_f16(hi) << 16)


def _llvm_mc_cmd() -> list[str] | None:
    env = os.environ.get("LLVM_MC")
    if env:
        return shlex.split(env)
    exe = shutil.which("llvm-mc")
    return [exe] if exe else None


def _assemble_gfx1030(asm: str) -> bytes:
    cmd = _llvm_mc_cmd()
    assert cmd is not None
    p = subprocess.run(
        cmd + ["-mcpu=gfx1030", "-triple=amdgcn--", "-show-encoding"],
        input=asm + "\n",
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        raise AssertionError(f"llvm-mc failed for {asm!r}:\n{p.stderr}")
    m = re.search(r"encoding:\s*\[([^\]]+)\]", p.stdout)
    if not m:
        raise AssertionError(f"no encoding in llvm-mc output for {asm!r}:\n{p.stdout}")
    return bytes(int(tok.strip(), 16) for tok in m.group(1).split(","))


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _decode_one(raw: bytes) -> dict:
    res = rdna2_decode.decode_bytes(raw, mcpu="gfx1030")
    if res["invalid"] or len(res["insns"]) != 1:
        raise AssertionError(f"expected 1 valid decoded instruction, got: {res}")
    return res["insns"][0]


class TestImageBvhA16Decode(unittest.TestCase):
    # Machine-code bytes verified with llvm-mc -mcpu=gfx1030 -triple=amdgcn-- -show-encoding.
    ENCODED_CASES = [
        (
            "image_bvh_intersect_ray v[0:3], v[0:10], s[0:3]",
            bytes([0x01, 0x9F, 0x98, 0xF1, 0x00, 0x00, 0x00, 0x00]),
            "image_bvh_intersect_ray",
            False,
            {"k": "vgpr", "r": 0, "n": 11},
        ),
        (
            "image_bvh_intersect_ray v[0:3], v[0:7], s[0:3] a16",
            bytes([0x01, 0x9F, 0x98, 0xF1, 0x00, 0x00, 0x00, 0x40]),
            "image_bvh_intersect_ray",
            True,
            {"k": "vgpr", "r": 0, "n": 8},
        ),
        (
            "image_bvh64_intersect_ray v[0:3], v[0:11], s[0:3]",
            bytes([0x01, 0x9F, 0x9C, 0xF1, 0x00, 0x00, 0x00, 0x00]),
            "image_bvh64_intersect_ray",
            False,
            {"k": "vgpr", "r": 0, "n": 12},
        ),
        (
            "image_bvh64_intersect_ray v[0:3], v[0:8], s[0:3] a16",
            bytes([0x01, 0x9F, 0x9C, 0xF1, 0x00, 0x00, 0x00, 0x40]),
            "image_bvh64_intersect_ray",
            True,
            {"k": "vgpr", "r": 0, "n": 9},
        ),
        (
            "image_bvh_intersect_ray v[0:3], [v10, v11, v12, v13, v14, v15, v16, v17], s[0:3] a16",
            bytes([0x05, 0x9F, 0x98, 0xF1, 0x0A, 0x00, 0x00, 0x40,
                   0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x00]),
            "image_bvh_intersect_ray",
            True,
            {"k": "vlist", "r": [10, 11, 12, 13, 14, 15, 16, 17]},
        ),
        (
            "image_bvh64_intersect_ray v[0:3], [v10, v11, v12, v13, v14, v15, v16, v17, v18], s[0:3] a16",
            bytes([0x05, 0x9F, 0x9C, 0xF1, 0x0A, 0x00, 0x00, 0x40,
                   0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12]),
            "image_bvh64_intersect_ray",
            True,
            {"k": "vlist", "r": [10, 11, 12, 13, 14, 15, 16, 17, 18]},
        ),
    ]

    def _assert_op(self, got: dict, want: dict) -> None:
        for k, v in want.items():
            self.assertEqual(got.get(k), v, msg=(k, got, want))

    def test_decode_bytes_bvh_and_bvh64_a16(self):
        for asm, raw, op, expect_a16, expect_vaddr in self.ENCODED_CASES:
            with self.subTest(asm=asm):
                ins = _decode_one(raw)
                self.assertEqual(ins["op"], op)
                self.assertEqual("a16" in ins["mods"], expect_a16)
                self._assert_op(ins["args"][0], {"k": "vgpr", "r": 0, "n": 4})
                self._assert_op(ins["args"][1], expect_vaddr)
                self._assert_op(ins["args"][2], {"k": "sgpr", "r": 0, "n": 4})

    def test_llvm_mc_roundtrip_gfx1030(self):
        if _llvm_mc_cmd() is None:
            self.skipTest("llvm-mc not available")
        for asm, expected_raw, op, expect_a16, expect_vaddr in self.ENCODED_CASES:
            with self.subTest(asm=asm):
                raw = _assemble_gfx1030(asm)
                self.assertEqual(raw, expected_raw)
                ins = _decode_one(raw)
                self.assertEqual(ins["op"], op)
                self.assertEqual("a16" in ins["mods"], expect_a16)
                self._assert_op(ins["args"][1], expect_vaddr)


class TestImageBvhA16TranslateAndValidate(unittest.TestCase):
    def _build_shader(self, stage: str, is64: bool, nsa: bool) -> str:
        op = "image_bvh64_intersect_ray" if is64 else "image_bvh_intersect_ray"
        n_addr = 9 if is64 else 8
        lines = [f"v_mov_b32_e32 v{i}, 0" for i in range(1, 1 + n_addr)]
        lines += ["s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0", "s_mov_b32 s6, -1", "s_mov_b32 s7, 0"]
        if nsa:
            addr = "[" + ", ".join(f"v{i}" for i in range(1, 1 + n_addr)) + "]"
        else:
            addr = f"v[1:{n_addr}]"
        lines.append(f"{op} v[20:23], {addr}, s[4:7] a16")
        if stage == "ps":
            lines.append("exp mrt0 v20, v21, v22, v23 done vm")
        lines.append("s_endpgm")
        name = f"{stage}_bvh{'64' if is64 else '32'}_a16_{'nsa' if nsa else 'rng'}"
        return support.translate_lines(name, lines, fssm=CS_FSSM if stage == "cs" else None)

    def test_translate_validate_and_unpack_shape(self):
        for stage in ("ps", "cs"):
            for is64 in (False, True):
                for nsa in (False, True):
                    with self.subTest(stage=stage, is64=is64, nsa=nsa):
                        spv = self._build_shader(stage, is64, nsa)
                        _validate(spv)
                        # Before the first OpSelectionMerge (tri/box dispatch), image_bvh a16 must emit
                        # exactly 3 UnpackHalf2x16 ops (for rest[4], rest[5], rest[6]), each extracted
                        # at lanes 0 and 1.
                        head = spv.split("OpSelectionMerge", 1)[0]
                        unpacks = re.findall(
                            r"^\s*(%\w+)\s*=\s*OpExtInst %v2float %glsl UnpackHalf2x16 (%\w+)$",
                            head,
                            re.M,
                        )
                        self.assertEqual(len(unpacks), 3, msg=f"expected 3 ray UnpackHalf2x16 ops, got {unpacks}")
                        for v2_id, _ in unpacks:
                            for lane in (0, 1):
                                pat = rf"^\s*%\w+\s*=\s*OpCompositeExtract %float {re.escape(v2_id)} {lane}$"
                                self.assertRegex(head, re.compile(pat, re.M))

    def test_too_few_vgprs_rejected(self):
        for op, bad_end in (("image_bvh_intersect_ray", 7), ("image_bvh64_intersect_ray", 8)):
            with self.subTest(op=op):
                lines = [f"v_mov_b32_e32 v{i}, 0" for i in range(1, bad_end + 1)]
                lines += [
                    "s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0", "s_mov_b32 s6, -1", "s_mov_b32 s7, 0",
                    f"{op} v[20:23], v[1:{bad_end}], s[4:7] a16",
                    "exp mrt0 v20, v21, v22, v23 done vm",
                    "s_endpgm",
                ]
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines("ps_bvh_a16_short", lines)


class TestImageBvhA16Semantics(unittest.TestCase):
    def _run_ray(
        self,
        is64: bool,
        a16: bool,
        node_type: int,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        inv_dir: tuple[float, float, float],
        mem: dict[int, int],
        extent: float = 100.0,
    ) -> list[int]:
        lines = []
        mov = lambda r, v: lines.append(f"v_mov_b32_e32 v{r}, 0x{v:x}")  # noqa: E731
        r = 1
        mov(r, 0x08 | node_type)
        r += 1
        if is64:
            mov(r, 0)
            r += 1
        mov(r, _f2u(extent))
        r += 1
        for c in origin:
            mov(r, _f2u(c))
            r += 1
        if a16:
            packed = [
                _pack_f16x2(direction[0], direction[1]),
                _pack_f16x2(direction[2], inv_dir[0]),
                _pack_f16x2(inv_dir[1], inv_dir[2]),
            ]
            for w in packed:
                mov(r, w)
                r += 1
        else:
            for c in (*direction, *inv_dir):
                mov(r, _f2u(c))
                r += 1
        end_r = r - 1
        op = "image_bvh64_intersect_ray" if is64 else "image_bvh_intersect_ray"
        mod = " a16" if a16 else ""
        lines += [
            "s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0", "s_mov_b32 s6, -1", "s_mov_b32 s7, 0",
            f"{op} v[20:23], v[1:{end_r}], s[4:7]{mod}",
            "exp mrt0 v20, v21, v22, v23 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_bvh{'64' if is64 else '32'}_{'a16' if a16 else 'f32'}_{node_type}", lines)
        _validate(spv)
        return list(support.mrt0_words(spv, memory=mem))

    def test_oblique_box_intersection_matches_f32(self):
        # Ray: origin = (0, 0, -2), dir = (0.5, 0.25, 1.0), inv = (2.0, 4.0, 1.0) (all exact in f16).
        # At t = 2: p = (1.0, 0.5, 0.0). At t = 4: p = (2.0, 1.0, 2.0).
        # Distinct non-zero inv.{x,y,z} ensure any mis-unpacking of rest[4..6] breaks the slab overlap.
        origin = (0.0, 0.0, -2.0)
        direction = (0.5, 0.25, 1.0)
        inv_dir = (2.0, 4.0, 1.0)
        children = [0x20, 0x28, 0x30, 0x38]
        boxes = [
            (0.75, 0.25, -0.5, 1.25, 0.75, 0.5),  # hit around t = 2
            (1.75, 0.75, 1.5, 2.25, 1.25, 2.5),   # hit around t = 4
            (-1.0, -1.0, 0.0, -0.5, -0.5, 1.0),   # miss on x and y
            (0.75, 2.0, -0.5, 1.25, 3.0, 0.5),    # miss on y only (tests inv.y)
        ]
        for node_type in (4, 5):
            mem = {}
            for i, child in enumerate(children):
                mem[NODE + 4 * i] = child
            for i, box in enumerate(boxes):
                if node_type == 4:
                    words = [_f16(v) for v in box]
                    for c in range(3):
                        mem[NODE + 16 + 12 * i + 4 * c] = words[2 * c] | (words[2 * c + 1] << 16)
                else:
                    for c, v in enumerate(box):
                        mem[NODE + 16 + 24 * i + 4 * c] = _f2u(v)
            for is64 in (False, True):
                with self.subTest(node_type=node_type, is64=is64):
                    got_a16 = self._run_ray(is64, True, node_type, origin, direction, inv_dir, mem)
                    got_f32 = self._run_ray(is64, False, node_type, origin, direction, inv_dir, mem)
                    self.assertEqual(got_a16, [0x20, 0x28, MISS, MISS])
                    self.assertEqual(got_a16, got_f32)

    def test_oblique_triangle_intersection_matches_f32(self):
        # Ray: origin = (-0.25, -0.125, -1.0), dir = (0.5, 0.25, 1.0), inv = (2.0, 4.0, 1.0).
        # Intersects z = 0 plane at t = 1.0, point (0.25, 0.125, 0.0) inside TRI0 = (0,0,0), (1,0,0), (0,1,0).
        origin = (-0.25, -0.125, -1.0)
        direction = (0.5, 0.25, 1.0)
        inv_dir = (2.0, 4.0, 1.0)
        slots = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.5, 2.0, 0.0)]
        mem = {}
        for s, v in enumerate(slots):
            for c in range(3):
                mem[NODE + 12 * s + 4 * c] = _f2u(v[c])
        mem[NODE + 60] = 0x09  # TRI0 remap: (v1, v2) -> (x, y) on unit right triangle
        for is64 in (False, True):
            with self.subTest(is64=is64):
                w_a16 = self._run_ray(is64, True, 0, origin, direction, inv_dir, mem)
                w_f32 = self._run_ray(is64, False, 0, origin, direction, inv_dir, mem)
                self.assertEqual(w_a16, w_f32)
                r = [_u2f(w) for w in w_a16]
                self.assertAlmostEqual(r[0] / r[1], 1.0, places=6)
                self.assertAlmostEqual(r[2] / r[1], 0.25, places=6)
                self.assertAlmostEqual(r[3] / r[1], 0.125, places=6)


if __name__ == "__main__":
    unittest.main()
