"""Unit tests for the TEST-184 MIMG GPU test fixtures and checkers (standard library only; no GPU or LLVM required).

    python -I tests/test_mimg_ops.py

Covers:
  - Translation of all four TEST-184 .asm fixtures (cs_imgatomic, ps_get_lod, ps_get_resinfo, cs_bvh32) through
    tools/rdna2spirv.py, verifying capabilities, bindings, SPIR-V opcodes, operand order, and optional spirv-val.
  - Pixel checkers and runner routing in runtime/test/shaders/check.py and scripts/run_gpu_tests.py (exact match,
    mismatched/swapped channels, NaN/short frames, and exit-77 SKIP handling).
"""
import contextlib
import io
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "runtime" / "test" / "shaders"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(SHADER_DIR))
sys.path.insert(0, str(ROOT / "scripts"))
import rdna2spirv  # noqa: E402
import compute_fixture  # noqa: E402
import check  # noqa: E402
import run_gpu_tests as rgt  # noqa: E402

EMPTY_DXBC = b"DXBC" + bytes(16) + struct.pack("<III", 1, 32, 0)


def translate_fixture(name: str) -> str:
    """Translate runtime/test/shaders/<name>.asm via rdna2spirv.py without needing llvm-mc."""
    asm_path = SHADER_DIR / f"{name}.asm"
    lines = []
    for raw in asm_path.read_text(encoding="utf-8").splitlines():
        code = raw.split(";", 1)[0].strip()
        if code:
            lines.append("\t" + code)
    with tempfile.TemporaryDirectory(prefix="leonida-mimg-") as tmp:
        tmp_path = Path(tmp)
        disasm = tmp_path / "disasm"
        disasm.mkdir()
        if name.startswith("cs_"):
            fssm = tmp_path / "fssm"
            fssm.mkdir()
            (fssm / f"{name}.fssm").write_bytes(compute_fixture.compute_header(str(asm_path)))
        else:
            (tmp_path / f"{name}.dxbc").write_bytes(EMPTY_DXBC)
        s_path = disasm / f"{name}.s"
        s_path.write_text(f"; {name} code 0x0..0x100\n" + "\n".join(lines) + "\n", encoding="utf-8")
        return rdna2spirv.translate(str(s_path))


class MimgFixtureTranslation(unittest.TestCase):
    def test_cs_imgatomic_emits_all_ten_atomics_for_2d_and_3d(self):
        spv = translate_fixture("cs_imgatomic")
        caps = set(re.findall(r"^\s*OpCapability (\w+)", spv, re.M))
        self.assertNotIn("StorageImageWriteWithoutFormat", caps)
        self.assertNotIn("StorageImageReadWithoutFormat", caps)
        self.assertIn("OpDecorate %simages Binding 4", spv)
        self.assertIn("OpDecorate %simages3d Binding 7", spv)
        for op in ("OpAtomicIAdd", "OpAtomicISub", "OpAtomicAnd", "OpAtomicOr", "OpAtomicXor",
                   "OpAtomicSMin", "OpAtomicSMax", "OpAtomicUMin", "OpAtomicUMax", "OpAtomicCompareExchange"):
            with self.subTest(op=op):
                self.assertIn(f"= {op} %uint ", spv)
        # Both 2D (%simages) and 3D (%simages3d) paths emit OpImageTexelPointer
        self.assertIn("OpAccessChain %_ptr_UC_simg %simages ", spv)
        self.assertIn("OpAccessChain %_ptr_UC_simg3 %simages3d ", spv)
        self.assertGreaterEqual(spv.count("OpAtomicCompareExchange"), 4)

    def test_ps_get_lod_emits_2d_and_2d_array_queries(self):
        spv = translate_fixture("ps_get_lod")
        self.assertIn("OpCapability ImageQuery", spv)
        self.assertIn("OpDecorate %images Binding 0", spv)
        self.assertIn("OpDecorate %images_arr Binding 3", spv)
        self.assertEqual(spv.count("OpImageQueryLod %v2float "), 4)
        self.assertIn("OpSampledImage %simg2d ", spv)
        self.assertIn("OpSampledImage %simg2da ", spv)

    def test_ps_get_resinfo_emits_size_and_level_queries(self):
        spv = translate_fixture("ps_get_resinfo")
        self.assertIn("OpCapability ImageQuery", spv)
        self.assertEqual(spv.count("OpImageQuerySizeLod %v2int "), 3)
        self.assertEqual(spv.count("OpImageQueryLevels %int "), 3)

    def test_cs_bvh32_translates_11_operand_32bit_bvh(self):
        spv = translate_fixture("cs_bvh32")
        self.assertIn("OpCapability PhysicalStorageBufferAddresses", spv)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_arr ", spv)
        # 32-bit node id widens via OpUConvert %ulong without shifting a second node_hi register
        self.assertGreaterEqual(spv.count("OpUConvert %ulong "), 4)

    def test_fixtures_pass_spirv_val(self):
        if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
            self.skipTest("spirv-as / spirv-val not in PATH")
        for name in ("cs_imgatomic", "ps_get_lod", "ps_get_resinfo", "cs_bvh32"):
            with self.subTest(name=name):
                spv_text = translate_fixture(name)
                with tempfile.TemporaryDirectory(prefix="leonida-val-") as tmp:
                    spv_bin = Path(tmp) / f"{name}.spv"
                    p = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-o", str(spv_bin), "-"],
                                       input=spv_text, capture_output=True, text=True)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    v = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(spv_bin)],
                                       capture_output=True, text=True)
                    self.assertEqual(v.returncode, 0, v.stderr)


class TextureQueryChecks(unittest.TestCase):
    def test_texture_query_problem_accepts_exact_and_rejects_bad_frames(self):
        want = (1.0, 3.0, -1.0, 2.0)
        good = struct.pack("<4f", *want) * (64 * 64)
        self.assertEqual(check.texture_query_problem(good, want), "")

        swapped = struct.pack("<4f", 3.0, 1.0, -1.0, 2.0) * (64 * 64)
        self.assertIn("4096 of 4096 pixels differ", check.texture_query_problem(swapped, want))

        with_nan = struct.pack("<4f", float("nan"), 3.0, -1.0, 2.0) + good[16:]
        self.assertIn("1 of 4096 pixels differ", check.texture_query_problem(with_nan, want))

        self.assertIn("expected 4096 RGBA32F pixels", check.texture_query_problem(good[:-16], want))

    def test_texture_query_test_skips_on_77_and_fails_on_error(self):
        for name in check.TEXTURE_QUERY_TESTS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "vs_ids.spv").touch()
                (Path(tmp) / f"{name}.spv").touch()
                with patch.object(check, "run", return_value=(77, "unsupported: need 144 push constant bytes\n")):
                    res = check.texture_query_test(tmp, tmp, name)
                self.assertIsInstance(res, check.Skip)
                self.assertIn("144 push constant bytes", res)

                with patch.object(check, "run", return_value=(1, "device lost\n")):
                    res = check.texture_query_test(tmp, tmp, name)
                self.assertNotIsInstance(res, check.Skip)
                self.assertIn("device lost", res)

    def test_run_gpu_tests_routes_new_mimg_fixtures_and_skips_on_77(self):
        for name, expected_skip in (
            ("cs_imgatomic", "unsupported: VK_FORMAT_R32_UINT lacks VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT"),
            ("ps_get_lod", "unsupported: need 144 push constant bytes, device has 128"),
            ("ps_get_resinfo", "unsupported: need 144 push constant bytes, device has 128"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(filter=f"{name}.asm", build=tmp, llvm_mc="unused")

                def fake_run_cmd(cmd, env=None, gpu=True):
                    if not gpu:
                        for stem in (name, "vs_ids"):
                            (Path(tmp) / "spirv" / f"{stem}.spv").touch()
                        return 0, ""
                    return 77, expected_skip

                out = io.StringIO()
                with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                      patch.object(rgt, "spirv_as"),
                      patch.object(rgt, "run_cmd", side_effect=fake_run_cmd),
                      patch.object(check, "run", return_value=(77, expected_skip)),
                      contextlib.redirect_stdout(out)):
                    rc = rgt.run_suite(args, tmp)
                self.assertEqual(rc, 0, out.getvalue())
                self.assertIn(f"SKIP ({expected_skip})", out.getvalue())
                self.assertIn("0/1 passed, 1 skipped", out.getvalue())


if __name__ == "__main__":
    unittest.main()
