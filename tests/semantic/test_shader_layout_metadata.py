"""Delayed index metadata must describe the layouts used to emit its module.

The CLI indexes immediately. Library callers can retain (generator, assembly),
restore options, and index later, or interleave modules with different options.
These tests translate synthetic programs and compare the delayed metadata with
both the emitted declarations and the immediate-indexing control.
"""
import contextlib
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support
import rdna2spirv
from test_mesh_cull_store import _dxbc, _mesh_psv0

COMPUTE = ["v_mov_b32 v0, 123", "v_mov_b32 v1, 0",
           "buffer_store_dword v0, v1, s[0:3], 0 offen", "s_endpgm"]
VERTEX = ["v_mov_b32 v1, 0x42004000", "exp pos0 v1, v1, v1, v1 done",
          "exp param0 v1, v1, v1, v1", "s_endpgm"]
PIXEL = ["v_interp_p1ll_f16 v2, v0, attr0.x", "v_interp_p2_f16 v2, v1, attr0.x, v2",
         "v_interp_p1ll_f16 v3, v0, attr0.x high", "v_interp_p2_f16 v3, v1, attr0.x, v3 high",
         "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm"]
OPTIONS = ("PC_TAIL_UBO", "F16_HALVES", "MESH_COMPUTE")


@contextlib.contextmanager
def options(tail=False, halves=False, mesh_compute=False):
    old = [getattr(rdna2spirv, key) for key in OPTIONS]
    for key, value in zip(OPTIONS, (tail, halves, mesh_compute)):
        setattr(rdna2spirv, key, value)
    try:
        yield
    finally:
        for key, value in zip(OPTIONS, old):
            setattr(rdna2spirv, key, value)


def translate(stage, *, tail=False, halves=False, mesh_compute=False, reads_tail=True):
    kwargs = {}
    if stage == "cs":
        lines = COMPUTE if reads_tail else ["s_endpgm"]
        header = bytearray(0x148)
        struct.pack_into("<4I", header, 0x138, 0x88, 32, 1, 1)
        kwargs["fssm"] = bytes(header)
    elif stage == "ms":
        source = pathlib.Path(support.ROOT, "runtime", "test", "shaders", "ms_tri.asm")
        lines = [line.split(";")[0].strip() for line in source.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line]
        kwargs["dxbc"] = _dxbc([(b"PSV0", _mesh_psv0(128, 64, 64))])
    else:
        lines = VERTEX if stage == "vs" else PIXEL
    with options(tail, halves, mesh_compute):
        asm = support.translate_lines(stage + "_layout_metadata", lines, **kwargs)
        gen = rdna2spirv.translate.last_gen
        immediate = rdna2spirv.shader_meta(gen, asm)
    return gen, asm, immediate


class ShaderLayoutMetadataTests(unittest.TestCase):
    def test_push_constant_layout_survives_option_restore(self):
        for tail in (False, True):
            for reads_tail in (False, True):
                with self.subTest(tail=tail, reads_tail=reads_tail):
                    gen, asm, immediate = translate("cs", tail=tail, reads_tail=reads_tail)
                    with options(tail=not tail):
                        meta = rdna2spirv.shader_meta(gen, asm)
                    self.assertEqual(meta, immediate)
                    self.assertEqual(meta["push_constant_bytes"], 128 if tail else 144)
                    self.assertEqual("%PcTail = OpTypeStruct" in asm, tail and reads_tail)
                    if tail and reads_tail:
                        self.assertEqual(meta["pc_tail"], {"set": 1, "binding": 0, "bytes": 16})
                        self.assertIn("OpDecorate %pc_tail DescriptorSet 1", asm)
                    else:
                        self.assertNotIn("pc_tail", meta)
                    if tail:
                        self.assertNotIn("OpMemberDecorate %UserData 1 Offset 128", asm)
                    else:
                        self.assertIn("OpMemberDecorate %UserData 4 Offset 140", asm)

    def test_f16_interfaces_survive_option_restore(self):
        for stage in ("vs", "ps", "ms"):
            for halves in (False, True):
                with self.subTest(stage=stage, halves=halves):
                    gen, asm, immediate = translate(stage, halves=halves)
                    with options(halves=not halves):
                        meta = rdna2spirv.shader_meta(gen, asm)
                    self.assertEqual(meta, immediate)
                    if halves:
                        self.assertEqual(meta["f16_halves"], {"0": [28, 29]})
                        self.assertIn("Location 28", asm)
                        self.assertIn("Location 29", asm)
                    else:
                        self.assertNotIn("f16_halves", meta)

    def test_mesh_compute_layout_survives_option_restore(self):
        for tail in (False, True):
            for halves in (False, True):
                with self.subTest(tail=tail, halves=halves):
                    gen, asm, immediate = translate("ms", tail=tail, halves=halves, mesh_compute=True)
                    with options(tail=not tail, halves=not halves):
                        meta = rdna2spirv.shader_meta(gen, asm)
                    self.assertEqual(meta, immediate)
                    self.assertIn('OpEntryPoint GLCompute %main "main"', asm)
                    layout = meta["mesh_compute"]
                    self.assertEqual(meta["push_constant_bytes"], 128 if tail else 152)
                    self.assertEqual(layout["push_constant_bytes"], meta["push_constant_bytes"])
                    if tail:
                        self.assertEqual(meta["pc_tail"], {"set": 1, "binding": 0, "bytes": 24})
                        self.assertEqual(layout["pc_tail_address_dwords"], [4, 5])
                    else:
                        self.assertNotIn("pc_tail", meta)
                        self.assertEqual(layout["address_dwords"], [36, 37])
                    if halves:
                        self.assertEqual(meta["f16_halves"], {"0": [28, 29]})
                        self.assertEqual(layout["f16_halves"], meta["f16_halves"])
                    else:
                        self.assertNotIn("f16_halves", meta)
                        self.assertNotIn("f16_halves", layout)

    def test_interleaved_modules_keep_their_own_layout(self):
        emitted = [translate("cs", tail=True), translate("vs", halves=True),
                   translate("cs", tail=False), translate("ms", mesh_compute=True)]
        for tail, halves in ((False, False), (True, True)):
            with options(tail=tail, halves=halves):
                for gen, asm, immediate in emitted:
                    with self.subTest(tail=tail, halves=halves, stage=gen.stage):
                        self.assertEqual(rdna2spirv.shader_meta(gen, asm), immediate)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs SPIRV-Tools")
    def test_emitted_layout_modules_validate(self):
        cases = [("cs", tail, False, False) for tail in (False, True)]
        cases += [(stage, False, halves, False) for stage in ("vs", "ps", "ms") for halves in (False, True)]
        cases += [("ms", tail, halves, True) for tail in (False, True) for halves in (False, True)]
        with tempfile.TemporaryDirectory() as tmp:
            for stage, tail, halves, mesh_compute in cases:
                with self.subTest(stage=stage, tail=tail, halves=halves, mesh_compute=mesh_compute):
                    _, asm, _ = translate(stage, tail=tail, halves=halves, mesh_compute=mesh_compute)
                    source, binary = pathlib.Path(tmp, "shader.spvasm"), pathlib.Path(tmp, "shader.spv")
                    source.write_text(asm, encoding="utf-8", newline="\n")
                    for command in (["spirv-as", "--target-env", "vulkan1.2", str(source), "-o", str(binary)],
                                    ["spirv-val", "--target-env", "vulkan1.2", str(binary)]):
                        result = subprocess.run(command, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
