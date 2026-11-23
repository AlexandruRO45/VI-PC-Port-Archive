"""Unit tests for tools/export_diff.py, the export-shape differential fuzzer (FUZZ-114). Standard library only.

    python -I tests/test_export_diff.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True
for _p in (os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests", "semantic")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import support  # noqa: E402
import export_diff  # noqa: E402

VS_PROBE_LINES = ["v_mov_b32_e32 v0, 1.0", "exp pos0 v0, v0, v0, v0 done vm", "s_endpgm"]
VS_CLIP_LINES = ["v_mov_b32_e32 v0, 1.0", "exp pos1 v0, v0, v0, v0", "exp pos0 v0, v0, v0, v0 done vm",
                 "s_endpgm"]


class Osg1BuilderTest(unittest.TestCase):
    def test_osg1_with_clip_and_cull_parses(self):
        dxbc = export_diff.osg1_dxbc([
            export_diff.Osg1Element(export_diff.D3D_NAME["clip"], 0, 0, 0x8, 0x0),
            export_diff.Osg1Element(export_diff.D3D_NAME["cull"], 0, 0, 0x2, 0x0),
        ])
        spvasm = support.translate_lines("vs_osg1_probe", VS_PROBE_LINES, dxbc=dxbc)
        self.assertIn("OpDecorate %out_pos BuiltIn Position", spvasm)

    def test_declared_clip_distance_reaches_the_module(self):
        # The translator reads the OSG1 the builder made: an SV_ClipDistance0 element (x component) puts the pos1
        # export into %out_clip, so the bytes are consumed, not merely tolerated.
        dxbc = export_diff.osg1_dxbc([export_diff.Osg1Element(export_diff.D3D_NAME["clip"], 0, 0, 0x1, 0x0)])
        spvasm = support.translate_lines("vs_clip_probe", VS_CLIP_LINES, dxbc=dxbc)
        self.assertIn("OpDecorate %out_clip BuiltIn ClipDistance", spvasm)

    def test_element_layout_matches_the_translator_reader(self):
        # output_signature_full() reads (register, semantic_index, system_value, e[24], e[25]) from each 32-byte
        # element; the builder must put exactly those values where that reader looks.
        import struct
        dxbc = export_diff.osg1_dxbc([export_diff.Osg1Element(23, 7, 3, 0x5, 0xA)])
        self.assertEqual(dxbc[:4], b"DXBC")
        count = struct.unpack_from("<I", dxbc, 28)[0]
        self.assertEqual(count, 1)
        off = struct.unpack_from("<I", dxbc, 32)[0]
        self.assertEqual(dxbc[off:off + 4], b"OSG1")
        payload = dxbc[off + 8:off + 8 + struct.unpack_from("<I", dxbc, off + 4)[0]]
        n, eo = struct.unpack_from("<II", payload, 0)
        self.assertEqual((n, eo), (1, 8))
        elem = payload[eo:eo + 32]
        _, _, sem_idx, sysv, typ, reg = struct.unpack_from("<6I", elem, 0)
        self.assertEqual((sem_idx, sysv, typ, reg), (7, 23, 3, 3))
        self.assertEqual((elem[24], elem[25]), (0x5, 0xA))
        self.assertEqual(len(dxbc) % 4, 0)


class PsModelTest(unittest.TestCase):
    E = export_diff.PsExport

    def test_sparse_targets(self):
        # SHADER-210: export mrtK writes the K-th declared SV_Target; the variable carries that target's register.
        pred = export_diff.ps_model(export_diff.PsShape([0, 2], True),
                                    [self.E("mrt", 0, False, False), self.E("mrt", 1, False, False)])
        self.assertEqual(pred.locations, {"%out_mrt0": 0, "%out_mrt2": 2})
        self.assertEqual(pred.refusals, [])

    def test_no_osg1_locations(self):
        pred = export_diff.ps_model(export_diff.PsShape([], False),
                                    [self.E("mrt", 0, False, False), self.E("mrt", 3, False, False)])
        self.assertEqual(pred.locations, {"%out_mrt0": 0, "%out_mrt3": 3})

    def test_empty_osg1_is_identity(self):
        # An OSG1 that declares no SV_Target: rdna2spirv.py:14340-14342 takes the export index as the target.
        pred = export_diff.ps_model(export_diff.PsShape([], True), [self.E("mrt", 2, False, False)])
        self.assertEqual(pred.locations, {"%out_mrt2": 2})

    def test_past_declared_refused(self):
        pred = export_diff.ps_model(export_diff.PsShape([0], True),
                                    [self.E("mrt", 0, False, False), self.E("mrt", 1, False, False)])
        self.assertEqual(pred.refusals, ["mrt1"])
        self.assertEqual(pred.locations, {"%out_mrt0": 0})

    def test_mrtz_compr_is_not_depth(self):
        compr = export_diff.ps_structure(export_diff.ps_model(export_diff.PsShape([], False),
                                                              [self.E("mrtz", 0, True, True)]))
        self.assertEqual(compr, [])
        plain = export_diff.ps_structure(export_diff.ps_model(export_diff.PsShape([], False),
                                                              [self.E("mrtz", 0, True, False)]))
        self.assertEqual(plain, ["BuiltIn %out_depth FragDepth"])
        off = export_diff.ps_structure(export_diff.ps_model(export_diff.PsShape([], False),
                                                            [self.E("mrtz", 0, False, False)]))
        self.assertEqual(off, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
