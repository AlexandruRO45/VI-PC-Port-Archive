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


class VsModelTest(unittest.TestCase):
    ON = (1, 1, 1, 1)
    N = export_diff.D3D_NAME

    def svs(self, clip=0, cull=0, misc=()):
        return [export_diff.Osg1Element(self.N["clip"], i, 0, 0x1, 0) for i in range(clip)] + \
               [export_diff.Osg1Element(self.N["cull"], i, 0, 0x1, 0) for i in range(cull)] + \
               [export_diff.Osg1Element(self.N[m], 0, 1, 0x1, 0) for m in misc]

    def pos(self, i, comps=ON):
        return export_diff.VsExport("pos", i, comps)

    def test_distances_start_at_pos1(self):
        shape = export_diff.VsShape(self.svs(clip=2, cull=1), True)
        pred = export_diff.vs_model(shape, [self.pos(i) for i in range(4)])
        self.assertEqual(pred.distances, [(1, 0, "clip", 0), (1, 1, "clip", 1), (1, 2, "cull", 0)])
        self.assertIsNone(pred.misc)
        self.assertEqual((pred.clip_count, pred.cull_count), (2, 1))
        self.assertEqual(export_diff.vs_structure(pred), sorted([
            "BuiltIn %out_pos Position", "BuiltIn %out_clip ClipDistance", "Array %out_clip 2",
            "BuiltIn %out_cull CullDistance", "Array %out_cull 1"]))

    def test_misc_vector_shifts_to_pos2(self):
        shape = export_diff.VsShape(self.svs(clip=1, misc=("rtai",)), True)
        pred = export_diff.vs_model(shape, [self.pos(i) for i in range(4)])
        self.assertEqual(pred.misc, "pos1")
        self.assertEqual(pred.distances, [(2, 0, "clip", 0)])
        self.assertTrue(pred.layer)
        self.assertFalse(pred.viewport)
        self.assertIn("BuiltIn %out_layer Layer", export_diff.vs_structure(pred))

    def test_misc_pos1_z_off_writes_no_misc_builtin(self):
        shape = export_diff.VsShape(self.svs(clip=1, misc=("rtai", "vai")), True)
        pred = export_diff.vs_model(shape, [self.pos(0), self.pos(1, (1, 1, 0, 1)), self.pos(2)])
        self.assertFalse(pred.layer)
        self.assertFalse(pred.viewport)
        self.assertEqual(pred.distances, [(2, 0, "clip", 0)])

    def test_no_osg1_discards(self):
        pred = export_diff.vs_model(export_diff.VsShape([], False), [self.pos(i) for i in range(4)])
        self.assertEqual(pred.discarded, [1, 2, 3])
        self.assertEqual(export_diff.vs_structure(pred), ["BuiltIn %out_pos Position"])

    def test_empty_osg1_vs(self):
        pred = export_diff.vs_model(export_diff.VsShape([], True), [self.pos(i) for i in range(4)])
        self.assertEqual(pred.discarded, [1, 2, 3])
        self.assertEqual(export_diff.vs_structure(pred), ["BuiltIn %out_pos Position"])

    def test_over_eight_floats_clamps(self):
        # "up to 8 floats total": the ninth clip component is dropped, nothing is refused.
        shape = export_diff.VsShape(self.svs(clip=9, cull=1), True)
        pred = export_diff.vs_model(shape, [self.pos(i) for i in range(4)])
        self.assertEqual((pred.clip_count, pred.cull_count), (8, 0))
        self.assertEqual(len(pred.distances), 8)
        self.assertEqual(pred.distances[:4], [(1, c, "clip", c) for c in range(4)])
        self.assertEqual(pred.distances[4:], [(2, c, "clip", 4 + c) for c in range(4)])
        self.assertEqual(pred.refusals, [])

    def test_params(self):
        shape = export_diff.VsShape(self.svs(), True)
        pred = export_diff.vs_model(shape, [self.pos(0), export_diff.VsExport("param", 0, self.ON),
                                            export_diff.VsExport("param", 5, self.ON)])
        self.assertEqual(pred.params, {0: 0, 5: 5})
        self.assertIn("Location %out_param5 5", export_diff.vs_structure(pred))

    def test_all_off_param_still_declares_its_output(self):
        # "exp paramN -> location N" has no exception for off components, and the translator registers the output
        # for any such export (rdna2spirv.py:14349). Found by the first randomized run (seed 1, program 4).
        pred = export_diff.vs_model(export_diff.VsShape([], True),
                                    [self.pos(0), export_diff.VsExport("param", 4, (0, 0, 0, 0))])
        self.assertEqual(pred.params, {4: 4})
        self.assertIn("Location %out_param4 4", export_diff.vs_structure(pred))


class HarnessTest(unittest.TestCase):
    E = export_diff.PsExport
    TWO_EXPORTS = ["v_mov_b32_e32 v0, 1.0", "v_mov_b32_e32 v1, 2.0", "exp mrt0 v0, v0, v0, v0",
                   "exp mrt1 v1, v1, v1, v1 done vm", "s_endpgm"]

    def sparse_module(self):
        elems = [export_diff.Osg1Element(export_diff.D3D_NAME["target"], n, n, 0xF, 0xF) for n in (0, 2)]
        return support.translate_lines("ps_cmp_probe", self.TWO_EXPORTS, dxbc=export_diff.osg1_dxbc(elems))

    def test_generate_is_deterministic(self):
        a, b = export_diff.generate(7, 3), export_diff.generate(7, 3)
        self.assertEqual((a.listing, a.dxbc, a.expected, a.refusals), (b.listing, b.dxbc, b.expected, b.refusals))
        self.assertNotEqual(export_diff.generate(7, 3).listing, export_diff.generate(7, 4).listing)

    def test_compare_catches_swap(self):
        spvasm = self.sparse_module()
        self.assertIsNone(export_diff.compare(spvasm, ["Location %out_mrt0 0", "Location %out_mrt2 2"]))
        message = export_diff.compare(spvasm, ["Location %out_mrt0 0", "Location %out_mrt2 1"])
        self.assertIn("%out_mrt2", message)

    def test_self_test_canaries(self):
        self.assertEqual(export_diff.main(["--self-test"]), 0)
        self.assertTrue(any("clip/cull" in name for name in export_diff.CANARIES))

    def test_happy_run_contract(self):
        self.assertIn(export_diff.main(["--programs", "5", "--seed", "1"]), (0, 1))

    def test_all_skipped_is_harness_error(self):
        self.assertEqual(export_diff.main(["--programs", "0", "--seed", "1"]), 2)

    def test_calibration_sparse_targets(self):
        for targets in ([0, 2], [1, 3], [0, 7]):
            case = export_diff.make_case(f"ps_cal_{''.join(map(str, targets))}", "ps", export_diff.PsShape(targets, True),
                                         [self.E("mrt", k, False, False) for k in range(len(targets))])
            self.assertEqual(export_diff.run_case(case)[0], "match", targets)

    def test_empty_osg1_ps_matches_translator(self):
        case = export_diff.make_case("ps_empty_osg1", "ps", export_diff.PsShape([], True), [self.E("mrt", 2, False, False)])
        self.assertEqual(export_diff.run_case(case)[0], "match")

    def test_past_declared_is_an_expected_skip(self):
        case = export_diff.make_case("ps_past", "ps", export_diff.PsShape([0], True),
                                     [self.E("mrt", 0, False, False), self.E("mrt", 1, False, False)])
        self.assertEqual(export_diff.run_case(case)[0], "skip")

    def test_partial_mrt_matches_translator(self):
        # EN gates each component of a colour export: the output is still declared at its location.
        case = export_diff.make_case("ps_partial_mrt", "ps", export_diff.PsShape([], False),
                                     [self.E("mrt", 0, False, False, (1, 0, 1, 0))])
        self.assertIn("exp mrt0 v0, off, v2, off done vm", case.listing)
        self.assertEqual(export_diff.run_case(case)[0], "match")

    def test_vs_probe_matches_translator(self):
        svs = [export_diff.Osg1Element(2, 0, 0, 0x3, 0), export_diff.Osg1Element(3, 0, 1, 0x1, 0),
               export_diff.Osg1Element(4, 0, 2, 0x1, 0)]
        exports = [export_diff.VsExport("pos", 0, (1, 1, 1, 1)), export_diff.VsExport("pos", 1, (1, 1, 1, 1)),
                   export_diff.VsExport("pos", 2, (1, 1, 1, 1)), export_diff.VsExport("param", 3, (1, 1, 1, 1))]
        case = export_diff.make_case("vs_probe_cal", "vs", export_diff.VsShape(svs, True), exports)
        self.assertEqual(export_diff.run_case(case)[0], "match", export_diff.run_case(case)[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
