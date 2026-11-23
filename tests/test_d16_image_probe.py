"""tools/d16_image_probe.py: the textures and samplers, the kernels (they assemble for gfx1031), the comparison and its
exit codes through a fake oracle that plays the GPU, and the exit-77 contract. The last class runs on the hardware
when the oracle tools are present and the GPU is GFX10, and is skipped otherwise. Standard library only."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import d16_format_probe as fp  # noqa: E402
import d16_image_probe as ip  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


class Textures(unittest.TestCase):
    def test_self_test(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ip.self_test(), 0)

    def test_every_value_of_a_16_bit_format_is_one_texel(self):
        seen = [v for b in range(ip.blocks("unorm16")) for v in ip.block_values("unorm16", b)]
        self.assertEqual(seen, list(range(65536)))

    def test_the_float32_set_starts_with_the_two_unorm_halfway_quotients(self):
        v, ties = ip.float32_texels()
        self.assertEqual([fp.f32_bits(fp.to_f32(fp.Fraction(65519, 65535))),
                          fp.f32_bits(fp.to_f32(fp.Fraction(65455, 65535)))], [v[0], v[2]])
        self.assertTrue(all((b & 0x1FFF) == 0x1000 for b in v[:ties]), "the ties come first")
        self.assertEqual(len(v), ip.TEXELS)

    def test_t_sharp_carries_the_format(self):
        for fmt, (code, _, _, width) in ip.FORMATS.items():
            w = ip.t_words("gfx1031", fmt)
            self.assertEqual((w[1] >> 20) & 0x1FF, code, fmt)
            self.assertEqual(w[2] >> 31, 1)                        # RESOURCE_LEVEL on GFX10
            self.assertEqual(((w[2] & 0xFFF) << 2 | w[1] >> 30) + 1, width)


class Kernels(unittest.TestCase):
    def test_kernels_assemble_for_rdna2(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        for wave in (32, 64):
            kernels = [asm.Kernel(ip.kernel_lines("gfx1031", f"{f}_0", f, ip.block_values(f, 0), 0.5), wave,
                                  ip.TEXELS, ip.TEXELS * ip.WORDS_PER_LANE, raw=True, vgprs=24, sgprs=32)
                       for f in ip.FORMATS]
            text, _, _ = asm.module_text(kernels, "gfx1031", wave=wave)
            with tempfile.TemporaryDirectory() as d:
                s = Path(d) / "k.s"
                s.write_text(text, encoding="utf-8")
                r = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1031",
                                    *(["-mwavefrontsize64"] if wave == 64 else []), "-c", str(s), "-o",
                                    str(Path(d) / "k.o")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[:800])
            self.assertIn("image_sample_lz v13, v[16:17], s[8:15], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm d16",
                          text)


def _val(kind, n, v, rule):
    return fp.models16(kind, n, v)[rule]


class FakeOracle:
    """Plays the GPU from each kernel's label (.Lpc<fmt>_<block>) and its texture: d16 results by `rule`, f32 loads
    correctly rounded, bilinear f32 the rounded exact mix and its d16 rounded from that f32 to nearest even."""
    arch = "gfx1031"
    device = "fake gfx1031"

    def __init__(self, rule="rte", f32_off=False, wave64_flip=False, tie="even"):
        self.rule, self.f32_off, self.wave64_flip, self.tie = rule, f32_off, wave64_flip, tie

    def run(self, kernels, words=4):
        out = []
        for k in kernels:
            fmt, block = re.match(r"\.Lpc(\w+?)_(\d+):", k.lines[1]).groups()
            _, kind, n, width = ip.FORMATS[fmt]
            vals = ip.block_values(fmt, int(block))
            frac = struct.unpack("<f", struct.pack("<I", int(re.search(
                r"v_add_f32 v16, 0x([0-9a-f]+)", "\n".join(k.lines)).group(1), 16)))[0]
            res = []
            for lane, v in enumerate(vals):
                right = vals[lane + 1] if lane % width + 1 < width else v
                if kind in ("unorm", "snorm"):
                    load = fp.models32(kind, n, v)["f32_rte"] + (1 if self.f32_off else 0)
                    d16 = _val(kind, n, v, self.rule)
                    mix = fp.quotient(kind, n, v) * (1 - fp.Fraction(frac)) + fp.quotient(kind, n, right) * \
                        fp.Fraction(frac)
                elif kind == "f16":
                    load = fp.f32_bits(struct.unpack("<e", struct.pack("<H", v))[0]) if (v & 0x7C00) != 0x7C00 \
                        else 0x7FC00000
                    d16, mix = v, fp.Fraction(0)
                else:
                    x = fp.Fraction(struct.unpack("<f", struct.pack("<I", v))[0])
                    load = v
                    d16 = fp.f16_bits(fp.to_f16(x, self.tie == "down" and (v & 0x1FFF) == 0x1000))
                    mix = fp.Fraction(0)
                if self.wave64_flip and k.wave == 64 and lane == 5:
                    d16 ^= 1
                b32 = fp.f32_bits(fp.to_f32(mix))
                b16 = fp.f16_bits(fp.to_f16(fp.to_f32(mix)))
                hi = ip.SEED & 0xFFFF0000
                res += [load, hi | d16, load, hi | d16, b32, hi | b16, lane, 0]
            out.append(res)
        return out

    def close(self):
        pass


def run_main(oracle, *args):
    buf = io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle), redirect_stdout(buf):
        rc = ip.main(list(args))
    return rc, buf.getvalue()


class Compare(unittest.TestCase):
    def test_single_rounding_hardware_is_four_values_off_the_translator(self):
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "r.json")
            rc, out = run_main(FakeOracle(), "--formats", "unorm16,snorm16", "--waves", "32", "--json", j)
            with open(j, encoding="utf-8") as f:
                rep = json.load(f)
        self.assertEqual(rc, 0, out)
        for fmt in ("unorm16", "snorm16"):
            e = rep["formats"][fmt]
            self.assertEqual(e["fields"]["load_d16"]["off_f32_rte"], 2)
            self.assertEqual(e["fields"]["load_d16"]["match"]["rte"], 65536)
            self.assertEqual(e["fields"]["bilinear"]["d16_is_rte_of_own_f32"], 65536)
            self.assertEqual(e["f32_load_is_rounded_quotient"], 65536)
        self.assertEqual(rep["formats"]["unorm16"]["fields"]["load_d16"]["first_off"][0]["value"], "0xffaf")

    def test_float32_halfway_points_are_counted_apart(self):
        for tie, key in (("even", "to_even"), ("down", "toward_zero")):
            with tempfile.TemporaryDirectory() as d:
                j = os.path.join(d, "r.json")
                rc, out = run_main(FakeOracle(tie=tie), "--formats", "float32", "--waves", "32", "--json", j)
                with open(j, encoding="utf-8") as f:
                    h = json.load(f)["formats"]["float32"]["fields"]["load_d16"]["halfway"]
            self.assertEqual(rc, 0, out)
            self.assertEqual(h[key] + h["either"], h["inputs"], tie)
            self.assertEqual(h["other"], 0)
            self.assertGreaterEqual(h[key], 20)                 # ties whose even neighbour is away from zero
            self.assertEqual(h["inputs"], ip.float32_texels()[1])

    def test_an_f32_load_one_ulp_off_is_counted_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "r.json")
            rc, out = run_main(FakeOracle(f32_off=True), "--formats", "unorm8", "--waves", "32", "--json", j)
            with open(j, encoding="utf-8") as f:
                e = json.load(f)["formats"]["unorm8"]
        self.assertEqual(rc, 0, out)
        self.assertEqual(e["f32_load_is_rounded_quotient"], 0)

    def test_a_short_runner_result_is_a_harness_error(self):
        """zip() would drop the tail and the counts would quietly cover fewer texels (BUILD-733 in d16_format_probe)."""
        short = FakeOracle()
        full = short.run

        def truncated(kernels, words=4):
            return full(kernels, words)[:-1]

        short.run = truncated
        rc, out = run_main(short, "--formats", "unorm16", "--waves", "32")
        self.assertEqual(rc, 2, out)
        self.assertIn("returned 63 results for 64 kernels", out)

    def test_a_wave_difference_exits_1(self):
        rc, out = run_main(FakeOracle(wave64_flip=True), "--formats", "unorm8,float16")
        self.assertEqual(rc, 1, out)

    def test_not_gfx10_exits_77(self):
        o = FakeOracle()
        o.arch = "gfx1150"
        rc, out = run_main(o, "--formats", "unorm8")
        self.assertEqual(rc, 77, out)


class FakeArrayOracle:
    """Plays the GPU for the --dims kernels (label .Lpc<dim>_<fmt>_<block>): d16 by the hardware rule ("hw": the
    quotient rounded once, 32_FLOAT toward zero) or the old translator's ("translator": f32 first, then nearest even),
    the four gather taps lanes t..t+3 of the kernel, `point_off` a point sample that differs from the load."""
    arch = "gfx1031"
    device = "fake gfx1031"

    def __init__(self, rule="hw", point_off=False):
        self.rule, self.point_off = rule, point_off

    def d16(self, kind, n, v):
        if kind in ("unorm", "snorm"):
            return _val(kind, n, v, "rte" if self.rule == "hw" else "f32_rte")
        if kind == "f16":
            return v
        x = fp.Fraction(struct.unpack("<f", struct.pack("<I", v))[0])
        return ip._f16_signed(x, v, self.rule == "hw")

    @staticmethod
    def f32(kind, n, v):
        if kind in ("unorm", "snorm"):
            return fp.models32(kind, n, v)["f32_rte"]
        if kind == "f16":
            return fp.f32_bits(struct.unpack("<e", struct.pack("<H", v))[0])
        return v

    def run(self, kernels, words=4):
        out = []
        for k in kernels:
            dim, fmt, block = re.match(r"\.Lpc(2d_array|cube)_(\w+?)_(\d+):", k.lines[1]).groups()
            _, kind, n, _ = ip.FORMATS[fmt]
            vals = ip.array_values(fmt, int(block))
            res = []
            hi = ip.SEED & 0xFFFF0000
            for lane, v in enumerate(vals):
                taps = [vals[(lane + i) % len(vals)] for i in range(4)]
                h = [self.d16(kind, n, t) for t in taps]
                load = self.f32(kind, n, v)
                res += [load, hi | self.d16(kind, n, v), load ^ (lane == 7 and self.point_off),
                        hi | self.d16(kind, n, v), *[self.f32(kind, n, t) for t in taps],
                        h[0] | h[1] << 16, h[2] | h[3] << 16, lane, 0]
            out.append(res)
        return out

    def close(self):
        pass


class ArrayDims(unittest.TestCase):
    def test_t_sharp_carries_type_layers_and_format(self):
        for dim, (kind, _) in ip.ARRAY_DIMS.items():
            for fmt in ip.ARRAY_FORMATS:
                with self.subTest(dim=dim, fmt=fmt):
                    w = ip.array_t_words(dim, fmt)
                    width, rows, layers, _, _ = ip.array_geometry(dim, fmt)
                    self.assertEqual((w[3] >> 28, w[4], (w[1] >> 20) & 0x1FF), (kind, layers - 1, ip.FORMATS[fmt][0]))
                    self.assertEqual((((w[2] & 0xFFF) << 2 | w[1] >> 30) + 1, (w[2] >> 14 & 0x3FFF) + 1), (width, rows))
                    self.assertEqual(width * ip.FORMATS[fmt][2] // 8, 256)       # rows are exactly 256 bytes
        self.assertEqual(ip.array_geometry("cube", "unorm16")[:3], (128, 128, 6))

    def test_kernels_assemble_for_rdna2(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        for wave in (32, 64):
            kernels = [asm.Kernel(ip.array_kernel_lines(d, f, 1, f"{d}_{f}_1"), wave, ip.TEXELS,
                                  ip.TEXELS * ip.ARRAY_WORDS, raw=True, vgprs=32, sgprs=32)
                       for d in ip.ARRAY_DIMS for f in ip.ARRAY_FORMATS]
            text, _, _ = asm.module_text(kernels, "gfx1031", wave=wave)
            with tempfile.TemporaryDirectory() as d:
                s = Path(d) / "k.s"
                s.write_text(text, encoding="utf-8")
                r = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1031",
                                    *(["-mwavefrontsize64"] if wave == 64 else []), "-c", str(s), "-o",
                                    str(Path(d) / "k.o")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[:800])
            self.assertIn("image_gather4_lz v[20:21], v[16:18], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_CUBE d16",
                          text)

    def test_hardware_rule_matches_every_lane_and_tap(self):
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "r.json")
            rc, out = run_main(FakeArrayOracle(), "--dims", "2d_array,cube", "--formats", "unorm16,float32,float16",
                               "--waves", "32", "--json", j)
            with open(j, encoding="utf-8") as f:
                rep = json.load(f)["dims"]
        self.assertEqual(rc, 0, out)
        for dim in ("2d_array", "cube"):
            u = rep[dim]["unorm16"]["forms"]
            self.assertEqual(u["image_sample_lz_d16"]["match"]["rte"], 65536)
            self.assertEqual(u["image_sample_lz_d16"]["match"]["f32_rte"], 65534)     # 0xffaf and 0xffef
            self.assertEqual(u["image_gather4_lz_d16"]["match"]["rte"], 4 * 65536)
            f = rep[dim]["float32"]["forms"]["image_gather4_lz_d16"]
            self.assertEqual((f["match"]["rtz"], f["halfway"]["to_even"]), (f["inputs"], 0))
            self.assertEqual(rep[dim]["float32"]["forms"]["image_sample_lz_d16"]["0x3f7ff001"][0], "0x3bff")
            self.assertEqual(rep[dim]["float16"]["forms"]["image_load_d16"]["match"]["own"], rep[dim]["float16"]["inputs"])

    def test_the_old_translator_rule_is_told_apart(self):
        rc, out = run_main(FakeArrayOracle("translator"), "--dims", "2d_array", "--formats", "float32", "--waves", "32")
        self.assertEqual(rc, 0, out)
        self.assertRegex(out, r'image_sample_lz_d16: \{"inputs": 1024, "match": \{"rte": 1024, "rtz": 515\}')
        self.assertRegex(out, r'"halfway": \{"inputs": 52, "to_even": 28, "toward_zero": 0')

    def test_a_point_sample_off_the_load_is_a_harness_error(self):
        rc, out = run_main(FakeArrayOracle(point_off=True), "--dims", "cube", "--formats", "float16", "--waves", "32")
        self.assertEqual(rc, 2, out)
        self.assertIn("point sample differs from image_load", out)

    def test_dims_take_only_their_formats(self):
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "d16_image_probe.py"), "--dims", "cube", "--formats",
                            "unorm8"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)


class Cli(unittest.TestCase):
    def test_exits_77_without_the_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"), LEONIDA_HW_DRM_NODE="")
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "d16_image_probe.py"), "--formats", "unorm8"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)

    def test_bad_arguments_are_a_usage_error(self):
        for args in (["--formats", "unorm12"], ["--frac", "1.5"]):
            r = subprocess.run([sys.executable, "-I", str(TOOLS / "d16_image_probe.py"), *args],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, args)


class OnHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"hardware oracle unavailable: {e}") from None
        if asm.family(cls.oracle.arch) != "gfx10":
            cls.oracle.close()
            raise unittest.SkipTest(f"{cls.oracle.arch} is not GFX10")

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_unorm8_controls_hold(self):
        hw = ip.sweep(self.oracle, "unorm8", 32, 0.5)
        res = ip.classify("unorm8", hw, 0.5)
        self.assertEqual(res["control"], [])
        self.assertEqual(hw["load"][255], 0x3F800000)
        self.assertEqual(hw["load_d16"][255] & 0xFFFF, 0x3C00)

    def test_array_float32_controls_hold_and_truncate(self):
        hw = ip.array_sweep(self.oracle, "2d_array", "float32", 32)
        res = ip.array_classify("2d_array", "float32", hw)
        self.assertEqual(res["control"], [])
        self.assertEqual(res["forms"]["image_sample_lz_d16"]["0x3f7ff001"], ["0x3bff"])


if __name__ == "__main__":
    unittest.main()
