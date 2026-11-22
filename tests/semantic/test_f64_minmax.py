"""v_min_f64 / v_max_f64 (runtime/test/shaders/cs_w32_f64_minmax.asm, docs/design.md "v_min_f64, v_max_f64").

A bit-exact selection of one source, checked against 28 words measured natively on RDNA2 gfx1036 (tools/hw_oracle
raw wave32 kernels, MODE.IEEE off): NaN and signalling-NaN operands, two NaNs, -0/+0, subnormals, infinities,
abs/neg, an inline constant, clamp, and an ignored omod. Run through tests/spirv_interp.py. Before, both raised
Unsupported.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import spirv_interp  # noqa: E402
import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
V = {"one": 0x3FF0000000000000, "two": 0x4000000000000000, "three": 0x4008000000000000,
     "qa": 0x7FF8000000000001, "qb": 0xFFF8000000000002, "nz": 0x8000000000000000, "pz": 0,
     "sn": 0x7FF0000000000001, "sub": 1, "ninf": 0xFFF0000000000000, "five": 0x4014000000000000,
     "nsub": 0x8000000000000001}
# (op, src0, src1, extra, gfx1036 word), MODE.IEEE off
HW = [
    ('v_min_f64', 'one', 'two', '', 0x3ff0000000000000),
    ('v_min_f64', 'qa', 'three', '', 0x4008000000000000),
    ('v_min_f64', 'three', 'qa', '', 0x4008000000000000),
    ('v_min_f64', 'qa', 'qb', '', 0x7ff8000000000001),
    ('v_min_f64', 'nz', 'pz', '', 0x8000000000000000),
    ('v_min_f64', 'pz', 'nz', '', 0x8000000000000000),
    ('v_min_f64', 'sn', 'one', '', 0x3ff0000000000000),
    ('v_min_f64', 'sub', 'pz', '', 0x0000000000000000),
    ('v_min_f64', 'ninf', 'five', '', 0xfff0000000000000),
    ('v_min_f64', 'nsub', 'pz', '', 0x8000000000000001),
    ('v_min_f64', 'one', 'sn', '', 0x3ff0000000000000),
    ('v_min_f64', 'sn', 'qa', '', 0x7ff0000000000001),
    ('v_max_f64', 'one', 'two', '', 0x4000000000000000),
    ('v_max_f64', 'qa', 'three', '', 0x4008000000000000),
    ('v_max_f64', 'three', 'qa', '', 0x4008000000000000),
    ('v_max_f64', 'qa', 'qb', '', 0x7ff8000000000001),
    ('v_max_f64', 'nz', 'pz', '', 0x0000000000000000),
    ('v_max_f64', 'pz', 'nz', '', 0x0000000000000000),
    ('v_max_f64', 'sn', 'one', '', 0x3ff0000000000000),
    ('v_max_f64', 'sub', 'pz', '', 0x0000000000000001),
    ('v_max_f64', 'ninf', 'five', '', 0x4014000000000000),
    ('v_max_f64', 'nsub', 'pz', '', 0x0000000000000000),
    ('v_max_f64', 'one', 'sn', '', 0x3ff0000000000000),
    ('v_max_f64', 'sn', 'qa', '', 0x7ff0000000000001),
]
# (instruction, src0, src1, gfx1036 word)
HW_FORMS = [
    ("v_max_f64 v[10:11], v[2:3], v[4:5] mul:2", "one", "two", 0x4000000000000000),     # omod ignored
    ("v_min_f64 v[10:11], v[2:3], v[4:5] clamp", "five", "three", 0x3FF0000000000000),  # 3.0 clamped to 1.0
    ("v_max_f64 v[10:11], -v[2:3], |v[4:5]|", "one", "ninf", 0x7FF0000000000000),       # max(-1, +inf)
    ("v_min_f64 v[10:11], v[2:3], 0.5", "one", "one", 0x3FE0000000000000),              # inline 0.5 as binary64
]


def run(ins, x, y):
    lines = []
    for reg, v in ((2, V[x]), (4, V[y])):
        lines += [f"v_mov_b32 v{reg}, {v & M32:#x}", f"v_mov_b32 v{reg + 1}, {v >> 32:#x}"]
    lines += [ins, "exp mrt0 v10, v11, v10, v11 done vm", "s_endpgm"]
    r = spirv_interp.run(support.translate_lines("ps_f64_minmax", lines))
    return (r.vgpr[10] & M32) | (r.vgpr[11] & M32) << 32


def ref(op, x, y):
    """Plain-Python reference of the measured rule (IEEE off): NaN gives the other operand, two NaNs src0,
    -0 < +0, otherwise the numeric order."""
    import struct

    def nan(b):
        return (b >> 52) & 0x7FF == 0x7FF and b & ((1 << 52) - 1)

    def val(b):
        return struct.unpack("<d", struct.pack("<Q", b))[0]
    a, b = V[x], V[y]
    if nan(a):
        return b if not nan(b) else a
    if nan(b):
        return a
    ka = (val(a), 0 if a >> 63 else 1)
    kb = (val(b), 0 if b >> 63 else 1)
    if op == "v_min_f64":
        return b if kb < ka else a
    return b if kb > ka else a


class MinMaxF64(unittest.TestCase):
    def test_hardware_words(self):
        for op, x, y, _, want in HW:
            with self.subTest(op=op, x=x, y=y):
                self.assertEqual(f"{run(f'{op} v[10:11], v[2:3], v[4:5]', x, y):#018x}", f"{want:#018x}")
        for ins, x, y, want in HW_FORMS:
            with self.subTest(ins=ins):
                self.assertEqual(f"{run(ins, x, y):#018x}", f"{want:#018x}")

    def test_reference_matches_hardware(self):
        for op, x, y, _, want in HW:
            with self.subTest(op=op, x=x, y=y):
                self.assertEqual(ref(op, x, y), want)

    def test_fixture_words(self):
        sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
        from compute_fixture import compute_header
        header = compute_header(os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_w32_f64_minmax.asm"))
        spv = support.translate_fixture("cs_w32_f64_minmax", fssm=bytes(header))
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        self.assertEqual([memory[4 * i] for i in range(8)], [0, 0x40080000, 0, 0x80000000, 1, 0, 1, 0x7FF00000])


if __name__ == "__main__":
    unittest.main()
