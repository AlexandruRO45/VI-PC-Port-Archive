"""Unit tests for tools/feature_queries.py.

The image is built here: a .text section of hand-assembled functions that call device slot 13 (`call [rax+68h]`)
with the feature id and data size set in the ways a compiler sets them (mov imm, xor, lea from a known register, a
register copy), with those values lost to a branch target, a call or another write, plus a wrapper called directly
and a slot-13 byte pattern hidden inside another instruction; .pdata function bounds in .rdata. No game files.

    python -I tests/test_feature_queries.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import feature_queries as fq  # noqa: E402
from pe_inspect import PeImage  # noqa: E402

IMAGE_BASE = 0x140000000
TEXT_RVA, TEXT_RAW, TEXT_SIZE = 0x1000, 0x400, 0x1000
RDATA_RVA, RDATA_RAW, RDATA_SIZE = 0x3000, 0x1400, 0x200

LOAD_DEVICE = "48 8B 01 4C 8D 44 24 20"      # mov rax,[rcx]; lea r8,[rsp+20h]
CALL_SLOT13 = "FF 50 68"                     # call qword ptr [rax+68h]

# name -> (hex before the slot call, or a full body when it holds its own calls)
FUNCTIONS = {
    # OPTIONS5 with the size from lea r9d,[rdx-0Fh] (27 - 15 = 12)
    "options5_lea": "BA 1B 00 00 00 44 8D 4A F1 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # D3D12_OPTIONS from xor edx,edx and mov r9d,3Ch
    "options_xor": "31 D2 41 B9 3C 00 00 00 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # the feature id copied from another register: mov ebx,7; mov edx,ebx; mov r9d,4 (SHADER_MODEL)
    "copy": "BB 07 00 00 00 89 DA 41 B9 04 00 00 00 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # a console-sized D3D12_OPTIONS: 0x40 bytes, not the PC 60
    "console_size": "31 D2 41 B9 40 00 00 00 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # edx overwritten from memory after the mov: feature not known, size known
    "overwritten": "BA 1B 00 00 00 8B 54 24 28 41 B9 0C 00 00 00 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # a call between the mov and the slot call forgets edx (rel32 to the function itself, never run)
    "after_call": "BA 1B 00 00 00 E8 00 00 00 00 41 B9 0C 00 00 00 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # a branch target between the mov and the slot call: jz +0 lands on the next instruction
    "label": "BA 1B 00 00 00 85 C0 74 00 41 B9 0C 00 00 00 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # the slot pattern FF 50 68 inside mov eax,6850FF00h: not a call
    "hidden": "B8 00 FF 50 68 C3",
    # a wrapper called directly with OPTIONS5 and its size: mov edx,1Bh; mov r9d,0Ch; call rel32 (to itself)
    "wrapper": "BA 1B 00 00 00 41 B9 0C 00 00 00 E8 00 00 00 00 C3",
    # 16-bit writes keep edx's upper half: mov edx,1001Bh; xor dx,dx leaves 10000h, not 0 (review of #7447)
    "xor_dx": "BA 1B 00 01 00 66 31 D2 41 B9 0C 00 00 00 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # mov dx,1Bh is a 2-byte immediate: reading 4 bytes would take the next instruction's bytes as the value
    "mov_dx": "BA 00 00 01 00 66 BA 1B 00 41 B9 0C 00 00 00 " + LOAD_DEVICE + " " + CALL_SLOT13 + " C3",
    # slot 20 through r10 with a disp32: call qword ptr [r10+0A0h]
    "slot20": "BA 01 00 00 00 41 B9 10 00 00 00 41 FF 92 A0 00 00 00 C3",
}


def build_image(with_pdata: bool = True) -> tuple[bytes, dict]:
    code, bounds = bytearray(), {}
    for name, text in FUNCTIONS.items():
        begin = TEXT_RVA + len(code)
        body = bytearray(bytes.fromhex(text))
        # a rel32 of 0 calls the next instruction; point it at the function's own start instead, as a real call
        # would point somewhere inside .text
        at = body.find(bytes.fromhex("E8 00 00 00 00"))
        if at >= 0:
            struct.pack_into("<i", body, at + 1, begin - (begin + at + 5))
        code += body
        bounds[name] = (begin, TEXT_RVA + len(code))
        code += b"\xcc" * (-len(code) % 16)
    assert len(code) <= TEXT_SIZE

    opt_size = 240
    buf = bytearray(RDATA_RAW + RDATA_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x40)
    buf[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, 0x44, 0x8664, 2, 0, 0, 0, opt_size, 0x0022)
    opt = 0x58
    struct.pack_into("<H", buf, opt, 0x20B)
    struct.pack_into("<I", buf, opt + 16, TEXT_RVA)
    struct.pack_into("<Q", buf, opt + 24, IMAGE_BASE)
    struct.pack_into("<II", buf, opt + 32, 0x1000, 0x200)
    struct.pack_into("<II", buf, opt + 56, 0x4000, TEXT_RAW)
    struct.pack_into("<H", buf, opt + 68, 2)
    struct.pack_into("<I", buf, opt + 108, 16)
    buf[TEXT_RAW:TEXT_RAW + len(code)] = code
    if with_pdata:
        for i, (b, e) in enumerate(bounds.values()):
            struct.pack_into("<III", buf, RDATA_RAW + i * 12, b, e, 0x3100)
        struct.pack_into("<II", buf, opt + 112 + 3 * 8, RDATA_RVA, len(bounds) * 12)
    sections = opt + opt_size
    for i, (name, rva, size, raw, chars) in enumerate(((b".text", TEXT_RVA, TEXT_SIZE, TEXT_RAW, 0x60000020),
                                                       (b".rdata", RDATA_RVA, RDATA_SIZE, RDATA_RAW, 0x40000040))):
        at = sections + i * 40
        buf[at:at + 8] = name.ljust(8, b"\0")
        struct.pack_into("<IIII", buf, at + 8, size, rva, size, raw)
        struct.pack_into("<I", buf, at + 36, chars)
    return bytes(buf), bounds


def calls_in(found: dict, bounds: dict, name: str, key: str = "calls") -> list:
    b, e = bounds[name]
    return [(c["feature"], c["size"]) for c in found[key] if b <= c["rva"] < e]


class ScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data, cls.bounds = build_image()
        cls.img = PeImage(data)
        cls.found = fq.scan(cls.img)

    def test_lea_from_a_known_register_gives_the_size(self):
        self.assertEqual(calls_in(self.found, self.bounds, "options5_lea"), [(27, 12)])

    def test_xor_and_mov_imm(self):
        self.assertEqual(calls_in(self.found, self.bounds, "options_xor"), [(0, 60)])

    def test_a_register_copy_carries_the_constant(self):
        self.assertEqual(calls_in(self.found, self.bounds, "copy"), [(7, 4)])

    def test_another_write_forgets_the_feature(self):
        self.assertEqual(calls_in(self.found, self.bounds, "overwritten"), [(None, 12)])

    def test_a_call_in_between_forgets_the_feature(self):
        self.assertEqual(calls_in(self.found, self.bounds, "after_call"), [(None, 12)])

    def test_a_branch_target_forgets_unless_across_labels(self):
        self.assertEqual(calls_in(self.found, self.bounds, "label"), [(None, 12)])
        across = fq.scan(self.img, across_labels=True)
        self.assertEqual(calls_in(across, self.bounds, "label"), [(27, 12)])

    def test_a_16_bit_write_forgets_instead_of_guessing(self):
        self.assertEqual(calls_in(self.found, self.bounds, "xor_dx"), [(None, 12)])
        self.assertEqual(calls_in(self.found, self.bounds, "mov_dx"), [(None, 12)])

    def test_a_pattern_inside_another_instruction_is_not_a_call(self):
        self.assertEqual(calls_in(self.found, self.bounds, "hidden"), [])

    def test_slot_calls_are_counted_once_each(self):
        self.assertEqual(len(self.found["calls"]), 9)
        self.assertEqual(self.found["undecoded_functions"], 0)

    def test_another_slot_with_a_disp32(self):
        found = fq.scan(self.img, slot=20)
        self.assertEqual(calls_in(found, self.bounds, "slot20"), [(1, 16)])
        self.assertEqual(len(found["calls"]), 1)

    def test_any_call_finds_the_wrapper_and_skips_the_slot_calls(self):
        found = fq.scan(self.img, any_call=True)
        self.assertEqual(calls_in(found, self.bounds, "wrapper", "other_calls"), [(27, 12)])
        self.assertEqual(calls_in(found, self.bounds, "options5_lea", "other_calls"), [])
        self.assertEqual(len(found["calls"]), 9)

    def test_without_any_call_the_wrapper_is_not_read(self):
        self.assertEqual(self.found["other_calls"], [])

    def test_no_pdata(self):
        data, _ = build_image(with_pdata=False)
        self.assertFalse(fq.scan(PeImage(data))["pdata"])


class SummaryTest(unittest.TestCase):
    def setUp(self):
        data, _ = build_image()
        found = fq.scan(PeImage(data), any_call=True)
        found["any_call"] = True
        self.r = fq.summarize("synthetic.exe", found, None)

    def test_feature_and_size_with_the_pc_size(self):
        rows = {(x["feature"], x["size"]): x for x in self.r["feature_and_size"]}
        self.assertEqual(rows[(27, 12)]["name"], "D3D12_OPTIONS5 (27)")
        self.assertEqual(rows[(27, 12)]["note"], "the PC size")
        self.assertEqual(rows[(0, 64)]["note"], "PC size 60")
        self.assertEqual(rows[(0, 60)]["note"], "the PC size")
        # only options5_lea still holds 27 in edx at the call; the others lost it on purpose
        self.assertEqual(self.r["options5_calls"], 1)

    def test_one_sided_rows(self):
        self.assertEqual(self.r["feature_only"], [])
        self.assertEqual(self.r["size_only"], [{"size": 12, "calls": 5}])
        self.assertEqual(self.r["neither"], 0)

    def test_other_calls_are_numbers_with_a_pc_size_mark(self):
        # the wrapper, and slot20's call through [r10+0A0h], which is not slot 13 and so another indirect call
        self.assertEqual(self.r["other_calls"],
                         [{"kind": "indirect", "edx": 1, "r9d": 16, "pc_size_of_feature": True, "calls": 1},
                          {"kind": "direct", "edx": 27, "r9d": 12, "pc_size_of_feature": True, "calls": 1}])

    def test_feature_filter(self):
        data, _ = build_image()
        found = fq.scan(PeImage(data), any_call=True)
        found["any_call"] = True
        self.assertEqual(fq.summarize("x", found, 5)["other_calls"], [])


class CliTest(unittest.TestCase):
    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fq.main(list(args))
        return rc, out.getvalue(), err.getvalue()

    def write(self, d, name, data):
        path = os.path.join(d, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_text_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, "synthetic.exe", build_image()[0])
            rc, out, _ = self.run_main(path)
            self.assertEqual(rc, 0)
            self.assertIn("synthetic.exe: 9 call(s) through slot 13 (+0x68)", out)
            self.assertIn("D3D12_OPTIONS5 (27), 12 bytes (the PC size): 1", out)
            self.assertIn("D3D12_OPTIONS5 (27) in edx: 1", out)
            self.assertNotIn("0x1", out.split("sites")[0].split("+0x68")[1])   # no RVA without --sites
            rc, out, _ = self.run_main(path, "--json")
            self.assertEqual(rc, 0)
            r = json.loads(out)[0]
            self.assertEqual(r["slot_calls"], 9)
            self.assertNotIn("sites", r)

    def test_sites_and_any_call(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, "synthetic.exe", build_image()[0])
            rc, out, _ = self.run_main(path, "--any-call", "--feature", "27", "--sites")
            self.assertEqual(rc, 0)
            self.assertIn("with r9d the PC data size of the feature id in edx: 1", out)
            self.assertIn("direct: edx 27, r9d 12 (the PC size of D3D12_OPTIONS5 (27)): 1", out)
            self.assertIn("sites (RVAs stay on your machine):", out)

    def test_no_pdata_exits_2_and_a_non_pe_exits_1(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = self.run_main(self.write(d, "bare.exe", build_image(with_pdata=False)[0]))
            self.assertEqual(rc, 2)
            self.assertIn("no .pdata function table", err)
            rc, _, err = self.run_main(self.write(d, "text.exe", b"not a PE file"))
            self.assertEqual(rc, 1)
            self.assertIn("not a readable PE image", err)

    def test_negative_slot_is_refused(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            fq.main(["x.exe", "--slot", "-1"])
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
