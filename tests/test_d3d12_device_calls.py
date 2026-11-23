"""Unit tests for tools/d3d12_device_calls.py on an x86-64 image assembled here byte by byte.

The image imports D3D12XboxCreateDevice from d3d12_xs.dll and has eight .pdata functions: two creator calls (through
the import thunk with the out-pointer a global, and `call [rip+IAT]` with it a stack slot), calls through the device
global and a copy of it (`call [vtable+disp]`, `call reg`, a CFG-style dispatch with the method in rax and the
device in rdx, a tail jump), an unrelated object's vtable call, the device passed to another function, and an
indexed vtable call no slot can be read from. No game files.

    python -I tests/test_d3d12_device_calls.py
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

import d3d12_device_calls as ddc  # noqa: E402
import pe_inspect  # noqa: E402

IMAGE_BASE = 0x140000000
TEXT_RVA = 0x1000
IDATA_RVA = 0x2000
DATA_RVA = 0x3000
PDATA_RVA = 0x4000
SIZE = 0x5000

IAT = IDATA_RVA + 0x60          # D3D12XboxCreateDevice's address-table slot
DEVICE = DATA_RVA + 0x00        # the creator's out-pointer
COPY = DATA_RVA + 0x08          # a global the device is copied to
OTHER = DATA_RVA + 0x10         # an unrelated object
GUARD = DATA_RVA + 0x18         # the dispatch routine's pointer


class Asm:
    """Appends instructions at known RVAs; `rip` and `rel` encode a RIP-relative or rel32 operand."""

    def __init__(self, rva):
        self.rva = rva
        self.code = bytearray()

    @property
    def here(self):
        return self.rva + len(self.code)

    def raw(self, b):
        self.code += b

    def rip(self, prefix, target):
        end = self.here + len(prefix) + 4
        self.code += prefix + struct.pack("<i", target - end)

    def rel(self, opcode, target):
        self.rip(opcode, target)


def build_image():
    """(image bytes, {function name: rva})."""
    a = Asm(TEXT_RVA)
    funcs = {}

    def begin(name):
        while len(a.code) % 16:
            a.raw(b"\xcc")
        funcs[name] = [a.here, None]

    def end(name):
        funcs[name][1] = a.here

    thunk = a.here
    a.rip(b"\xff\x25", IAT)                              # jmp [rip+IAT]

    begin("create")
    a.rip(b"\x4c\x8d\x0d", DEVICE)                       # lea r9,[rip+DEVICE]
    a.rel(b"\xe8", thunk)                                # call thunk
    a.raw(b"\xc3")
    end("create")

    begin("copy")                                        # slot 14 through call [rax+disp8]
    a.rip(b"\x48\x8b\x0d", DEVICE)                       # mov rcx,[rip+DEVICE]
    a.rip(b"\x48\x89\x0d", COPY)                         # mov [rip+COPY],rcx
    a.raw(b"\x48\x8b\x01")                               # mov rax,[rcx]
    a.raw(b"\x31\xd2")                                   # xor edx,edx
    a.raw(b"\x41\xb8\x01\x00\x00\x00")                   # mov r8d,1
    a.raw(b"\xff\x50\x70")                               # call [rax+0x70]
    a.raw(b"\xc3")
    end("copy")

    begin("method")                                      # slot 48 through the copy, call rax
    a.rip(b"\x48\x8b\x1d", COPY)                         # mov rbx,[rip+COPY]
    a.raw(b"\x48\x8b\x03")                               # mov rax,[rbx]
    a.raw(b"\x48\x8b\x80\x80\x01\x00\x00")               # mov rax,[rax+0x180]
    a.raw(b"\x48\x89\xd9")                               # mov rcx,rbx
    a.raw(b"\xff\xd0")                                   # call rax
    a.raw(b"\xc3")
    end("method")

    begin("dispatch")                                    # slot 26, device in rdx, dispatch routine
    a.rip(b"\x48\x8b\x15", DEVICE)                       # mov rdx,[rip+DEVICE]
    a.raw(b"\x48\x8d\x4c\x24\x20")                       # lea rcx,[rsp+0x20]
    a.raw(b"\x48\x8b\x02")                               # mov rax,[rdx]
    a.raw(b"\x48\x8b\x80\xd0\x00\x00\x00")               # mov rax,[rax+0xd0]
    a.rip(b"\xff\x15", GUARD)                            # call [rip+GUARD]
    a.raw(b"\xc3")
    end("dispatch")

    begin("stack")                                       # creator into a stack slot, then a tail jump at slot 119
    a.raw(b"\x48\x83\xec\x48")                           # sub rsp,0x48
    a.raw(b"\x4c\x8d\x4c\x24\x30")                       # lea r9,[rsp+0x30]
    a.rip(b"\xff\x15", IAT)                              # call [rip+IAT]
    a.raw(b"\x48\x8b\x4c\x24\x30")                       # mov rcx,[rsp+0x30]
    a.raw(b"\x48\x83\xc4\x48")                           # add rsp,0x48
    a.raw(b"\x48\x8b\x01")                               # mov rax,[rcx]
    a.raw(b"\xff\xa0\xb8\x03\x00\x00")                   # jmp [rax+0x3b8]
    end("stack")

    begin("unrelated")                                   # another object's vtable call: not counted
    a.rip(b"\x48\x8b\x0d", OTHER)                        # mov rcx,[rip+OTHER]
    a.raw(b"\x48\x8b\x01")                               # mov rax,[rcx]
    a.raw(b"\xff\x50\x18")                               # call [rax+0x18]
    a.raw(b"\xc3")
    end("unrelated")

    begin("passed")                                      # the device handed to another function
    a.rip(b"\x48\x8b\x0d", DEVICE)                       # mov rcx,[rip+DEVICE]
    a.rel(b"\xe8", funcs["unrelated"][0])                # call unrelated
    a.raw(b"\xc3")
    end("passed")

    begin("indexed")                                     # call [rax+rdx*8]: no slot
    a.rip(b"\x48\x8b\x0d", DEVICE)                       # mov rcx,[rip+DEVICE]
    a.raw(b"\x48\x8b\x01")                               # mov rax,[rcx]
    a.raw(b"\xff\x14\xd0")                               # call [rax+rdx*8]
    a.raw(b"\xc3")
    end("indexed")

    assert len(a.code) < 0x1000
    buf = bytearray(SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x40)
    buf[0x40:0x44] = b"PE\0\0"
    sections = ((b".text", TEXT_RVA, 0x60000020), (b".idata", IDATA_RVA, 0xC0000040),
                (b".data", DATA_RVA, 0xC0000040), (b".pdata", PDATA_RVA, 0x40000040))
    struct.pack_into("<HHIIIHH", buf, 0x44, 0x8664, len(sections), 0, 0, 0, 0xF0, 0x22)
    opt = 0x58
    struct.pack_into("<H", buf, opt, 0x20B)
    struct.pack_into("<Q", buf, opt + 24, IMAGE_BASE)
    struct.pack_into("<II", buf, opt + 32, 0x1000, 0x1000)       # section and file alignment
    struct.pack_into("<II", buf, opt + 56, SIZE, 0x1000)         # SizeOfImage, SizeOfHeaders
    struct.pack_into("<I", buf, opt + 108, 16)
    struct.pack_into("<II", buf, opt + 112 + 8, IDATA_RVA, 40)   # import directory
    struct.pack_into("<II", buf, opt + 112 + 3 * 8, PDATA_RVA, 12 * len(funcs))
    for i, (name, rva, chars) in enumerate(sections):
        at = opt + 0xF0 + i * 40
        buf[at:at + len(name)] = name
        struct.pack_into("<IIII", buf, at + 8, 0x1000, rva, 0x1000, rva)
        struct.pack_into("<I", buf, at + 36, chars)

    lookup = IDATA_RVA + 0x40
    hint = IDATA_RVA + 0xA0
    struct.pack_into("<IIIII", buf, IDATA_RVA, lookup, 0, 0, IDATA_RVA + 0xD0, IAT)
    struct.pack_into("<Q", buf, lookup, hint)
    struct.pack_into("<Q", buf, IAT, hint)
    buf[hint + 2:hint + 2 + 22] = b"D3D12XboxCreateDevice\0"
    buf[IDATA_RVA + 0xD0:IDATA_RVA + 0xD0 + 13] = b"d3d12_xs.dll\0"

    buf[TEXT_RVA:TEXT_RVA + len(a.code)] = a.code
    for i, (begin_rva, end_rva) in enumerate(sorted(funcs.values())):
        struct.pack_into("<III", buf, PDATA_RVA + 12 * i, begin_rva, end_rva, 0)
    return bytes(buf), {k: v[0] for k, v in funcs.items()}


class DeviceCallsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.funcs = build_image()
        cls.r = ddc.scan(cls.data)

    def slots(self):
        return {s["slot"]: s for s in self.r["slots"]}

    def test_known_sites_are_counted_exactly(self):
        slots = self.slots()
        self.assertEqual(sorted(slots), [14, 26, 48, 119])
        self.assertEqual(self.r["sites"], 4)
        for slot in slots.values():
            self.assertEqual(slot["sites"], 1)
        self.assertEqual((slots[14]["calls"], slots[14]["this_rcx"]), (1, 1))
        self.assertEqual((slots[48]["calls"], slots[48]["this_rcx"]), (1, 1))
        self.assertEqual((slots[26]["calls"], slots[26]["this_rdx"]), (1, 1))
        self.assertEqual((slots[119]["tail_jumps"], slots[119]["calls"], slots[119]["this_rcx"]), (1, 0, 1))

    def test_the_unrelated_vtable_call_is_not_counted(self):
        self.assertNotIn(3, self.slots())
        rvas = {int(s["rva"], 16) for s in self.r["site_rvas"]}
        start, stop = self.funcs["unrelated"], self.funcs["passed"]
        self.assertFalse(any(start <= rva < stop for rva in rvas))

    def test_creators_and_globals(self):
        self.assertEqual(self.r["creators"], [{"import": "d3d12_xs.dll!D3D12XboxCreateDevice", "call_sites": 2,
                                               "out": {"global": 1, "memory slot": 1}}])
        self.assertEqual(self.r["device_globals"], 2)

    def test_unresolved_sites_are_reported_not_guessed(self):
        self.assertEqual(self.r["unresolved"], {
            "vtable_loads_without_matched_call": 1,      # the indexed call
            "device_passed_to_other_calls": 1,
            "device_stored_to_non_stack_memory": 0,
            "creator_calls_out_not_followed": 0,
        })

    def test_names_and_argument_estimate(self):
        slots = self.slots()
        self.assertEqual(slots[14]["name"], "CreateDescriptorHeap")
        self.assertEqual(slots[14]["name_source"], "PC order, console measured")
        self.assertEqual(slots[48]["name"], "CreateCommandQueueX")
        self.assertIsNone(slots[119]["name"])
        self.assertEqual(slots[14]["args_max"], 3)          # this, rdx, r8
        self.assertEqual(ddc.argument_count({2, 8}, {4, 5, 9}), 6)

    def test_no_import_means_nothing_followed(self):
        data = bytearray(self.data)
        data[IDATA_RVA + 0xA2:IDATA_RVA + 0xA2 + 21] = b"SomethingElseEntirely"
        r = ddc.scan(bytes(data))
        self.assertEqual((r["creators"], r["slots"], r["sites"]), ([], [], 0))

    def test_a_32_bit_image_is_refused(self):
        data = bytearray(self.data)
        struct.pack_into("<H", data, 0x58, 0x10B)
        with self.assertRaises(pe_inspect.PeFormatError):
            ddc.scan(bytes(data))

    def test_cli_needs_out_and_writes_stable_reports_without_addresses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "image.exe")
            with open(path, "wb") as f:
                f.write(self.data)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                ddc.main([path])
            self.assertEqual(e.exception.code, 2)
            outputs = []
            for run in ("a", "b"):
                out = os.path.join(tmp, run)
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    self.assertEqual(ddc.main([path, "--out", out]), 0)
                with open(os.path.join(out, "image.exe.device_calls.txt"), encoding="utf-8") as f:
                    text = f.read()
                with open(os.path.join(out, "image.exe.device_calls.json"), encoding="utf-8") as f:
                    raw = f.read()
                self.assertEqual(printed.getvalue(), text)
                outputs.append((text, raw))
            self.assertEqual(outputs[0], outputs[1])
            text, raw = outputs[0]
            self.assertNotIn("0x", text)
            self.assertIn("4 site(s) across 4 slot(s)", text)
            self.assertEqual(json.loads(raw)["sites"], 4)

    def test_cli_refuses_a_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ddc.main([os.path.join(tmp, "none.exe"), "--out", tmp]), 1)


if __name__ == "__main__":
    unittest.main()
