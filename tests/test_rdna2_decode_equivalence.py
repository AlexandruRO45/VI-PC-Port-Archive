"""Equivalence tests for the speed-ups in tools/rdna2_decode.py.

The decoder keeps a few derived facts in bounded caches and takes shortcuts for the common operand shapes. Nothing
here may change what it prints or returns, so these tests pin it from three sides:

  - every cache or shortcut is compared with the plain formulation it replaced, over its whole input range;
  - a hand-built corpus (every opcode of every table, several field patterns each, both wave sizes, both targets,
    every truncation of a multi-word instruction, plus raw junk words) is decoded and listed, and the SHA-256 of
    the result is compared with the one the decoder gave before the speed-ups. If decoding changes on purpose,
    regenerate the digests with `python -I tests/test_rdna2_decode_equivalence.py --print-digests`;
  - the caches stay bounded and decoding never changes the interpreter's garbage-collector setting.

No LLVM, GPU or game files are needed, and the corpus is built from the opcode tables alone.
"""
from __future__ import annotations

import gc
import hashlib
import os
import struct
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")

sys.dont_write_bytecode = True
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import rdna2_decode as D  # noqa: E402
import rdna2_isa_tables as T  # noqa: E402

# SHA-256 of `corpus_digest(<target>)`, measured with the decoder before the speed-ups (origin/main 896b56c6a).
EXPECTED_DIGESTS = {
    "gfx1013/wave32": "24c49dfb073c4051466958c98038436df895df8bfb13f7805624c76ea36ba182",
    "gfx1013/wave64": "7b72977ef3d6d419c0b29add9670d45fd1ac6d6d999dbf1f8fcec7c4276de83a",
    "gfx1030/wave32": "867115c0a47f8d36e85057886c76f6a06dae172e1925dcbcf2ecbc9ed99bffc8",
    "gfx1030/wave64": "14e42722f80e409e71c40d5b48a8d75ff363a48d986fe7f2b786edf2e7c2dd50",
}


class _Rng:
    """xorshift32, so the corpus is the same on every Python version."""

    def __init__(self, seed: int) -> None:
        self.s = seed or 1

    def u32(self) -> int:
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x
        return x

    def below(self, n: int) -> int:
        return self.u32() % n

    def pick(self, seq):
        return seq[self.below(len(seq))]


def _src9(r: _Rng) -> int:
    """A 9-bit source code: registers, inline constants, specials, the literal marker and VGPRs."""
    return r.pick((0, 1, 7, 42, 100, 105, 106, 107, 108, 115, 123, 124, 125, 126, 127, 128, 129, 192, 193, 208, 209,
                   240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 300,
                   511, r.below(512)))


def _build(r: _Rng) -> list[tuple[str, int, list[int]]]:
    """(family, opcode, words) for every table entry and several field patterns."""
    out = []

    def add(fam, opc, words):
        out.append((fam, opc, [w & 0xFFFFFFFF for w in words]))

    for rep in range(6):
        for opc in sorted(T.SOP1):
            s0 = r.pick((r.below(128), 128 + r.below(127), 255, 0, 105, 106))
            add("SOP1", opc, [0xBE800000 | (r.below(128) << 16) | (opc << 8) | s0, r.u32()])
        for opc in sorted(T.SOP2):
            s0, s1 = r.pick((r.below(128), 129, 255, 105, 106)), r.pick((r.below(256), 255, 3))
            add("SOP2", opc, [0x80000000 | (opc << 23) | (r.below(128) << 16) | (s1 << 8) | s0, r.u32()])
        for opc in sorted(T.SOPC):
            s0, s1 = r.pick((r.below(128), 255, 129)), r.below(256)
            add("SOPC", opc, [0xBF000000 | (opc << 16) | (s1 << 8) | s0, r.u32()])
        for opc in sorted(T.SOPK):
            add("SOPK", opc, [0xB0000000 | (opc << 23) | (r.below(128) << 16) | r.below(0x10000), r.u32()])
        for opc in sorted(T.SOPP):
            add("SOPP", opc, [0xBF800000 | (opc << 16) | r.pick((0, 1, 0x7FFF, 0x8000, 0xFFFF, r.below(0x10000))),
                              r.u32()])
        for opc in sorted(T.VOP1):
            add("VOP1", opc, [0x7E000000 | (r.below(256) << 17) | (opc << 9) | _src9(r), r.u32()])
        for opc in sorted(T.VOP2):
            add("VOP2", opc, [(opc << 25) | (r.below(256) << 17) | (r.below(256) << 9) | _src9(r), r.u32()])
        for opc in sorted(T.VOPC):
            add("VOPC", opc, [0x7C000000 | (opc << 17) | (r.below(256) << 9) | _src9(r), r.u32()])
        for tab in (T.VOP3, T.VOP3B):
            for opc in sorted(tab):
                add("VOP3", opc, [0xD4000000 | (opc << 16) | r.below(0x10000),
                                  _src9(r) | (_src9(r) << 9) | (_src9(r) << 18) | (r.below(32) << 27), r.u32()])
        for opc in sorted(T.VOP3P):
            add("VOP3P", opc, [0xCC000000 | (opc << 16) | r.below(0x10000),
                               _src9(r) | (_src9(r) << 9) | (_src9(r) << 18) | (r.below(32) << 27), r.u32()])
        for opc in sorted(T.VINTRP):
            add("VINTRP", opc, [0xC8000000 | (opc << 16) | r.below(0x10000)])
        for opc in sorted(T.SMEM):
            add("SMEM", opc, [0xF4000000 | (opc << 18) | r.below(0x40000), r.u32()])
        for opc in sorted(T.MUBUF):
            add("MUBUF", opc, [0xE0000000 | (opc << 18) | r.below(0x40000), r.u32()])
        for opc in sorted(T.MTBUF):
            add("MTBUF", opc, [0xE8000000 | ((opc & 7) << 16) | r.below(0x10000) | (r.below(128) << 19),
                               r.u32() & ~(1 << 21) | ((opc >> 3) << 21)])
        for opc in sorted(T.MIMG):
            nsa = r.pick((0, 0, 1, 2, 3))
            add("MIMG", opc, [0xF0000000 | ((opc & 0x7F) << 18) | (opc >> 7) | (nsa << 1) | (r.below(8) << 3)
                              | (r.below(2) << 7) | (r.below(16) << 8) | (r.below(2) << 12) | (r.below(2) << 13)
                              | (r.below(2) << 15) | (r.below(2) << 16) | (r.below(2) << 17) | (r.below(2) << 25),
                              r.u32()] + [r.u32() for _ in range(nsa)])
        for opc in sorted(T.DS):
            add("DS", opc, [0xD8000000 | (opc << 18) | r.below(0x40000), r.u32()])
        for seg, tab in ((0, T.FLAT), (1, T.SCRATCH), (2, T.GLOBAL)):
            for opc in sorted(tab):
                add("FLAT", opc, [0xDC000000 | (opc << 18) | (seg << 14) | r.below(0x4000),
                                  r.u32() & ~(0x7F << 16) | r.pick((0x7D, 0x7F, r.below(128))) << 16])
        for _ in range(40):
            add("EXP", 0, [0xF8000000 | r.below(0x2000), r.u32()])
        add("FLAT", 3, [0xDC000000 | (3 << 14), r.u32()])  # a segment value no table has
    return out


def corpus_digest(mcpu: str, wave32: bool) -> str:
    """SHA-256 over decode_bytes + disasm_bytes of the whole corpus, one instruction (or truncation) at a time."""
    h = hashlib.sha256()
    for fam, opc, words in _build(_Rng(0x9E3779B9)):
        data = b"".join(struct.pack("<I", w) for w in words)
        cuts = {len(data)}
        if len(words) > 1:
            cuts.update(range(4, len(data), 4))  # every truncation: the decoder must refuse them the same way
            cuts.add(len(data) - 1)  # and a stream that ends inside a word
        for n in sorted(cuts):
            blob = data[:n]
            for code_start in (0, 0x40):
                dec = D.decode_bytes(blob, mcpu=mcpu, wave32=wave32, code_start=code_start, name=fam)
                txt = D.disasm_bytes(blob, mcpu=mcpu, wave32=wave32, code_start=code_start, name=fam)
                h.update(repr((fam, opc, n, code_start, dec, txt)).encode())
    # Raw junk words: mostly undecodable, and what is decodable still has to agree.
    r = _Rng(12345)
    junk = b"".join(struct.pack("<I", r.u32()) for _ in range(6000))
    h.update(repr((D.decode_bytes(junk, mcpu=mcpu, wave32=wave32), D.disasm_bytes(junk, mcpu=mcpu, wave32=wave32))).encode())
    return h.hexdigest()


class DecoderEquivalenceTests(unittest.TestCase):
    def test_corpus_digests_match_the_decoder_before_the_speedups(self):
        for target, want in EXPECTED_DIGESTS.items():
            mcpu, wave = target.split("/")
            with self.subTest(target=target):
                self.assertEqual(corpus_digest(mcpu, wave == "wave32"), want)

    def test_corpus_covers_every_table_entry(self):
        seen = {(fam, opc) for fam, opc, _ in _build(_Rng(0x9E3779B9))}
        for fam, tab in (("SOP1", T.SOP1), ("SOP2", T.SOP2), ("SOPC", T.SOPC), ("SOPK", T.SOPK), ("SOPP", T.SOPP),
                         ("VOP1", T.VOP1), ("VOP2", T.VOP2), ("VOPC", T.VOPC), ("VOP3", T.VOP3), ("VOP3", T.VOP3B),
                         ("VOP3P", T.VOP3P), ("VINTRP", T.VINTRP), ("SMEM", T.SMEM), ("MUBUF", T.MUBUF),
                         ("MTBUF", T.MTBUF), ("MIMG", T.MIMG), ("DS", T.DS)):
            for opc in tab:
                self.assertIn((fam, opc), seen)

    # ---- each shortcut against the formulation it replaced -------------------------------------------------------

    def test_spec_info_matches_a_scan_of_every_spec(self):
        tables = (T.SOP1, T.SOP2, T.SOPC, T.SOPK, T.SOPP, T.VOP1, T.VOP2, T.VOPC, T.VOP3, T.VOP3B, T.VOP3P,
                  T.VINTRP, T.SMEM, T.MUBUF, T.MTBUF, T.MIMG, T.DS, T.FLAT, T.GLOBAL, T.SCRATCH)
        for tab in tables:
            for _name, spec in tab.values():
                info = D._spec_info(spec)
                self.assertIs(info[0], spec)
                self.assertEqual(info[1], sum(1 for fn, _, _ in spec if fn in ("SRC0", "SRC1", "SRC2")))
                self.assertEqual(info[2], any(fn in ("OFFSET", "SOFFSET") for fn, _, _ in spec))

    def test_spec_info_is_bounded_and_never_serves_another_spec(self):
        D._SPEC_INFO.clear()
        for i in range(D._SPEC_INFO_MAX + 50):
            spec = (("VDST", 1, "i"), ("SRC0", i % 3, "i"))
            self.assertEqual(D._spec_info(spec)[1], 1)
            self.assertLessEqual(len(D._SPEC_INFO), D._SPEC_INFO_MAX)
        spec_a = (("SRC0", 1, "i"), ("SRC1", 1, "i"))
        spec_b = (("OFFSET", 1, "i"),)
        self.assertEqual(D._spec_info(spec_a)[1:], (2, False))
        self.assertEqual(D._spec_info(spec_b)[1:], (0, True))

    def test_src_matches_the_reference_decoding(self):
        def reference(code, n, lit, neg, absm):
            if code <= 127:
                op = D._sreg(code, n)
            elif 128 <= code <= 192:
                op = {"k": "imm", "v": code - 128}
            elif 193 <= code <= 208:
                op = {"k": "imm", "v": -(code - 192)}
            elif code in D._INLINE_FLOAT:
                op = {"k": "fimm", "v": D._INLINE_FLOAT[code], "abs": False}
            elif code in (251, 252, 253, 254):
                t = {251: "src_vccz", 252: "src_execz", 253: "src_scc", 254: "src_lds_direct"}[code]
                op = {"k": "special", "t": t, "neg": False, "abs": False}
            elif code == 255:
                op = {"k": "imm", "v": 0 if lit is None else lit}
            elif 256 <= code <= 511:
                op = D._vreg(code - 256, n)
            else:
                op = {"k": "raw", "t": f"src_{code}"}
            if absm:
                op["abs"] = True
            if neg:
                op["neg"] = True
            return op

        for code in range(0, 520):
            for n in (0, 1, 2, 3, 4, 8, 16):
                for lit in (None, 0x1234):
                    for neg, absm in ((False, False), (True, False), (False, True), (True, True)):
                        got = D._src(code, n, lit, neg=neg, absm=absm)
                        self.assertEqual(got, reference(code, n, lit, neg, absm), (code, n, lit, neg, absm))
                        # truthy ints (the VOP3 modifier bits) behave like bools, positionally or by keyword
                        got2 = D._src(code, n, lit, 4 if neg else 0, 2 if absm else 0)
                        self.assertEqual(got2, got)
                        self.assertEqual(list(got2), list(got))

    def test_src_results_are_independent_dicts(self):
        a = D._src(3, 1, None)
        b = D._src(3, 1, None)
        a["neg"] = True
        self.assertFalse(b["neg"])
        self.assertFalse(D._src(3, 1, None)["neg"])

    def test_mimg_counts_match_their_uncached_functions_and_stay_bounded(self):
        for fn, args_iter in (
            (D._mimg_addr_count, ((n, d, a) for _, (n, _) in sorted(T.MIMG.items())
                                  for d in range(8) for a in (False, True))),
            (D._mimg_data_count, ((n, m, d16, tfe) for _, (n, _) in sorted(T.MIMG.items())
                                  for m in (0, 1, 3, 5, 8, 15) for d16 in (False, True) for tfe in (False, True))),
        ):
            for args in args_iter:
                self.assertEqual(fn(*args), fn.__wrapped__(*args), args)
            info = fn.cache_info()
            self.assertIsNotNone(info.maxsize)
            self.assertLessEqual(info.currsize, info.maxsize)

    def test_vop3_op_name_matches_the_inline_rule(self):
        def reference(opc, name):
            has_e32 = (opc < 0x100 and opc in T.VOPC) or (0x100 <= opc < 0x140 and (opc - 0x100) in T.VOP2) or \
                      (0x180 <= opc < 0x200 and (opc - 0x180) in T.VOP1)
            op_name = (name + "_e64") if (has_e32 and not name.startswith(("v_mad_", "v_fma_"))) else name
            if opc == 0x100 + 0x06 or opc == 0x100 + 0x1F:
                op_name = name + "_e64"
            return op_name

        for tab in (T.VOP3, T.VOP3B):
            for opc, (name, _) in tab.items():
                self.assertEqual(D._vop3_op_name(opc, name), reference(opc, name))
        self.assertEqual(D._vop3_op_name(0x140, "v_mad_legacy_f32"), reference(0x140, "v_mad_legacy_f32"))
        self.assertIsNotNone(D._vop3_op_name.cache_info().maxsize)

    def test_vop3p_modifier_bits_match_the_per_bit_lists(self):
        # The fast path prints a list only when one of the nsrc real bits is set; compare with the old list tests.
        for opc, (name, spec) in sorted(T.VOP3P.items()):
            nsrc = sum(1 for fn, _, _ in spec if fn in ("SRC0", "SRC1", "SRC2"))
            is_mix = name.startswith("v_fma_mix")
            for op_sel in range(8):
                for op_sel_hi in range(8):
                    for neg_lo, neg_hi in ((0, 0), (1, 0), (0, 4), (7, 7), (6, 1)):
                        w0 = (0xCC000000 | (opc << 16) | (((op_sel_hi >> 2) & 1) << 14) | (op_sel << 11)
                              | (neg_hi << 8))
                        w1 = 0x0201 | (op_sel_hi & 3) << 27 | (neg_lo << 29)
                        ins, = D.decode_bytes(struct.pack("<II", w0, w1))["insns"]
                        sel = [(op_sel >> i) & 1 for i in range(nsrc)]
                        selhi = [(op_sel_hi >> i) & 1 for i in range(nsrc)]
                        lo = [(neg_lo >> i) & 1 for i in range(nsrc)]
                        hi = [(neg_hi >> i) & 1 for i in range(nsrc)]
                        want = []
                        if any(sel):
                            want.append(f"op_sel:[{','.join(map(str, sel))}]")
                        if selhi != ([0] * nsrc if is_mix else [1] * nsrc):
                            want.append(f"op_sel_hi:[{','.join(map(str, selhi))}]")
                        if not is_mix and any(lo):
                            want.append(f"neg_lo:[{','.join(map(str, lo))}]")
                        if not is_mix and any(hi):
                            want.append(f"neg_hi:[{','.join(map(str, hi))}]")
                        self.assertEqual(ins["mods"], want, (name, op_sel, op_sel_hi, neg_lo, neg_hi))

    def test_format_insn_text_matches_the_reference_formatting(self):
        for fam, opc, words in _build(_Rng(7)):
            data = b"".join(struct.pack("<I", w) for w in words)
            for ins in D.decode_bytes(data)["insns"]:
                enc = ",".join(f"0x{b:02x}" for b in bytes.fromhex(ins["enc"]))
                args = ins["args"]
                hex_imm = ins["op"] == "s_ttracedata_imm"
                if ins["op"] == "exp" and args:
                    head = f"{ins['op']} {D._format_operand(args[0], hex_imm)} " + ", ".join(
                        D._format_operand(x, hex_imm) for x in args[1:])
                elif args:
                    head = f"{ins['op']} " + ", ".join(D._format_operand(x, hex_imm) for x in args)
                else:
                    head = ins["op"]
                if ins["mods"]:
                    head += " " + " ".join(ins["mods"])
                self.assertEqual(D.format_insn(ins), f"\t{head:<40s}; encoding: [{enc}]")

    def test_inline_float_text_matches_the_scan_and_tolerates_odd_values(self):
        def reference(v):
            return next((txt for code, txt in D._INLINE_FLOAT_STR.items() if D._INLINE_FLOAT[code] == v), repr(v))

        for v in list(D._INLINE_FLOAT.values()) + [0.0, -0.0, 1, 1.0, 3.5, float("inf"), float("nan"), 0.15915494,
                                                     0.159154943, "x", None, [1.0], {}]:
            self.assertEqual(D._format_operand({"k": "fimm", "v": v}), reference(v), repr(v))

    # ---- decoding never touches the interpreter's collector setting ----------------------------------------------

    def test_decode_leaves_the_garbage_collector_alone(self):
        code = struct.pack("<II", 0xBF800000, 0xBF810000)
        was = gc.isenabled()
        try:
            for state in (True, False):
                (gc.enable if state else gc.disable)()
                D.decode_bytes(code)
                self.assertEqual(gc.isenabled(), state)
                D.disasm_bytes(code)
                self.assertEqual(gc.isenabled(), state)
        finally:
            (gc.enable if was else gc.disable)()

    def test_decode_does_not_reenable_a_collector_another_thread_disabled(self):
        entered, resume = threading.Event(), threading.Event()

        class Sample(bytes):
            def __len__(self):
                entered.set()
                resume.wait(5)
                return super().__len__()

        was = gc.isenabled()
        gc.enable()
        try:
            worker = threading.Thread(target=D.decode_bytes, args=(Sample(struct.pack("<I", 0xBF810000)),))
            worker.start()
            self.assertTrue(entered.wait(5))
            gc.disable()
            resume.set()
            worker.join()
            self.assertFalse(gc.isenabled())
        finally:
            (gc.enable if was else gc.disable)()


if __name__ == "__main__":
    if "--print-digests" in sys.argv:
        for _mcpu in ("gfx1013", "gfx1030"):
            for _w32 in (True, False):
                print(f'    "{_mcpu}/wave{32 if _w32 else 64}": "{corpus_digest(_mcpu, _w32)}",')
        raise SystemExit(0)
    unittest.main()
