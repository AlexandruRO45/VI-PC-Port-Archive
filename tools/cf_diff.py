r"""FUZZ-8: differential fuzzer for control flow and EXEC (standard library only).

Generates seeded RDNA2 pixel-shader programs built from the control-flow shapes compilers emit, runs every lane
of a wave through two independent sides and compares each lane's export bit for bit:

  * tools/cf_ref.py: the ISA model (cpu_ref's Machine plus branches, compares, v_cmpx and saveexec), one wave
  * tools/rdna2spirv.py + tests/semantic/spirv_wave_interp.py: the translated SPIR-V, every invocation of one
    subgroup run in lockstep with structured reconvergence (subgroup size = wave size, the translator's contract)

Shapes (each nests, up to --depth):
  if        v_cmp -> s_and_saveexec -> [s_cbranch_execz] -> body -> s_or exec        (LLVM SI_IF / SI_END_CF)
  ifelse    ... s_xor saveexec mask, s_or_saveexec + s_xor exec for the else side     (LLVM SI_ELSE)
  cmpx      s_mov save, exec -> v_cmpx -> [s_cbranch_execz] -> body -> s_mov exec, save
  vccz      v_cmp -> s_cbranch_vccz/vccnz over a body that every active lane runs      (wave-uniform branch)
  sif       s_cmp -> s_cbranch_scc0/1 [-> s_branch over an else]                       (uniform branch)
  sloop     uniform counter loop closed by s_cbranch_scc1                              (natural loop)
  vloop     per-lane trip counts: break mask, s_andn2 exec, s_cbranch_execnz          (LLVM SI_LOOP)
  irr       two-entry cycle on uniform SCC branches                                   (irreducible graph)
  sccmask   s_and/andn2/or/xor of VCC and EXEC, or s_not of VCC -> s_cbranch_scc0/1 (SCC = whole mask != 0)
  csel      the same SCC read by s_cselect_b32 into a uniform SGPR
  fif       v_cmp_<cc>_f32 / v_cmp_class_f32 -> s_and_saveexec -> [s_cbranch_execz] -> body -> s_or exec
  fcmpx     s_mov save, exec -> v_cmpx_<cc>_f32 / v_cmpx_class_f32 -> [s_cbranch_execz] -> body -> s_mov exec
            (both compare a per-lane f32 built in v12 from a data VGPR: +-1.0..1.875, +-0, +-inf or a NaN, never a
            denormal, against a float constant, one of those specials or a class mask in v13; all 16 conditions,
            so NaNs exercise the ordered, unordered and n forms)
  wrelse    s_and_saveexec -> then -> s_andn2_wrexec D, save (EXEC = save & ~EXEC: the else lanes) ->
            [s_cbranch_execz/scc0] -> else -> s_mov exec, save
  wrloop    per-lane trip counts closed by s_andn1_wrexec D, vcc (EXEC = ~done & EXEC), back edge
            s_cbranch_scc1 or s_cbranch_execnz, then s_mov exec, save
  ifs       v_cmp_*_e64 into an SGPR mask -> s_and_saveexec from it [-> v_cndmask_b32_e64 on it]
  cexec     a constant EXEC (s_mov_b32 exec_lo/exec_hi, literal) around a body
  halfexec  s_and_b32 on one half of EXEC (wave64: exec_lo or exec_hi; the other half's lanes are untouched)
  halfscc   SCC of s_and/andn2/or/xor_b32 on one half of VCC and EXEC -> s_cbranch_scc0/1
  scmpmask  s_cmp_eq/lg of a mask against 0 (u32 on an SGPR mask in wave32, u64 on VCC in wave64)
  cselmask  s_cselect_b32/b64 between a compare mask and EXEC -> s_and_saveexec
  vccloop   per-lane trip counts, back edge s_cbranch_vccnz (the wave loops while any lane wants to)
  cmpxloop  per-lane trip counts, v_cmpx drops finished lanes, back edge s_cbranch_execnz
  mread     s_bcnt1 / s_ff1 of EXEC, VCC or an SGPR compare mask (wave64: _b64, or _b32 on one half) into a
            uniform SGPR, added to every active lane [-> s_cmp on it -> s_cbranch_scc0/1 over a body]
  waterfall v_readfirstlane of a lane value -> v_cmp_eq -> s_and_saveexec -> body -> s_xor exec, back edge
            s_cbranch_execnz: one round per distinct value (the descriptor-per-group-of-lanes loop)
  lanes     v_readlane (added to every active lane) / v_writelane (into a data VGPR) of a constant lane (also one
            past the wave, which wraps) or an SGPR lane, at any depth: EXEC is ignored, so an inactive lane is read
            or written too
  far       the far-branch idiom s_getpc_b64 -> s_add_u32 K -> s_addc_u32 0 -> s_setpc_b64 over a block that never
            runs, or as the back edge of a uniform counter loop (s_sub_u32 K / s_subb_u32 0, or s_add_u32 -K /
            s_addc_u32 -1)
  wqm       s_wqm of EXEC (after narrowing it to a compare) or of VCC into EXEC: every lane of a quad with an active
            lane runs the body, even one an enclosing if had turned off [-> s_cbranch_execz]; EXEC restored with
            s_mov; or s_wqm of a compare into an SGPR -> s_cbranch_scc0/1 (SCC = any quad on)
(vloop also closes some loops with s_cbranch_scc1 on the SCC of its s_andn2 exec.)
Bodies mix integer VALU and SALU ops with "probes": v11 = v11 * 33 + K at each site, so v11 records which sites
each lane ran, in which order and how often. v8/v9 carry data, v10 the uniform SGPRs. Immediates stay in the
inline-constant range so every instruction has a known size, and the listing carries `; encoding:` bytes of
those sizes (zeros: the translator reads only their length), which places the branch targets.

Usage:
    python -I tools/cf_diff.py [--programs N] [--seed S] [--wave 32|64|both] [--dispatch-loop] [--depth D]
    python -I tools/cf_diff.py --self-test
    python -I tools/cf_diff.py --spirv-val ...                              # also spirv-as + spirv-val each module
    python -I tools/cf_diff.py --seed S --program K --show [--minimize]     # print (and shrink) one program

Exit code 0 when every program matches, 1 on a mismatch (case, seed, lane, the expected and actual words and the
listing; with --minimize also a reduced listing), 2 on harness errors or when every program was skipped (nothing was
compared). Skips are counted with reasons.
"""
import argparse
import gc
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

import cf_ref  # noqa: E402
import cpu_ref  # noqa: E402
import rdna2spirv  # noqa: E402
from spirv_wave_interp import UnsupportedConstruct, run_wave  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
DATA = (1, 2, 3, 8, 9)            # VGPRs the bodies compute on (v0 = lane id is kept intact)
SREGS = (10, 11, 12, 13)          # uniform SGPRs the bodies compute on
CCS = ("eq", "ne", "lt", "le", "gt", "ge")
FCCS = ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")
# Float compare operands: inline float constants, and literals for -0, +-inf and a quiet NaN.
FCONSTS = ("0", "1.0", "-1.0", "0.5", "2.0", "-4.0", "0x80000000", "0x7f800000", "0xff800000", "0x7fc00000")
VOP3 = ("v_mul_lo_u32", "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "v_readlane_b32", "v_writelane_b32")


# ---- program tree --------------------------------------------------------------------------------------------
# A program is a list of nodes. ("op", line) is one instruction; compound nodes hold child lists and are expanded
# by Emitter into instructions, labels and branches. Keeping the tree lets --minimize delete whole structures.

class Gen:
    def __init__(self, rng, wave, max_depth, budget):
        self.r, self.wave, self.max_depth, self.budget = rng, wave, max_depth, budget
        self.site = 0

    def cmp(self):
        """A per-lane condition: (cc, type, constant, vgpr). Constants near 0 and 64 make some conditions
        true or false for every lane, so the execz/vccz skips are taken too."""
        r = self.r
        k = r.choice((0, 1, 2, 3, 4, 5, 7, 8, 12, 16, 31, 32, 63, 64, -1, -16))
        return (r.choice(CCS), r.choice(("u32", "u32", "i32")), k, r.choice((0, 0, 1, 2, 3, 8, 9)))

    def fcond(self):
        """A per-lane float condition: (cc, operand, source VGPR). cc is one of the 16 f32 conditions or "class";
        the operand is a float constant or special (as a literal), or a 10-bit class mask."""
        r = self.r
        if r.random() < 0.25:
            return ("class", r.randrange(1, 1024), r.choice(DATA))
        return (r.choice(FCCS), r.choice(FCONSTS), r.choice(DATA))

    def half(self):
        """Which 32-bit half of a mask a _b32 op works on: always the low (whole) one in wave32."""
        return self.r.choice(("lo", "hi")) if self.wave == 64 else "lo"

    def probe(self):
        self.site += 1
        return ("probe", 1 + (self.site * 37) % 64)

    def leaf(self):
        r = self.r
        d, s = r.choice(DATA), r.choice(DATA)
        k = r.randrange(-16, 65)
        sr = r.choice(SREGS)
        choices = [
            ("op", f"v_add_nc_u32 v{d}, {k}, v{s}"),
            ("op", f"v_sub_nc_u32 v{d}, v{s}, v{r.choice(DATA)}"),
            ("op", f"v_xor_b32 v{d}, s{sr}, v{s}"),
            ("op", f"v_and_b32 v{d}, {r.randrange(0, 65)}, v{s}"),
            ("op", f"v_or_b32 v{d}, v{r.choice(DATA)}, v{s}"),
            ("op", f"v_lshlrev_b32 v{d}, {r.randrange(0, 8)}, v{s}"),
            ("op", f"v_lshrrev_b32 v{d}, {r.randrange(0, 8)}, v{s}"),
            ("op", f"v_mul_lo_u32 v{d}, v{s}, {r.randrange(0, 65)}"),
            ("op", f"v_mov_b32 v{d}, s{sr}"),
            ("op", f"v_add_nc_u32 v{d}, s{sr}, v{s}"),
            ("op", f"s_add_u32 s{sr}, s{sr}, {r.randrange(0, 65)}"),
            ("op", f"s_xor_b32 s{sr}, s{r.choice(SREGS)}, {r.randrange(0, 65)}"),
            ("op", f"s_mov_b32 s{sr}, {r.randrange(-16, 65)}"),
            ("cnd", d, s, self.cmp()),
        ]
        return r.choice(choices)

    def block(self, depth):
        out = []
        n = self.r.randrange(1, 4)
        for _ in range(n):
            if self.budget <= 0:
                break
            self.budget -= 1
            if depth < self.max_depth and self.r.random() < 0.55:
                out.append(self.compound(depth + 1))
            else:
                out.append(self.leaf())
            if self.r.random() < 0.6:
                out.append(self.probe())
        return out

    def compound(self, depth):
        r = self.r
        kind = r.choice(("if", "if", "ifelse", "ifelse", "cmpx", "saveexec", "vccz", "sif", "sloop", "vloop",
                         "vloop", "irr", "sccmask", "csel", "ifs", "ifs", "cexec", "halfexec", "halfscc", "scmpmask",
                         "cselmask", "vccloop", "cmpxloop", "mread", "mread", "waterfall",
                         "waterfall", "wqm", "wqm", "lanes", "lanes", "far", "wrelse", "wrloop", "fif", "fcmpx"))
        if kind in ("fif", "fcmpx"):
            return (kind, self.fcond(), r.random() < 0.6, [self.probe()] + self.block(depth))
        if kind == "far":
            if r.random() < 0.5:
                return ("far", "fwd", 0, [self.probe()] + self.block(depth))
            return ("far", r.choice(("sub", "addneg")), r.randrange(1, 4), [self.probe()] + self.block(depth))
        if kind == "lanes":
            lane = f"s{r.choice(SREGS)}" if r.random() < 0.3 else r.randrange(0, 2 * self.wave)
            # Writes favour v8/v9, which go straight to the export, so a lost write to an inactive lane shows.
            rw = r.choice(("read", "write"))
            v = r.choice((8, 9, 8, 9, 1, 2, 3)) if rw == "write" else r.choice(DATA)
            return ("lanes", rw, lane, v, r.choice(SREGS))
        if kind == "wqm":
            width = r.choice(("b64", "b32")) if self.wave == 64 else "b32"
            return ("wqm", self.cmp(), r.choice(("exec", "vcc", "scc")), width,
                    self.half() if width == "b32" else "lo", r.choice((None, "s_cbranch_execz", "s_cbranch_scc0",
                                                                       "s_cbranch_scc1")),
                    [self.probe()] + self.block(depth))
        if kind == "mread":
            width = r.choice(("b64", "b32")) if self.wave == 64 else "b32"
            branch = (r.choice(CCS), r.randrange(0, 66), r.choice(("s_cbranch_scc0", "s_cbranch_scc1"))) \
                if r.random() < 0.5 else None
            return ("mread", self.cmp(), r.choice(("bcnt1", "ff1")), r.choice(("exec", "vcc", "sgpr")), width,
                    self.half() if width == "b32" else "lo", branch,
                    [self.probe()] + self.block(depth) if branch else [])
        if kind == "waterfall":
            # v1 / v2 / v3 hold 8 or 16 distinct values; a source with one value per lane (v8) costs 64 rounds and,
            # nested, runs past the wave interpreter's step limit.
            return ("waterfall", r.choice((1, 2, 3)), r.random() < 0.5, [self.probe()] + self.block(depth))
        if kind == "wrelse":
            return ("wrelse", self.cmp(), r.random() < 0.5, r.choice((None, "s_cbranch_execz", "s_cbranch_scc0")),
                    [self.probe()] + self.block(depth), [self.probe()] + self.block(depth))
        if kind == "wrloop":
            return ("wrloop", (r.choice((0, 1, 2, 3)), r.choice((1, 3))),
                    r.choice(("s_cbranch_scc1", "s_cbranch_execnz")), [self.probe()] + self.block(depth))
        if kind == "saveexec":
            return ("saveexec", self.cmp(),
                    r.choice(("andn1", "andn2", "orn1", "orn2", "xor", "nand", "nor", "xnor")),
                    r.choice((None, "s_cbranch_execz", "s_cbranch_scc0")), [self.probe()] + self.block(depth))
        if kind in ("vccloop", "cmpxloop"):
            return (kind, (r.choice((0, 1, 2, 3)), r.choice((1, 3))), [self.probe()] + self.block(depth))
        if kind == "ifs":
            return ("ifs", self.cmp(), r.random() < 0.7, r.random() < 0.5, [self.probe()] + self.block(depth))
        if kind == "cexec":
            return ("cexec", r.getrandbits(32), r.getrandbits(32), [self.probe()] + self.block(depth))
        if kind == "halfexec":
            return ("halfexec", self.cmp(), self.half(), [self.probe()] + self.block(depth))
        if kind == "halfscc":
            return ("halfscc", self.cmp(), r.choice(("and", "andn2", "or", "xor")), self.half(),
                    r.choice(("s_cbranch_scc0", "s_cbranch_scc1")), [self.probe()] + self.block(depth))
        if kind == "scmpmask":
            return ("scmpmask", self.cmp(), r.choice(("eq", "lg")), r.choice(("s_cbranch_scc0", "s_cbranch_scc1")),
                    [self.probe()] + self.block(depth))
        if kind == "cselmask":
            return ("cselmask", (r.choice(SREGS), r.randrange(0, 65)), self.cmp(), [self.probe()] + self.block(depth))
        if kind == "sccmask":
            return ("sccmask", self.cmp(), r.choice(("and", "andn2", "or", "xor", "not")),
                    r.choice(("s_cbranch_scc0", "s_cbranch_scc1")), [self.probe()] + self.block(depth))
        if kind == "csel":
            return ("csel", self.cmp(), r.choice(("and", "andn2", "or", "xor", "not")), r.choice(SREGS))
        if kind == "if":
            return ("if", self.cmp(), r.random() < 0.7, [self.probe()] + self.block(depth))
        if kind == "ifelse":
            return ("ifelse", self.cmp(), [self.probe()] + self.block(depth), [self.probe()] + self.block(depth))
        if kind == "cmpx":
            return ("cmpx", self.cmp(), r.random() < 0.7, [self.probe()] + self.block(depth))
        if kind == "vccz":
            return ("vccz", self.cmp(), r.choice(("s_cbranch_vccz", "s_cbranch_vccnz")),
                    [self.probe()] + self.block(depth))
        if kind == "sif":
            other = [self.probe()] + self.block(depth) if r.random() < 0.5 else None
            return ("sif", (r.choice(CCS), r.choice(SREGS), r.randrange(0, 65)), r.random() < 0.5,
                    [self.probe()] + self.block(depth), other)
        if kind == "sloop":
            return ("sloop", r.randrange(1, 4), [self.probe()] + self.block(depth))
        if kind == "vloop":
            return ("vloop", (r.choice((0, 1, 2, 3)), r.choice((1, 3)), r.random() < 0.3),
                    self.cmp() if r.random() < 0.4 else None, [self.probe()] + self.block(depth))
        return ("irr", (r.choice(SREGS), r.randrange(0, 65)), [self.probe()] + self.block(depth),
                [self.probe()] + self.block(depth))


def generate(rng, wave, max_depth=3, budget=10):
    g = Gen(rng, wave, max_depth, budget)
    body = []
    while g.budget > 0:
        body.extend(g.block(0))
    body.append(g.probe())
    return body


class Emitter:
    """Expands a program tree into tab-less instruction lines with sizes, labels and symbolic branches."""

    def __init__(self, wave):
        self.wave = wave
        self.items = []            # ("ins", text) | ("label", name) | ("br", mnemonic, label)
        self.n_labels = 0
        self.depth = 0

    # register names -------------------------------------------------------------------------------------------
    def exec_(self):
        return "exec" if self.wave == 64 else "exec_lo"

    def vcc(self):
        return "vcc" if self.wave == 64 else "vcc_lo"

    def b(self):
        return "b64" if self.wave == 64 else "b32"

    def sreg(self, base):
        """A mask SGPR for the current nesting depth (an aligned pair in wave64)."""
        n = base + 2 * self.depth
        return f"s[{n}:{n + 1}]" if self.wave == 64 else f"s{n}"

    def label(self):
        self.n_labels += 1
        return f"L{self.n_labels}"

    def ins(self, text):
        self.items.append(("ins", text))

    def br(self, mnemonic, label):
        self.items.append(("br", mnemonic, label))

    def place(self, label):
        self.items.append(("label", label))

    def mask_logic(self, logic, dst):
        """A mask op on VCC (and EXEC): s_<logic> dst, vcc, exec, or s_not dst, vcc (SOP1)."""
        if logic == "not":
            return f"s_not_{self.b()} {dst}, {self.vcc()}"
        return f"s_{logic}_{self.b()} {dst}, {self.vcc()}, {self.exec_()}"

    def fvalue(self, src):
        """v12 = a per-lane f32 from data VGPR `src`: bits 0-2 pick a mantissa (1.0..1.875), bits 4-5 swap in a
        quiet NaN, +inf or +0, bit 3 sets the sign (so -inf, -0 and negative NaNs occur too). Never a denormal, so
        the compare does not depend on the denorm mode. Clobbers VCC and v13."""
        self.ins(f"v_and_b32 v12, 7, v{src}")
        self.ins("v_lshlrev_b32 v12, 20, v12")
        self.ins("v_or_b32 v12, 0x3f800000, v12")
        self.ins(f"v_and_b32 v13, 48, v{src}")
        for sel, special in ((16, "0x7fc00000"), (32, "0x7f800000"), (48, "0")):
            self.ins(f"v_cmp_ne_u32_e32 {self.vcc()}, {sel}, v13")
            self.ins(f"v_cndmask_b32_e32 v12, {special}, v12, {self.vcc()}")
        self.ins(f"v_and_b32 v13, 8, v{src}")
        self.ins("v_lshlrev_b32 v13, 28, v13")
        self.ins("v_xor_b32 v12, v12, v13")

    def fcompare(self, fc, cmpx):
        """The float compare of fc on v12: into VCC (v_cmp) or EXEC (v_cmpx)."""
        cc, operand, src = fc
        self.fvalue(src)
        dst = "" if cmpx else f"{self.vcc()}, "
        mn = "v_cmpx" if cmpx else "v_cmp"
        if cc == "class":
            self.ins(f"v_mov_b32 v13, {operand}")
            self.ins(f"{mn}_class_f32_e32 {dst}v12, v13")
        else:
            self.ins(f"{mn}_{cc}_f32_e32 {dst}{operand}, v12")

    def vcmp(self, c, dst=None):
        cc, ty, k, v = c
        self.ins(f"v_cmp_{cc}_{ty}_e32 {dst or self.vcc()}, {k}, v{v}")

    # program ------------------------------------------------------------------------------------------------------
    def program(self, body):
        if self.wave == 64:
            self.ins("v_mbcnt_lo_u32_b32 v0, -1, 0")
            self.ins("v_mbcnt_hi_u32_b32 v0, -1, v0")
        else:
            self.ins("v_mbcnt_lo_u32_b32 v0, -1, 0")
        self.ins("v_and_b32 v1, 7, v0")
        self.ins("v_lshrrev_b32 v2, 2, v0")
        self.ins("v_mul_lo_u32 v3, v0, 5")
        self.ins("v_and_b32 v3, 15, v3")
        self.ins("v_xor_b32 v8, 9, v0")
        self.ins("v_mov_b32 v9, 0")
        self.ins("v_mov_b32 v11, 1")
        for i, s in enumerate(SREGS):
            self.ins(f"s_mov_b32 s{s}, {(3, 5, 0, 12)[i]}")
        self.nodes(body)
        self.ins(f"v_mov_b32 v10, s{SREGS[0]}")
        for s in SREGS[1:]:
            self.ins("v_lshlrev_b32 v10, 3, v10")
            self.ins(f"v_xor_b32 v10, s{s}, v10")
        self.ins("exp mrt0 v8, v9, v10, v11 done vm")
        self.ins("s_endpgm")

    def nodes(self, nodes):
        for n in nodes:
            self.node(n)

    def node(self, n):
        kind = n[0]
        w = self.b()
        ex = self.exec_()
        if kind == "op":
            self.ins(n[1])
        elif kind == "probe":
            self.ins("v_mul_lo_u32 v11, v11, 33")
            self.ins(f"v_add_nc_u32 v11, {n[1]}, v11")
        elif kind == "cnd":
            _, d, s, c = n
            self.vcmp(c)
            self.ins(f"v_cndmask_b32_e32 v{d}, 7, v{s}, {self.vcc()}")
        elif kind == "if":
            _, c, skip, body = n
            save, end = self.sreg(20), self.label()
            self.vcmp(c)
            self.ins(f"s_and_saveexec_{w} {save}, {self.vcc()}")
            if skip:
                self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_or_{w} {ex}, {ex}, {save}")
        elif kind == "ifelse":
            _, c, then, other = n
            save, els, end = self.sreg(20), self.label(), self.label()
            self.vcmp(c)
            self.ins(f"s_and_saveexec_{w} {save}, {self.vcc()}")
            self.ins(f"s_xor_{w} {save}, {ex}, {save}")
            self.br("s_cbranch_execz", els)
            self.depth += 1
            self.nodes(then)
            self.depth -= 1
            self.place(els)
            self.ins(f"s_or_saveexec_{w} {save}, {save}")
            self.ins(f"s_xor_{w} {ex}, {ex}, {save}")
            self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(other)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_or_{w} {ex}, {ex}, {save}")
        elif kind == "cmpx":
            _, (cc, ty, k, v), skip, body = n
            save, end = self.sreg(20), self.label()
            self.ins(f"s_mov_{w} {save}, {ex}")
            self.ins(f"v_cmpx_{cc}_{ty}_e32 {k}, v{v}")
            if skip:
                self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "saveexec":
            _, c, op_name, skip_br, body = n
            save, end = self.sreg(20), self.label()
            self.vcmp(c)
            self.ins(f"s_{op_name}_saveexec_{w} {save}, {self.vcc()}")
            if skip_br is not None:
                self.br(skip_br, end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "vccz":
            _, c, mnemonic, body = n
            end = self.label()
            self.vcmp(c)
            self.br(mnemonic, end)
            self.nodes(body)
            self.place(end)
        elif kind == "sif":
            _, (cc, sr, k), on_one, then, other = n
            els, end = self.label(), self.label()
            self.ins(f"s_cmp_{cc}_u32 s{sr}, {k}")
            self.br("s_cbranch_scc0" if on_one else "s_cbranch_scc1", els if other else end)
            self.nodes(then)
            if other:
                self.br("s_branch", end)
                self.place(els)
                self.nodes(other)
            self.place(end)
        elif kind == "sloop":
            _, trips, body = n
            ctr, head = 14 + self.depth, self.label()
            self.ins(f"s_mov_b32 s{ctr}, 0")
            self.place(head)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.ins(f"s_add_u32 s{ctr}, s{ctr}, 1")
            self.ins(f"s_cmp_lt_u32 s{ctr}, {trips}")
            self.br("s_cbranch_scc1", head)
        elif kind == "mread":
            # A scalar read of a whole mask: one value for the wave (a ballot's bit count or lowest bit).
            _, c, what, src, width, half, branch, body = n
            dst = f"s{100 + self.depth}"
            if src == "exec":
                m = ex if width == "b64" or self.wave == 32 else f"exec_{half}"
            elif src == "vcc":
                self.vcmp(c)
                m = self.vcc() if width == "b64" or self.wave == 32 else f"vcc_{half}"
            else:
                cc, ty, k, v = c
                pair = self.sreg(60)
                self.ins(f"v_cmp_{cc}_{ty}_e64 {pair}, {k}, v{v}")
                n0 = 60 + 2 * self.depth
                m = pair if width == "b64" or self.wave == 32 else f"s{n0 + (half == 'hi')}"
            self.ins(f"s_{what}_i32_{width} {dst}, {m}")
            self.ins(f"v_add_nc_u32 v11, {dst}, v11")
            if branch:
                cc, k, mnemonic = branch
                end = self.label()
                self.ins(f"s_cmp_{cc}_u32 {dst}, {k}")
                self.br(mnemonic, end)
                self.depth += 1
                self.nodes(body)
                self.depth -= 1
                self.place(end)
        elif kind == "far":
            # s[2:3] is scratch: the four instructions use it and nothing reads it after the jump.
            _, form, trips, body = n
            if form == "fwd":
                end = self.label()
                self.items.append(("far", "add", end))
                self.depth += 1
                self.nodes(body)            # never runs
                self.depth -= 1
                self.place(end)
            else:
                ctr, head, out_ = 14 + self.depth, self.label(), self.label()
                self.ins(f"s_mov_b32 s{ctr}, 0")
                self.place(head)
                self.depth += 1
                self.nodes(body)
                self.depth -= 1
                self.ins(f"s_add_u32 s{ctr}, s{ctr}, 1")
                self.ins(f"s_cmp_ge_u32 s{ctr}, {trips}")
                self.br("s_cbranch_scc1", out_)
                self.items.append(("far", form, head))
                self.place(out_)
        elif kind == "lanes":
            _, rw, lane, v, sr = n
            sel = lane if isinstance(lane, str) else str(lane)
            if rw == "read":
                dst = f"s{100 + self.depth}"
                self.ins(f"v_readlane_b32 {dst}, v{v}, {sel}")
                self.ins(f"v_add_nc_u32 v11, {dst}, v11")
            else:
                self.ins(f"v_writelane_b32 v{v}, s{sr}, {sel}")
        elif kind == "wqm":
            _, c, src, width, half, mnemonic, body = n
            save, end = self.sreg(80), self.label()
            half_reg = width == "b32" and self.wave == 64
            exm = f"exec_{half}" if half_reg else ex
            vcm = f"vcc_{half}" if half_reg else self.vcc()
            if src == "scc":
                # WQM of a compare into an SGPR: SCC = some quad has a lane on; a uniform branch on it.
                tmp = self.sreg(50)
                tmp = f"s{50 + 2 * self.depth + (half == 'hi')}" if half_reg else tmp
                self.vcmp(c)
                self.ins(f"s_wqm_{width} {tmp}, {vcm}")
                self.br(mnemonic if mnemonic and "scc" in mnemonic else "s_cbranch_scc0", end)
                self.depth += 1
                self.nodes(body)
                self.depth -= 1
                self.place(end)
            else:
                self.ins(f"s_mov_{w} {save}, {ex}")
                self.vcmp(c)
                if src == "exec":
                    self.ins(f"s_and_{w} {ex}, {ex}, {self.vcc()}")
                    self.ins(f"s_wqm_{width} {exm}, {exm}")
                else:
                    self.ins(f"s_wqm_{width} {exm}, {vcm}")
                if mnemonic:
                    self.br(mnemonic, end)
                self.depth += 1
                self.nodes(body)
                self.depth -= 1
                self.place(end)
                self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "waterfall":
            # Each round takes the first active lane's value, runs the body for every lane holding it and drops
            # those lanes; the loop ends when EXEC is empty, then EXEC is restored.
            _, src, scc_exit, body = n
            val, save, t, head = f"s{100 + self.depth}", self.sreg(80), self.sreg(90), self.label()
            self.ins(f"s_mov_{w} {save}, {ex}")
            self.place(head)
            self.ins(f"v_readfirstlane_b32 {val}, v{src}")
            self.ins(f"v_cmp_eq_u32_e32 {self.vcc()}, {val}, v{src}")
            self.ins(f"s_and_saveexec_{w} {t}, {self.vcc()}")
            self.ins(f"v_add_nc_u32 v11, {val}, v11")
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.ins(f"s_xor_{w} {ex}, {ex}, {t}")
            self.br("s_cbranch_scc1" if scc_exit else "s_cbranch_execnz", head)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "vccloop":
            # A loop whose back edge is a VCC vote: the wave goes round again while any lane's own trip count is
            # not reached, and every active lane runs every round (EXEC is not changed).
            _, (src, mask), body = n
            it, lim, head = 4 + self.depth, 30 + self.depth, self.label()
            self.ins(f"v_and_b32 v{lim}, {mask}, v{src}")
            self.ins(f"v_mov_b32 v{it}, 0")
            self.place(head)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.ins(f"v_add_nc_u32 v{it}, 1, v{it}")
            self.ins(f"v_cmp_le_u32_e32 {self.vcc()}, v{it}, v{lim}")
            self.br("s_cbranch_vccnz", head)
        elif kind == "cmpxloop":
            # A loop that drops lanes with v_cmpx as their trip count is reached and closes on EXECNZ.
            _, (src, mask), body = n
            it, lim, save, head = 4 + self.depth, 30 + self.depth, self.sreg(40), self.label()
            self.ins(f"v_and_b32 v{lim}, {mask}, v{src}")
            self.ins(f"v_mov_b32 v{it}, 0")
            self.ins(f"s_mov_{w} {save}, {ex}")
            self.place(head)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.ins(f"v_add_nc_u32 v{it}, 1, v{it}")
            self.ins(f"v_cmpx_le_u32_e32 v{it}, v{lim}")
            self.br("s_cbranch_execnz", head)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "ifs":
            # The compare writes an SGPR mask (VOP3), which feeds the saveexec; some bodies also v_cndmask on it.
            _, c, skip, use_cnd, body = n
            cc, ty, k, v = c
            m, save, end = self.sreg(60), self.sreg(20), self.label()
            self.ins(f"v_cmp_{cc}_{ty}_e64 {m}, {k}, v{v}")
            self.ins(f"s_and_saveexec_{w} {save}, {m}")
            if skip:
                self.br("s_cbranch_execz", end)
            if use_cnd:
                self.ins(f"v_cndmask_b32_e64 v9, 3, v8, {m}")
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_or_{w} {ex}, {ex}, {save}")
        elif kind == "cexec":
            # A constant EXEC: each lane takes its own bit of the literal (also lanes that were off).
            _, klo, khi, body = n
            save = self.sreg(20)
            self.ins(f"s_mov_{w} {save}, {ex}")
            self.ins(f"s_mov_b32 exec_lo, 0x{klo:x}")
            if self.wave == 64:
                self.ins(f"s_mov_b32 exec_hi, 0x{khi:x}")
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "halfexec":
            # A 32-bit op on one half of EXEC: the lanes of the other half keep their state.
            _, c, half, body = n
            save = self.sreg(20)
            self.vcmp(c)
            self.ins(f"s_mov_{w} {save}, {ex}")
            self.ins(f"s_and_b32 exec_{half}, exec_{half}, vcc_{half}")
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "halfscc":
            # SCC of a 32-bit op on one half: whether that half's result is non-zero (in wave32, the whole mask).
            _, c, logic, half, mnemonic, body = n
            end = self.label()
            self.vcmp(c)
            self.ins(f"s_{logic}_b32 s50, vcc_{half}, exec_{half}")
            self.br(mnemonic, end)
            self.nodes(body)
            self.place(end)
        elif kind == "scmpmask":
            # s_cmp of a mask against 0: s_cmp_eq/lg_u32 on an SGPR mask (wave32) or _u64 on VCC (wave64).
            _, c, cc, mnemonic, body = n
            end = self.label()
            if self.wave == 64:
                self.vcmp(c)
                self.ins(f"s_cmp_{cc}_u64 vcc, 0")
            else:
                cc_, ty, k, v = c
                self.ins(f"v_cmp_{cc_}_{ty}_e64 s60, {k}, v{v}")
                self.ins(f"s_cmp_{cc}_u32 s60, 0")
            self.br(mnemonic, end)
            self.nodes(body)
            self.place(end)
        elif kind == "cselmask":
            # A uniform s_cmp picks one of two masks (compare result or EXEC) for the saveexec.
            _, (sr, k), c, body = n
            cc, ty, kk, v = c
            m, t, save, end = self.sreg(60), self.sreg(70), self.sreg(20), self.label()
            self.ins(f"v_cmp_{cc}_{ty}_e64 {m}, {kk}, v{v}")
            self.ins(f"s_cmp_lt_u32 s{sr}, {k}")
            self.ins(f"s_cselect_{w} {t}, {m}, {ex}")
            self.ins(f"s_and_saveexec_{w} {save}, {t}")
            self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_or_{w} {ex}, {ex}, {save}")
        elif kind == "sccmask":
            # SCC of a mask op is (whole result != 0): one value for the wave, like a vote.
            _, c, logic, mnemonic, body = n
            tmp, end = self.sreg(50), self.label()
            self.vcmp(c)
            self.ins(self.mask_logic(logic, tmp))
            self.br(mnemonic, end)
            self.nodes(body)
            self.place(end)
        elif kind == "csel":
            _, c, logic, sr = n
            self.vcmp(c)
            self.ins(self.mask_logic(logic, self.sreg(50)))
            self.ins(f"s_cselect_b32 s{sr}, s{sr}, 17")
        elif kind == "vloop":
            _, (src, mask, scc_exit), brk, body = n
            it, lim, brk_mask, head = 4 + self.depth, 30 + self.depth, self.sreg(40), self.label()
            self.ins(f"v_and_b32 v{lim}, {mask}, v{src}")
            self.ins(f"v_add_nc_u32 v{lim}, 1, v{lim}")
            self.ins(f"v_mov_b32 v{it}, 0")
            self.ins(f"s_mov_{w} {brk_mask}, 0")
            self.place(head)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            if brk is not None:
                # An early exit (SI_IF_BREAK): lanes meeting `brk` join the break mask for this iteration.
                self.vcmp(brk)
                self.ins(f"s_or_{w} {brk_mask}, {self.vcc()}, {brk_mask}")
            self.ins(f"v_add_nc_u32 v{it}, 1, v{it}")
            self.ins(f"v_cmp_ge_u32_e32 {self.vcc()}, v{it}, v{lim}")
            self.ins(f"s_or_{w} {brk_mask}, {self.vcc()}, {brk_mask}")
            self.ins(f"s_andn2_{w} {ex}, {ex}, {brk_mask}")
            # Loop again while any lane is left: EXECNZ, or SCC of the s_andn2 (= new EXEC != 0).
            self.br("s_cbranch_scc1" if scc_exit else "s_cbranch_execnz", head)
            self.ins(f"s_or_{w} {ex}, {ex}, {brk_mask}")
        elif kind == "fif":
            _, fc, skip, body = n
            save, end = self.sreg(20), self.label()
            self.fcompare(fc, cmpx=False)
            self.ins(f"s_and_saveexec_{w} {save}, {self.vcc()}")
            if skip:
                self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_or_{w} {ex}, {ex}, {save}")
        elif kind == "fcmpx":
            _, fc, skip, body = n
            save, end = self.sreg(20), self.label()
            self.ins(f"s_mov_{w} {save}, {ex}")
            self.fcompare(fc, cmpx=True)
            if skip:
                self.br("s_cbranch_execz", end)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "wrelse":
            # The else switch through s_andn2_wrexec: EXEC = save & ~EXEC is the lanes of `save` the then side did
            # not run, and D gets the same mask (D is scratch, as for sccmask: nothing reads it).
            _, c, skip_then, skip_else, then, other = n
            save, dst, els, end = self.sreg(20), self.sreg(50), self.label(), self.label()
            self.vcmp(c)
            self.ins(f"s_and_saveexec_{w} {save}, {self.vcc()}")
            if skip_then:
                self.br("s_cbranch_execz", els)
            self.depth += 1
            self.nodes(then)
            self.depth -= 1
            self.place(els)
            self.ins(f"s_andn2_wrexec_{w} {dst}, {save}")
            if skip_else is not None:
                self.br(skip_else, end)
            self.depth += 1
            self.nodes(other)
            self.depth -= 1
            self.place(end)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "wrloop":
            # A loop whose finished lanes leave through s_andn1_wrexec: EXEC = ~vcc & EXEC, SCC = (EXEC != 0).
            _, (src, mask), back, body = n
            it, lim, save, dst, head = 4 + self.depth, 30 + self.depth, self.sreg(20), self.sreg(50), self.label()
            self.ins(f"v_and_b32 v{lim}, {mask}, v{src}")
            self.ins(f"v_add_nc_u32 v{lim}, 1, v{lim}")
            self.ins(f"v_mov_b32 v{it}, 0")
            self.ins(f"s_mov_{w} {save}, {ex}")
            self.place(head)
            self.depth += 1
            self.nodes(body)
            self.depth -= 1
            self.ins(f"v_add_nc_u32 v{it}, 1, v{it}")
            self.ins(f"v_cmp_ge_u32_e32 {self.vcc()}, v{it}, v{lim}")
            self.ins(f"s_andn1_wrexec_{w} {dst}, {self.vcc()}")
            self.br(back, head)
            self.ins(f"s_mov_{w} {ex}, {save}")
        elif kind == "irr":
            _, (sr, k), part_a, part_b = n
            ctr, a_lbl, b_lbl, end = 14 + self.depth, self.label(), self.label(), self.label()
            self.ins(f"s_mov_b32 s{ctr}, 0")
            self.ins(f"s_cmp_lt_u32 s{sr}, {k}")
            self.br("s_cbranch_scc0", b_lbl)
            self.place(a_lbl)
            self.depth += 1
            self.nodes(part_a)
            self.ins(f"s_add_u32 s{ctr}, s{ctr}, 1")
            self.ins(f"s_cmp_lt_u32 s{ctr}, 4")
            self.br("s_cbranch_scc0", end)
            self.place(b_lbl)
            self.nodes(part_b)
            self.depth -= 1
            self.ins(f"s_add_u32 s{ctr}, s{ctr}, 2")
            self.ins(f"s_cmp_lt_u32 s{ctr}, 7")
            self.br("s_cbranch_scc1", a_lbl)
            self.place(end)
        else:
            raise ValueError(f"unknown node {kind}")

    def assemble(self):
        """Lines with real branch offsets: [(text, size_in_bytes)]."""
        pc, where, sized = 0, {}, []
        for it in self.items:
            if it[0] == "label":
                where[it[1]] = pc
                continue
            size = 4 if it[0] == "br" else FAR_SIZE if it[0] == "far" else _size(it[1])
            sized.append((it, pc, size))
            pc += size
        out = []
        for it, at, size in sized:
            if it[0] == "br":
                off = (where[it[2]] - (at + 4)) // 4
                out.append((f"{it[1]} {off}", size))
            elif it[0] == "far":
                # s_getpc_b64 gives at + 4; the target is that plus K (the s_add_u32 literal always takes 4 bytes).
                _, form, label = it
                k = where[label] - (at + 4)
                if form == "add":
                    lines = [f"s_add_u32 s2, s2, 0x{k & 0xFFFFFFFF:x}", "s_addc_u32 s3, s3, 0"]
                elif form == "sub":
                    lines = [f"s_sub_u32 s2, s2, 0x{-k & 0xFFFFFFFF:x}", "s_subb_u32 s3, s3, 0"]
                else:
                    lines = [f"s_add_u32 s2, s2, 0x{k & 0xFFFFFFFF:x}", "s_addc_u32 s3, s3, -1"]
                for text, sz in [("s_getpc_b64 s[2:3]", 4), (lines[0], 8), (lines[1], 4), ("s_setpc_b64 s[2:3]", 4)]:
                    out.append((text, sz))
            else:
                out.append((it[1], size))
        return out


FAR_SIZE = 20       # s_getpc_b64 (4) + s_add/sub_u32 with a literal (8) + s_addc/subb_u32 (4) + s_setpc_b64 (4)


def _size(text):
    """Bytes of the encoding: 4, or 8 for VOP3 and exp, plus 4 for a literal (an integer outside -16..64)."""
    op, _, rest = text.partition(" ")
    base = 8 if op == "exp" or op.endswith("_e64") or op in VOP3 else 4
    return base + (4 if any(not -16 <= v <= 64 for v in _ints(rest)) else 0)


def _ints(operands):
    out = []
    for t in (x.strip() for x in operands.split(",")):
        try:
            out.append(int(t, 0))
        except ValueError:
            pass
    return out


def listing_text(lines, name):
    out = [f"; {name}  code 0x0..0x{sum(s for _, s in lines):x}"]
    for text, size in lines:
        out.append("\t" + text + " ; encoding: [" + ",".join(["0x00"] * size) + "]")
    return "\n".join(out) + "\n"


def build(body, wave):
    e = Emitter(wave)
    e.program(body)
    return e.assemble()


# ---- one comparison ------------------------------------------------------------------------------------------

def run_case(name, body, wave, structured=True):
    """Returns (status, detail): status in {"match", "mismatch", "skip"}."""
    lines = build(body, wave)
    text = listing_text(lines, name)
    with tempfile.TemporaryDirectory(prefix="cf-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            ref = cf_ref.run(path, wave=wave)
        except cpu_ref.CpuRefError as e:
            return ("skip", "cf_ref: " + re.sub(r" at pc 0x[0-9a-f]+", "", str(e)))
        old = rdna2spirv.STRUCTURED
        rdna2spirv.STRUCTURED = structured
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(path)
                gen = rdna2spirv.translate.last_gen
                gc.collect()
        except rdna2spirv.Unsupported as e:
            return ("skip", f"translator unsupported: {e}")
        finally:
            rdna2spirv.STRUCTURED = old
        if SPIRV_VAL:
            bad = _spirv_val(spv, tmp)
            if bad:
                return ("mismatch", f"spirv-val: {bad}")
    if bool(getattr(gen, "wave64", False)) != (wave == 64):
        return ("mismatch", f"translator detected wave{64 if gen.wave64 else 32}, the program is wave{wave}")
    try:
        got = run_wave(spv, wave=wave)["outputs"]
    except UnsupportedConstruct as e:
        return ("skip", f"wave interp: {e}")
    for lane in range(wave):
        want = ref["exports"].get(lane)
        have = (got[lane] or {}).get("%out_mrt0") if got[lane] is not None else None
        if want == have:
            continue
        if want is None or have is None:
            return ("mismatch", f"lane {lane}: export {'missing' if have is None else 'unexpected'} in the "
                                f"translation (reference {_words(want)}, translation {_words(have)})")
        bad = [i for i in range(4) if want[i] != have[i]]
        what = {0: "v8 (data)", 1: "v9 (data)", 2: "v10 (uniform SGPRs)", 3: "v11 (path probes)"}
        return ("mismatch", f"lane {lane}: {', '.join(what[i] for i in bad)} differ: reference {_words(want)}, "
                            f"translation {_words(have)}")
    return ("match", "")


SPIRV_VAL = False       # --spirv-val: also assemble and validate every translation (spirv-as/spirv-val on PATH)


def _spirv_val(spv, tmp):
    """First spirv-as/spirv-val error for the module (Vulkan 1.2, as tests/conftest.py), or None."""
    sa, sv = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
    with open(sa, "w", encoding="utf-8", newline="\n") as f:
        f.write(spv)
    for cmd in (["spirv-as", sa, "-o", sv, "--target-env", "vulkan1.2"],
                ["spirv-val", "--target-env", "vulkan1.2", sv]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return (cmd[0] + ": " + (r.stderr or r.stdout).strip().splitlines()[0]) if (r.stderr or r.stdout) \
                else f"{cmd[0]} exited with {r.returncode}"
    return None


def _words(w):
    return "none" if w is None else "(" + ", ".join(f"0x{x:08x}" for x in w) + ")"


# ---- reduction -----------------------------------------------------------------------------------------------

def _variants(body):
    """Smaller programs: each node deleted, and each compound node replaced by one of its child lists."""
    for i, n in enumerate(body):
        yield body[:i] + body[i + 1:]
        kids = [x for x in n[1:] if isinstance(x, list)]
        for k in kids:
            yield body[:i] + k + body[i + 1:]
        for j, x in enumerate(n):
            if isinstance(x, list):
                for v in _variants(x):
                    m = list(n)
                    m[j] = v
                    yield body[:i] + [tuple(m)] + body[i + 1:]


def minimize(body, wave, structured=True):
    """Greedy reduction that keeps a mismatch."""
    changed = True
    while changed:
        changed = False
        for cand in _variants(body):
            status, _ = run_case("ps_cf_min", cand, wave, structured)
            if status == "mismatch":
                body, changed = cand, True
                break
    return body


# ---- suite ---------------------------------------------------------------------------------------------------

def program_for(seed, index, wave, depth):
    rng = random.Random(f"{seed}:{index}:{wave}")
    return generate(rng, wave, max_depth=depth, budget=rng.randrange(4, 13))


def run_suite(programs, seed, waves, structured=True, depth=3, limit=None):
    counts = {"pass": 0, "mismatch": 0, "skip": {}, "harness-error": 0}
    failures = []
    total = programs if limit is None else min(programs, limit)
    for i in range(total):
        for wave in waves:
            name = f"ps_cf_{i:04d}_w{wave}"
            body = program_for(seed, i, wave, depth)
            try:
                status, detail = run_case(name, body, wave, structured)
            except Exception as e:  # noqa: BLE001 - harness bugs must be loud
                counts["harness-error"] += 1
                failures.append((i, wave, f"harness error {type(e).__name__}: {e}"))
                continue
            if status == "match":
                counts["pass"] += 1
            elif status == "mismatch":
                counts["mismatch"] += 1
                failures.append((i, wave, detail))
            else:
                counts["skip"][detail] = counts["skip"].get(detail, 0) + 1
    return counts, failures


def self_test():
    """Harness canaries: a program must match, a sabotaged reference must be caught, the interpreter must
    reconverge after a divergent branch."""
    ok = True
    body = [("if", ("gt", "u32", 4, 0), True, [("probe", 5), ("op", "v_add_nc_u32 v9, 1, v9")]), ("probe", 6)]
    for wave in (32, 64):
        status, detail = run_case("ps_canary_ok", body, wave)
        if status != "match":
            print(f"canary FAILED (wave{wave}): hand-written if did not match ({status}: {detail})")
            ok = False
    # The reference must be able to disagree: drop the s_or that restores EXEC and the lanes > 4 never export.
    e = Emitter(32)
    e.program(body)
    e.items = [it for it in e.items if not (it[0] == "ins" and it[1].startswith("s_or_b32 exec_lo"))]
    with tempfile.TemporaryDirectory(prefix="cf-diff-") as tmp:
        p = os.path.join(tmp, "c.s")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(listing_text(e.assemble(), "ps_canary_bad"))
        if len(cf_ref.run(p, wave=32)["exports"]) != 4:     # 4 > lane: lanes 0-3 stay on
            print("canary FAILED: the reference did not drop lanes whose EXEC was never restored")
            ok = False
    # Divergent loop + vote: every lane must leave the loop with its own trip count.
    body = [("vloop", (0, 3, False), None, [("probe", 9), ("vccz", ("eq", "u32", 0, 1), "s_cbranch_vccz",
                                                    [("op", "s_add_u32 s10, s10, 1")])])]
    for wave in (32, 64):
        status, detail = run_case("ps_canary_loop", body, wave)
        if status != "match":
            print(f"canary FAILED (wave{wave}): divergent loop did not match ({status}: {detail})")
            ok = False
    if ok:
        print("canaries: match ok, sabotaged reference detected, divergent loop reconverges")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="FUZZ-8: control-flow/EXEC differential fuzzer.")
    ap.add_argument("--programs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wave", choices=("32", "64", "both"), default="both")
    ap.add_argument("--depth", type=int, default=3, help="maximum nesting depth")
    ap.add_argument("--dispatch-loop", action="store_true", help="translate in the dispatch-loop form")
    ap.add_argument("--program", type=int, help="only this program index (with --show / --minimize)")
    ap.add_argument("--show", action="store_true", help="print the listing of --program")
    ap.add_argument("--minimize", action="store_true", help="reduce a mismatching program before printing it")
    ap.add_argument("--spirv-val", action="store_true",
                    help="also run spirv-as and spirv-val on every translation (a failure counts as a mismatch)")
    ap.add_argument("--self-test", action="store_true")
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        ap.error("no arguments given; pass --help for the available options")
    a = ap.parse_args(argv)
    if a.show and a.program is None:
        ap.error("--show requires --program")
    if a.programs < 1:
        ap.error("--programs must be at least 1")
    if a.spirv_val:
        missing = [t for t in ("spirv-as", "spirv-val") if not shutil.which(t)]
        if missing:
            print(f"--spirv-val needs {' and '.join(missing)} on PATH")
            return 2
        global SPIRV_VAL
        SPIRV_VAL = True
    if a.self_test:
        return 0 if self_test() else 1
    waves = (32, 64) if a.wave == "both" else (int(a.wave),)
    structured = not a.dispatch_loop
    if a.program is not None:
        rc = 0
        matched = 0
        for wave in waves:
            body = program_for(a.seed, a.program, wave, a.depth)
            status, detail = run_case(f"ps_cf_{a.program:04d}_w{wave}", body, wave, structured)
            print(f"program {a.program} wave{wave}: {status} {detail}")
            matched += status == "match"
            if status == "mismatch":
                rc = 1
                if a.minimize:
                    body = minimize(body, wave, structured)
                    print("reduced:", run_case("ps_cf_min", body, wave, structured)[1])
            if a.show or a.minimize:
                print(listing_text(build(body, wave), f"ps_cf_{a.program:04d}_w{wave}"))
        if not rc and not matched:   # same order as the suite: a mismatch (1) outranks "nothing was compared" (2)
            print("no program ran: every one was skipped")
            return 2
        return rc
    counts, failures = run_suite(a.programs, a.seed, waves, structured, a.depth)
    skips = sum(counts["skip"].values())
    form = "dispatch loop" if a.dispatch_loop else "structured"
    print(f"{a.programs} programs x wave {'/'.join(map(str, waves))} (seed {a.seed}, {form}): "
          f"{counts['pass']} pass, {counts['mismatch']} mismatch, {skips} skipped, "
          f"{counts['harness-error']} harness errors")
    for reason, n in sorted(counts["skip"].items(), key=lambda kv: -kv[1]):
        print(f"  skip x{n}: {reason}")
    for i, wave, detail in failures[:10]:
        print(f"  program {i} wave{wave}: {detail}")
    if len(failures) > 10:
        print(f"  ... {len(failures) - 10} more")
    if counts["harness-error"]:
        return 2
    if counts["mismatch"]:
        return 1
    if not counts["pass"]:
        print("no program ran: every one was skipped")
        return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
