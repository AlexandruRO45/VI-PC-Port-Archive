"""Differential check for VEX AVX FP compare lowering (vcmpps, vcmppd, vcmpss, vcmpsd)."""
import ctypes
from fractions import Fraction
import re
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

from iced_x86 import Code as C, Decoder, EncodingKind, MemorySize, Mnemonic as M, OpKind as K, Register as R
import pefile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

PREDICATE_EXPORTS = (
    tuple(f"vcmpps_xmm_p{p}" for p in range(32))
    + tuple(f"vcmpps_ymm_p{p}" for p in range(32))
    + tuple(f"vcmppd_xmm_p{p}" for p in range(32))
    + tuple(f"vcmppd_ymm_p{p}" for p in range(32))
    + tuple(f"vcmpss_p{p}" for p in range(32))
    + tuple(f"vcmpsd_p{p}" for p in range(32))
)
MEM_EXPORTS = (
    "vcmpps_xmm_mem_q_op", "vcmpps_xmm_mem_s_op",
    "vcmpps_ymm_mem_q_op", "vcmpps_ymm_mem_s_op",
    "vcmppd_xmm_mem_q_op", "vcmppd_xmm_mem_s_op",
    "vcmppd_ymm_mem_q_op", "vcmppd_ymm_mem_s_op",
    "vcmpss_mem_q_op", "vcmpss_mem_s_op",
    "vcmpsd_mem_q_op", "vcmpsd_mem_s_op",
)
ALIAS_EXPORTS = (
    "vcmpps_ymm_alias_dst_src1_op",
    "vcmpps_ymm_alias_dst_src2_op",
    "vcmpps_ymm_alias_src1_src2_op",
    "vcmpps_ymm_alias_all_op",
    "vcmpps_xmm_alias_dst_src1_op",
    "vcmpps_xmm_alias_dst_src2_op",
    "vcmppd_xmm_alias_dst_src1_op",
    "vcmpss_alias_dst_src1_op",
    "vcmpss_alias_dst_src2_op",
    "vcmpss_alias_src1_src2_op",
    "vcmpsd_alias_dst_src1_op",
    "vcmpsd_alias_dst_src2_op",
)
MISC_EXPORTS = ("vcmpps_xmm_imm_highbits_op",)
EXPORTS = PREDICATE_EXPORTS + MEM_EXPORTS + ALIAS_EXPORTS + MISC_EXPORTS

# Export -> (kind, pred) where kind in {"ps_xmm", "ps_ymm", "pd_xmm", "pd_ymm", "ss", "sd"}
EXPORT_SPECS = {}
for p in range(32):
    EXPORT_SPECS[f"vcmpps_xmm_p{p}"] = ("ps_xmm", p)
    EXPORT_SPECS[f"vcmpps_ymm_p{p}"] = ("ps_ymm", p)
    EXPORT_SPECS[f"vcmppd_xmm_p{p}"] = ("pd_xmm", p)
    EXPORT_SPECS[f"vcmppd_ymm_p{p}"] = ("pd_ymm", p)
    EXPORT_SPECS[f"vcmpss_p{p}"] = ("ss", p)
    EXPORT_SPECS[f"vcmpsd_p{p}"] = ("sd", p)
EXPORT_SPECS.update({
    "vcmpps_xmm_mem_q_op": ("ps_xmm", 0x0c),
    "vcmpps_xmm_mem_s_op": ("ps_xmm", 0x01),
    "vcmpps_ymm_mem_q_op": ("ps_ymm", 0x1e),
    "vcmpps_ymm_mem_s_op": ("ps_ymm", 0x02),
    "vcmppd_xmm_mem_q_op": ("pd_xmm", 0x00),
    "vcmppd_xmm_mem_s_op": ("pd_xmm", 0x10),
    "vcmppd_ymm_mem_q_op": ("pd_ymm", 0x1d),
    "vcmppd_ymm_mem_s_op": ("pd_ymm", 0x0d),
    "vcmpss_mem_q_op": ("ss", 0x11),
    "vcmpss_mem_s_op": ("ss", 0x05),
    "vcmpsd_mem_q_op": ("sd", 0x04),
    "vcmpsd_mem_s_op": ("sd", 0x1b),
    "vcmpps_ymm_alias_dst_src1_op": ("ps_ymm", 0x01),
    "vcmpps_ymm_alias_dst_src2_op": ("ps_ymm", 0x02),
    "vcmpps_ymm_alias_src1_src2_op": ("ps_ymm", 0x00),
    "vcmpps_ymm_alias_all_op": ("ps_ymm", 0x04),
    "vcmpps_xmm_alias_dst_src1_op": ("ps_xmm", 0x01),
    "vcmpps_xmm_alias_dst_src2_op": ("ps_xmm", 0x02),
    "vcmppd_xmm_alias_dst_src1_op": ("pd_xmm", 0x0c),
    "vcmpss_alias_dst_src1_op": ("ss", 0x01),
    "vcmpss_alias_dst_src2_op": ("ss", 0x02),
    "vcmpss_alias_src1_src2_op": ("ss", 0x00),
    "vcmpsd_alias_dst_src1_op": ("sd", 0x0e),
    "vcmpsd_alias_dst_src2_op": ("sd", 0x11),
    "vcmpps_xmm_imm_highbits_op": ("ps_xmm", 0xe5 & 0x1f),
})

CPUID_HELPER = """#include <intrin.h>
extern "C" __declspec(dllexport) uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
extern "C" __declspec(dllexport) uint32_t host_exchange_mxcsr(uint32_t next) {
    const uint32_t current = _mm_getcsr();
    _mm_setcsr(next);
    return current;
}
"""

MUTATED_CONTROLS = (
    ("vcmpps_ymm_p12", "vcmpps_flipped_ord_unord_control",
     re.compile(r"avx_cmp_f32\(a, b, 12u\)"), "avx_cmp_f32(a, b, 4u)"),
    ("vcmpps_xmm_p0", "vcmpps_xmm_no_zero_hi_control",
     re.compile(r"memset\(wide0\.data\(\) \+ 16, 0, 16\);"), ""),
    ("vcmpss_p0", "vcmpss_no_zero_hi_control",
     re.compile(r"wide0 = \{\};"), ""),
    ("vcmpss_p1", "vcmpss_quiet_vs_sig_control",
     re.compile(r"avx_cmp_f32\(a, b, 1u\)"), "avx_cmp_f32(a, b, 17u)"),
)


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    lowered_by_name = {}
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in EXPORTS:
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            lowered_by_name[name] = code
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{code}\n}}'
            )
    for src_name, ctrl_name, pattern, replacement in MUTATED_CONTROLS:
        mutated, count = pattern.subn(replacement, lowered_by_name[src_name])
        if not count:
            raise AssertionError(f"mutated control {ctrl_name} changed nothing")
        parts.append(
            f'extern "C" __declspec(dllexport) uint64_t {ctrl_name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{mutated}\n}}'
        )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _make_synthetic_cmp_insn(
    mnemonic=M.VCMPPS,
    encoding=EncodingKind.VEX,
    op0=R.XMM0,
    op1=R.XMM1,
    op2_kind=K.REGISTER,
    op2=R.XMM2,
    op3_kind=K.IMMEDIATE8,
    imm=0,
    memory_size=MemorySize.UINT128,
):
    kinds = (K.REGISTER, K.REGISTER, op2_kind, op3_kind)
    regs = (op0, op1, op2, R.NONE)
    return SimpleNamespace(
        ip=0x1000,
        mnemonic=mnemonic,
        code=C.INVALID,
        encoding=encoding,
        op_count=4,
        op0_kind=kinds[0],
        op1_kind=kinds[1],
        op2_kind=kinds[2],
        op3_kind=kinds[3],
        op0_register=regs[0],
        op1_register=regs[1],
        op2_register=regs[2],
        memory_size=memory_size,
        segment_prefix=R.NONE,
        has_lock_prefix=False,
        immediate8=imm,
        op_kind=lambda n: kinds[n],
        op_register=lambda n: regs[n],
        immediate=lambda n: imm if n == 3 else 0,
    )


def check_controls() -> None:
    rejected_bytes = (
        ("c5f0c2da00", "VEX vcmpps xmm3, xmm1, xmm2, 0 (untracked dst)", ValueError),
        ("c5e0c2c200", "VEX vcmpps xmm0, xmm3, xmm2, 0 (untracked src1)", ValueError),
        ("c5f0c2c300", "VEX vcmpps xmm0, xmm1, xmm3, 0 (untracked src2)", ValueError),
        ("c5f4c2da00", "VEX vcmpps ymm3, ymm1, ymm2, 0 (untracked dst)", ValueError),
        ("c5e5c2c200", "VEX vcmppd ymm0, ymm3, ymm2, 0 (untracked src1)", ValueError),
        ("c5f2c2c300", "VEX vcmpss xmm0, xmm1, xmm3, 0 (untracked src2)", ValueError),
        ("c5f3c21a00", "VEX vcmpsd xmm3, xmm1, [rdx], 0 (untracked dst with mem)", ValueError),
        ("64c5f4c20200", "VEX vcmpps ymm0, ymm1, fs:[rdx], 0 (segment prefix)", ValueError),
        ("67c5f0c2050600000000", "VEX vcmpps xmm0, xmm1, [eip], 0 (EIP-relative)", ValueError),
        ("c5f2c2050700000000", "VEX vcmpss xmm0, xmm1, [rip], 0 (unbound RIP-relative)", KeyError),
        ("62f17428c2ca00", "EVEX vcmpps k1, ymm1, ymm2, 0", ValueError),
        ("62f1f508c2ca00", "EVEX vcmppd k1, xmm1, xmm2, 0", ValueError),
        ("62f17608c2ca00", "EVEX vcmpss k1, xmm1, xmm2, 0", ValueError),
        ("62f1f708c2ca00", "EVEX vcmpsd k1, xmm1, xmm2, 0", ValueError),
        ("0fc2c100", "Legacy cmpps xmm0, xmm1, 0", ValueError),
        ("660fc2c100", "Legacy cmppd xmm0, xmm1, 0", ValueError),
        ("f30fc2c100", "Legacy cmpss xmm0, xmm1, 0", ValueError),
        ("f20fc2c100", "Legacy cmpsd xmm0, xmm1, 0", ValueError),
    )
    for encoded, description, exc_type in rejected_bytes:
        body = list(Decoder(64, bytes.fromhex(encoded + "c3"), ip=0x1000))
        try:
            lower(body, [], {}, {}, {})
        except exc_type:
            continue
        raise AssertionError(f"expected {description} to raise {exc_type.__name__}")

    synthetic_bad = (
        (_make_synthetic_cmp_insn(encoding=EncodingKind.EVEX), "non-VEX EVEX encoding"),
        (_make_synthetic_cmp_insn(encoding=EncodingKind.LEGACY), "non-VEX legacy encoding"),
        (_make_synthetic_cmp_insn(op0=R.YMM0, op1=R.XMM1, op2=R.YMM2), "mixed YMM/XMM width"),
        (_make_synthetic_cmp_insn(op0=R.XMM0, op1=R.YMM1, op2=R.XMM2), "mixed XMM/YMM width"),
        (_make_synthetic_cmp_insn(mnemonic=M.VCMPSS, op0=R.YMM0, op1=R.YMM1, op2=R.YMM2), "YMM scalar vcmpss"),
        (_make_synthetic_cmp_insn(mnemonic=M.VCMPSD, op0=R.YMM0, op1=R.YMM1, op2=R.YMM2), "YMM scalar vcmpsd"),
        (_make_synthetic_cmp_insn(op0=R.XMM0, op1=R.XMM1, op2_kind=K.MEMORY, op2=R.NONE,
                                  memory_size=MemorySize.UINT256), "mismatched memory size on XMM vcmpps"),
        (_make_synthetic_cmp_insn(op0=R.YMM0, op1=R.YMM1, op2_kind=K.MEMORY, op2=R.NONE,
                                  memory_size=MemorySize.UINT128), "mismatched memory size on YMM vcmpps"),
        (_make_synthetic_cmp_insn(mnemonic=M.VCMPSS, op0=R.XMM0, op1=R.XMM1, op2_kind=K.MEMORY, op2=R.NONE,
                                  memory_size=MemorySize.UINT128), "mismatched memory size on vcmpss"),
        (_make_synthetic_cmp_insn(mnemonic=M.VCMPSD, op0=R.XMM0, op1=R.XMM1, op2_kind=K.MEMORY, op2=R.NONE,
                                  memory_size=MemorySize.UINT32), "mismatched memory size on vcmpsd"),
        (_make_synthetic_cmp_insn(op3_kind=K.REGISTER), "non-immediate8 op3"),
    )
    for insn, description in synthetic_bad:
        try:
            lower([insn], [], {}, {}, {})
        except ValueError:
            continue
        raise AssertionError(f"expected synthetic {description} to be rejected")


# --- Independent IEEE-754 binary32 / binary64 comparison reference model ---

class _FPCompareFormat:
    def __init__(self, ebits: int, mbits: int):
        self.ebits = ebits
        self.mbits = mbits
        self.bits = 1 + ebits + mbits
        self.bias = (1 << (ebits - 1)) - 1
        self.emax = (1 << ebits) - 1
        self.frac_mask = (1 << mbits) - 1
        self.quiet_bit = 1 << (mbits - 1)

    def unpack(self, raw: int):
        sign = (raw >> (self.bits - 1)) & 1
        exp = (raw >> self.mbits) & self.emax
        frac = raw & self.frac_mask
        return sign, exp, frac

    def is_nan(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == self.emax and frac != 0

    def is_snan(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == self.emax and frac != 0 and (frac & self.quiet_bit) == 0

    def is_inf(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == self.emax and frac == 0

    def flush_daz(self, raw: int, daz: bool) -> int:
        if not daz:
            return raw
        sign, exp, frac = self.unpack(raw)
        if exp == 0 and frac != 0:
            return sign << (self.bits - 1)
        return raw

    def to_value(self, raw: int):
        sign, exp, frac = self.unpack(raw)
        if exp == self.emax and frac == 0:
            return (1, sign)  # infinity tag
        if exp == 0:
            if frac == 0:
                return (0, Fraction(0, 1))
            mag = Fraction(frac, 1) * (Fraction(1, 2) ** (self.bias - 1 + self.mbits))
        else:
            mag = Fraction((1 << self.mbits) | frac, 1) * (Fraction(2, 1) ** (exp - self.bias - self.mbits))
        return (0, -mag if sign else mag)

    def classify(self, raw_a: int, raw_b: int, mxcsr: int):
        daz = bool(mxcsr & 0x40)
        a = self.flush_daz(raw_a, daz)
        b = self.flush_daz(raw_b, daz)
        if self.is_nan(a) or self.is_nan(b):
            snan = self.is_snan(a) or self.is_snan(b)
            return "UNORD", True, snan
        va, vb = self.to_value(a), self.to_value(b)
        if va[0] == 1 and vb[0] == 1:
            if va[1] == vb[1]:
                rel = "EQ"
            else:
                rel = "LT" if va[1] == 1 else "GT"
        elif va[0] == 1:
            rel = "LT" if va[1] == 1 else "GT"
        elif vb[0] == 1:
            rel = "GT" if vb[1] == 1 else "LT"
        else:
            rel = "LT" if va[1] < vb[1] else ("GT" if va[1] > vb[1] else "EQ")
        return rel, False, False

    def eval_pred(self, raw_a: int, raw_b: int, pred: int, mxcsr: int):
        p = pred & 0x1f
        sig = ((p & 3) in (1, 2)) != bool(p & 0x10)
        rel, any_nan, any_snan = self.classify(raw_a, raw_b, mxcsr)
        ie = any_nan if sig else any_snan
        table = {
            0x0: rel == "EQ",
            0x1: rel == "LT",
            0x2: rel in ("LT", "EQ"),
            0x3: rel == "UNORD",
            0x4: rel != "EQ",
            0x5: rel != "LT",
            0x6: rel in ("GT", "UNORD"),
            0x7: rel != "UNORD",
            0x8: rel in ("EQ", "UNORD"),
            0x9: rel in ("LT", "UNORD"),
            0xa: rel != "GT",
            0xb: False,
            0xc: rel in ("LT", "GT"),
            0xd: rel in ("EQ", "GT"),
            0xe: rel == "GT",
            0xf: True,
        }
        return table[p & 0x0f], ie


F32 = _FPCompareFormat(8, 23)
F64 = _FPCompareFormat(11, 52)


def ref_op(name: str, a_bytes: bytes, b_bytes: bytes, mxcsr: int):
    if name in ("vcmpps_ymm_alias_src1_src2_op", "vcmpps_ymm_alias_all_op", "vcmpss_alias_src1_src2_op"):
        b_bytes = a_bytes
    kind, pred = EXPORT_SPECS[name]
    is_f64 = kind.startswith("pd") or kind == "sd"
    fmt = F64 if is_f64 else F32
    pack_one = "<Q" if is_f64 else "<I"
    true_val = 0xffffffffffffffff if is_f64 else 0xffffffff
    elem_bytes = 8 if is_f64 else 4

    if kind in ("ss", "sd"):
        ra = struct.unpack_from(pack_one, a_bytes, 0)[0]
        rb = struct.unpack_from(pack_one, b_bytes, 0)[0]
        res, ie = fmt.eval_pred(ra, rb, pred, mxcsr)
        lane0 = struct.pack(pack_one, true_val if res else 0)
        return lane0 + a_bytes[elem_bytes:16] + b"\x00" * 16, ie

    nbytes = 32 if kind.endswith("ymm") else 16
    unpack_fmt = f"<{nbytes // elem_bytes}{'Q' if is_f64 else 'I'}"
    la = struct.unpack_from(unpack_fmt, a_bytes, 0)
    lb = struct.unpack_from(unpack_fmt, b_bytes, 0)
    out = []
    any_ie = False
    for ra, rb in zip(la, lb):
        res, ie = fmt.eval_pred(ra, rb, pred, mxcsr)
        out.append(true_val if res else 0)
        any_ie |= ie
    lo_and_hi = struct.pack(unpack_fmt, *out)
    if nbytes == 16:
        lo_and_hi += b"\x00" * 16
    return lo_and_hi, any_ie


def _guarded_buf(payload: bytes, align_offset: int = 0):
    guard = b"\xCC" * 32
    raw = ctypes.create_string_buffer(guard + payload + guard + bytes(32))
    base = ctypes.addressof(raw) + 32
    mis = base % 32
    aligned = base + ((32 - mis) % 32)
    ptr = aligned + align_offset
    ctypes.memmove(ptr - 32, guard, 32)
    ctypes.memmove(ptr, payload, len(payload))
    ctypes.memmove(ptr + len(payload), guard, 32)
    return raw, ptr


def check_guard_page(native, lifted, a32, b32) -> None:
    kernel32 = ctypes.windll.kernel32
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.VirtualProtect.restype = ctypes.c_int
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    kernel32.VirtualFree.restype = ctypes.c_int

    page_size = 4096
    mem = kernel32.VirtualAlloc(None, 2 * page_size, 0x3000, 0x04)
    if not mem:
        raise ctypes.WinError()
    try:
        old_prot = ctypes.c_uint32()
        if not kernel32.VirtualProtect(mem + page_size, page_size, 0x01, ctypes.byref(old_prot)):
            raise ctypes.WinError()
        _ra, pa = _guarded_buf(a32, 0)
        out_n = (ctypes.c_ubyte * 32)()
        out_l = (ctypes.c_ubyte * 32)()
        for name, width in (
            ("vcmpss_mem_q_op", 4),
            ("vcmpsd_mem_q_op", 8),
            ("vcmpps_xmm_mem_q_op", 16),
            ("vcmpps_ymm_mem_q_op", 32),
        ):
            edge_ptr = mem + page_size - width
            ctypes.memmove(edge_ptr, b32[:width], width)
            want, _ = ref_op(name, a32, b32, 0x1f80)
            for dll, out_buf in ((native, out_n), (lifted, out_l)):
                fn = getattr(dll, name)
                fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
                fn.restype = ctypes.c_uint64
                ctypes.memset(out_buf, 0xCC, 32)
                fn(pa, edge_ptr, ctypes.addressof(out_buf))
            if bytes(out_n) != want or bytes(out_l) != want:
                raise AssertionError(f"{name} guard-page check failed")
    finally:
        kernel32.VirtualFree(mem, 0, 0x8000)


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.argtypes, lifted.host_has_avx.restype = [], ctypes.c_uint32
    if not lifted.host_has_avx():
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))
    exchange = lifted.host_exchange_mxcsr
    exchange.argtypes, exchange.restype = [ctypes.c_uint32], ctypes.c_uint32

    def _f32(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0] for v in vals]
        return struct.pack("<8I", *words)

    def _f64(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<Q", struct.pack("<d", v))[0] for v in vals]
        return struct.pack("<4Q", *words)

    f32_qnan_pos, f32_qnan_neg = 0x7fc01234, 0xffc05678
    f32_snan_pos, f32_snan_neg = 0x7f800042, 0xffa00099
    f32_sub_pos, f32_sub_neg = 0x00400001, 0x80400001
    f32_min_norm = 0x00800000

    f32_pairs = (
        # <, ==, >, signed zeros, infinities across both 128-bit lanes + upper-lane preservation for vcmpss
        (_f32(-2.5, 1.5, 4.0, 0x00000000, 0x80000000, 0xff800000, 0x7f800000, f32_min_norm),
         _f32(1.0, 1.5, -3.0, 0x80000000, 0x00000000, 0x7f800000, 0x7f800000, 0xff800000)),
        # Subnormal vs zero/min_norm (DAZ sensitive) + upper-lane NaN in scalar src1 (must preserve without #I)
        (_f32(f32_sub_pos, f32_snan_pos, f32_qnan_neg, f32_sub_neg, f32_sub_neg, 0x00000000, 1.0, -1.0),
         _f32(0x00000000, 1.0, 2.0, 0x80000000, 0x80000000, f32_sub_pos, f32_min_norm, 0.0)),
        # QNaN vs finite (non-signaling quiet vs signaling #I)
        (_f32(f32_qnan_pos, 1.0, 2.0, f32_qnan_neg, 3.0, f32_qnan_pos, -0.0, 0.0),
         _f32(1.0, f32_qnan_neg, 2.0, f32_qnan_pos, -1.0, 0.0, 0.0, 1.0)),
        # SNaN vs finite / QNaN (must raise #I for both quiet and signaling predicates)
        (_f32(f32_snan_pos, 1.0, f32_qnan_pos, f32_snan_neg, 5.0, -5.0, 0.0, f32_snan_pos),
         _f32(1.0, f32_snan_neg, f32_snan_pos, f32_qnan_neg, 5.0, 2.0, -0.0, f32_snan_neg)),
    )

    f64_qnan_pos, f64_qnan_neg = 0x7ff8000012345678, 0xfff8000087654321
    f64_snan_pos, f64_snan_neg = 0x7ff0000000000042, 0xfff4000000000099
    f64_sub_pos, f64_sub_neg = 0x0008000000000001, 0x8008000000000001
    f64_min_norm = 0x0010000000000000

    f64_pairs = (
        (_f64(-2.5, 1.5, 0x0000000000000000, 0x7ff0000000000000),
         _f64(1.0, 1.5, 0x8000000000000000, 0xfff0000000000000)),
        (_f64(f64_sub_pos, f64_snan_pos, f64_sub_neg, f64_min_norm),
         _f64(0x0000000000000000, 1.0, 0x8000000000000000, f64_sub_pos)),
        (_f64(f64_qnan_pos, f64_snan_neg, 2.0, f64_qnan_neg),
         _f64(1.0, 2.0, f64_qnan_pos, f64_qnan_pos)),
        (_f64(f64_snan_pos, f64_qnan_pos, 3.0, -3.0),
         _f64(1.0, f64_snan_neg, 3.0, 1.0)),
    )

    mxcsr_modes = (
        0x1f80,  # default RNE, DAZ=0
        0x1fc0,  # RNE, DAZ=1
    )

    out_n = (ctypes.c_ubyte * 32)()
    out_l = (ctypes.c_ubyte * 32)()
    cases = 0

    def run_case(name: str, a_bytes: bytes, b_bytes: bytes, mxcsr: int, b_align: int = 0) -> None:
        nonlocal cases
        _ra, pa = _guarded_buf(a_bytes, 0)
        _rb, pb = _guarded_buf(b_bytes, b_align)
        want, want_ie = ref_op(name, a_bytes, b_bytes, mxcsr)
        statuses = []
        for dll, out_buf in ((native, out_n), (lifted, out_l)):
            fn = getattr(dll, name)
            fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
            fn.restype = ctypes.c_uint64
            ctypes.memset(out_buf, 0xCC, 32)
            previous = exchange(mxcsr)
            try:
                fn(pa, pb, ctypes.addressof(out_buf))
            finally:
                statuses.append(exchange(previous) & 0x3f)
        ie_n = bool(statuses[0] & 1)
        ie_l = bool(statuses[1] & 1)
        if bytes(out_n) != want or bytes(out_l) != want or ie_n != want_ie or ie_l != want_ie:
            raise AssertionError(
                f"{name} mxcsr={mxcsr:#x} align={b_align}: expected=({want.hex()}, ie={want_ie}) "
                f"native=({bytes(out_n).hex()}, ie={ie_n}) lifted=({bytes(out_l).hex()}, ie={ie_l})"
            )
        cases += 1

    for name in EXPORTS:
        kind, _ = EXPORT_SPECS[name]
        pairs = f64_pairs if (kind.startswith("pd") or kind == "sd") else f32_pairs
        for mxcsr in mxcsr_modes:
            for a_bytes, b_bytes in pairs:
                run_case(name, a_bytes, b_bytes, mxcsr, b_align=0)
        if "_mem_" in name:
            a_bytes, b_bytes = pairs[0]
            for off in (1, 3):
                run_case(name, a_bytes, b_bytes, 0x1f80, b_align=off)

    check_guard_page(native, lifted, f32_pairs[0][0], f32_pairs[0][1])

    def invoke_lifted(ctrl_name: str, a_bytes: bytes, b_bytes: bytes):
        _ra, pa = _guarded_buf(a_bytes, 0)
        _rb, pb = _guarded_buf(b_bytes, 0)
        fn = getattr(lifted, ctrl_name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = ctypes.c_uint64
        ctypes.memset(out_l, 0xCC, 32)
        prev = exchange(0x1f80)
        try:
            fn(pa, pb, ctypes.addressof(out_l))
        finally:
            status = exchange(prev) & 0x3f
        return bytes(out_l), bool(status & 1)

    # Mutated negative controls must mismatch expected output or MXCSR #I
    for ctrl_name, ref_name, pair in (
        ("vcmpps_flipped_ord_unord_control", "vcmpps_ymm_p12", f32_pairs[2]),
        ("vcmpps_xmm_no_zero_hi_control", "vcmpps_xmm_p0", f32_pairs[0]),
        ("vcmpss_no_zero_hi_control", "vcmpss_p0", f32_pairs[0]),
    ):
        want, _ = ref_op(ref_name, *pair, 0x1f80)
        got, _ = invoke_lifted(ctrl_name, *pair)
        if got == want:
            raise AssertionError(f"mutated control {ctrl_name} unexpectedly matched {want.hex()}")

    _, want_sig_ie = ref_op("vcmpss_p1", *f32_pairs[2], 0x1f80)
    _, got_sig_ie = invoke_lifted("vcmpss_quiet_vs_sig_control", *f32_pairs[2])
    if not want_sig_ie or got_sig_ie == want_sig_ie:
        raise AssertionError("mutated control vcmpss_quiet_vs_sig_control unexpectedly matched MXCSR #I")

    print(
        f"{cases} VEX AVX FP compare cases across {len(EXPORTS)} exports and {len(mxcsr_modes)} MXCSR modes "
        f"(plus guard-page bounds) matched native DLL and IEEE-754 reference; "
        f"{len(MUTATED_CONTROLS)} mutated controls mismatched"
    )


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
