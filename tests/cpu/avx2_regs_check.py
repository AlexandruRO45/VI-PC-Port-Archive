"""Differential checks, mutation controls and rejection controls for AVX2 register-domain AOT."""

import ctypes
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_avx2_aot import lift_code, translate
from cpu_leaf_aot import export_rva

MASK32 = 0xffffffff
MASK64 = 0xffffffffffffffff
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))
FILL = b"\x5A" * 32

EXPORTS = (
    "op_broadcasti128",
    "op_extracti128_lo",
    "op_extracti128_hi",
    "op_inserti128_lo",
    "op_inserti128_hi",
    "op_perm2i128_mix",
    "op_perm2i128_swap",
    "op_perm2i128_zero_lo",
    "op_perm2i128_zero_hi",
    "op_perm2i128_zero_both",
    "op_perm2i128_ignored",
    "op_pbroadcastb_xmm",
    "op_pbroadcastb_ymm",
    "op_pbroadcastw_xmm",
    "op_pbroadcastw_ymm",
    "op_pbroadcastd_xmm",
    "op_pbroadcastd_ymm",
    "op_pbroadcastq_xmm",
    "op_pbroadcastq_ymm",
    "op_permd",
    "op_permps",
    "op_permq",
    "op_permq_dup",
    "op_permpd",
    "op_pblendd_xmm",
    "op_pblendd_ymm",
    "op_pblendd_ymm_all",
    "op_pblendd_ymm_none",
    "op_psllvd_xmm",
    "op_psllvd_ymm",
    "op_psllvq_xmm",
    "op_psllvq_ymm",
    "op_psravd_xmm",
    "op_psravd_ymm",
    "op_psrlvd_xmm",
    "op_psrlvd_ymm",
    "op_psrlvq_xmm",
    "op_psrlvq_ymm",
    "op_zeroupper",
    "op_pipeline",
)


# --- Independent Python ISA reference model ---

def u32s(buf):
    return struct.unpack("<8I", buf)


def pack_u32s(vals):
    return struct.pack("<8I", *(v & MASK32 for v in vals))


def u64s(buf):
    return struct.unpack("<4Q", buf)


def pack_u64s(vals):
    return struct.pack("<4Q", *(v & MASK64 for v in vals))


def ref_vbroadcasti128(a):
    return a[:16] + a[:16]


def ref_vextracti128(a, imm8):
    lane = a[16:32] if (imm8 & 1) else a[:16]
    return lane + b"\x00" * 16


def ref_vinserti128(a, b, imm8):
    if imm8 & 1:
        return a[:16] + b[:16]
    return b[:16] + a[16:32]


def ref_vperm2i128(a, b, imm8):
    lanes = (a[:16], a[16:32], b[:16], b[16:32])
    def pick(nibble):
        return b"\x00" * 16 if (nibble & 8) else lanes[nibble & 3]
    return pick(imm8 & 0xf) + pick((imm8 >> 4) & 0xf)


def ref_vpbroadcast(a, width, wide):
    elem = a[:width]
    if wide:
        return elem * (32 // width)
    return elem * (16 // width) + b"\x00" * 16


def ref_vpermd(idx_buf, src_buf):
    idx = u32s(idx_buf)
    src = u32s(src_buf)
    return pack_u32s(src[i & 7] for i in idx)


def ref_vpermq(src_buf, imm8):
    q = u64s(src_buf)
    return pack_u64s(q[(imm8 >> (2 * i)) & 3] for i in range(4))


def ref_vpblendd(a, b, imm8, wide):
    da, db = u32s(a), u32s(b)
    lanes = 8 if wide else 4
    out = [db[i] if ((imm8 >> i) & 1) else da[i] for i in range(lanes)]
    if not wide:
        out.extend([0] * 4)
    return pack_u32s(out)


def ref_vpsllvd(a, b, wide):
    da, dc = u32s(a), u32s(b)
    lanes = 8 if wide else 4
    out = [0 if dc[i] >= 32 else ((da[i] << dc[i]) & MASK32) for i in range(lanes)]
    if not wide:
        out.extend([0] * 4)
    return pack_u32s(out)


def ref_vpsrlvd(a, b, wide):
    da, dc = u32s(a), u32s(b)
    lanes = 8 if wide else 4
    out = [0 if dc[i] >= 32 else (da[i] >> dc[i]) for i in range(lanes)]
    if not wide:
        out.extend([0] * 4)
    return pack_u32s(out)


def ref_vpsravd(a, b, wide):
    da, dc = u32s(a), u32s(b)
    lanes = 8 if wide else 4
    out = []
    for i in range(lanes):
        v = da[i]
        s = v - (1 << 32) if (v & 0x80000000) else v
        c = dc[i]
        out.append((-1 if s < 0 else 0) & MASK32 if c >= 32 else ((s >> c) & MASK32))
    if not wide:
        out.extend([0] * 4)
    return pack_u32s(out)


def ref_vpsllvq(a, b, wide):
    qa, qc = u64s(a), u64s(b)
    lanes = 4 if wide else 2
    out = [0 if qc[i] >= 64 else ((qa[i] << qc[i]) & MASK64) for i in range(lanes)]
    if not wide:
        out.extend([0] * 2)
    return pack_u64s(out)


def ref_vpsrlvq(a, b, wide):
    qa, qc = u64s(a), u64s(b)
    lanes = 4 if wide else 2
    out = [0 if qc[i] >= 64 else (qa[i] >> qc[i]) for i in range(lanes)]
    if not wide:
        out.extend([0] * 2)
    return pack_u64s(out)


def ref_pipeline(a, b):
    ymm0 = ref_vbroadcasti128(a)
    ymm1 = b
    ymm2 = ref_vpbroadcast(ymm1, 4, True)
    ymm3 = ref_vperm2i128(ymm0, ymm1, 0x21)
    ymm4 = ref_vpermd(ymm1, ymm3)
    ymm5 = ref_vpblendd(ymm4, ymm2, 0x5A, True)
    ymm5 = ref_vpsravd(ymm5, ymm1, True)
    ymm3 = ref_vextracti128(ymm5, 1)
    ymm5 = ref_vinserti128(ymm5, ymm3, 0)
    ymm5 = ref_vpermq(ymm5, 0xD8)
    return ymm5


def reference(name, a, b, altered=False):
    if name == "op_broadcasti128":
        return ref_vbroadcasti128(a)
    if name == "op_extracti128_lo":
        return ref_vextracti128(a, 0xFE)
    if name == "op_extracti128_hi":
        return ref_vextracti128(a, 0xFF)
    if name == "op_inserti128_lo":
        return ref_vinserti128(a, b, 0xFE)
    if name == "op_inserti128_hi":
        return ref_vinserti128(a, b, 0xFF)
    if name == "op_perm2i128_mix":
        return ref_vperm2i128(a, b, 0x31 if altered else 0x20)
    if name == "op_perm2i128_swap":
        return ref_vperm2i128(a, b, 0x03)
    if name == "op_perm2i128_zero_lo":
        return ref_vperm2i128(a, b, 0x28)
    if name == "op_perm2i128_zero_hi":
        return ref_vperm2i128(a, b, 0x81)
    if name == "op_perm2i128_zero_both":
        return ref_vperm2i128(a, b, 0x88)
    if name == "op_perm2i128_ignored":
        return ref_vperm2i128(a, b, 0x45)
    if name == "op_pbroadcastb_xmm":
        return ref_vpbroadcast(a, 1, False)
    if name == "op_pbroadcastb_ymm":
        return ref_vpbroadcast(a, 1, True)
    if name == "op_pbroadcastw_xmm":
        return ref_vpbroadcast(a, 2, False)
    if name == "op_pbroadcastw_ymm":
        return ref_vpbroadcast(a, 2, True)
    if name == "op_pbroadcastd_xmm":
        return ref_vpbroadcast(a, 4, False)
    if name == "op_pbroadcastd_ymm":
        return ref_vpbroadcast(a, 4, True)
    if name == "op_pbroadcastq_xmm":
        return ref_vpbroadcast(a, 8, False)
    if name == "op_pbroadcastq_ymm":
        return ref_vpbroadcast(a, 8, True)
    if name in ("op_permd", "op_permps"):
        return ref_vpermd(a, b)
    if name == "op_permq":
        return ref_vpermq(a, 0x4E if altered else 0x1B)
    if name == "op_permq_dup":
        return ref_vpermq(a, 0xB1)
    if name == "op_permpd":
        return ref_vpermq(a, 0x93)
    if name == "op_pblendd_xmm":
        return ref_vpblendd(a, b, 0xFA, False)
    if name == "op_pblendd_ymm":
        return ref_vpblendd(a, b, 0x55 if altered else 0xAA, True)
    if name == "op_pblendd_ymm_all":
        return ref_vpblendd(a, b, 0xFF, True)
    if name == "op_pblendd_ymm_none":
        return ref_vpblendd(a, b, 0x00, True)
    if name == "op_psllvd_xmm":
        return ref_vpsllvd(a, b, False)
    if name == "op_psllvd_ymm":
        return ref_vpsllvd(a, b, True)
    if name == "op_psllvq_xmm":
        return ref_vpsllvq(a, b, False)
    if name == "op_psllvq_ymm":
        return ref_vpsllvq(a, b, True)
    if name == "op_psravd_xmm":
        return ref_vpsravd(a, b, False)
    if name == "op_psravd_ymm":
        return ref_vpsravd(a, b, True)
    if name == "op_psrlvd_xmm":
        return ref_vpsrlvd(a, b, False)
    if name == "op_psrlvd_ymm":
        return ref_vpsrlvd(a, b, True)
    if name == "op_psrlvq_xmm":
        return ref_vpsrlvq(a, b, False)
    if name == "op_psrlvq_ymm":
        return ref_vpsrlvq(a, b, True)
    if name == "op_zeroupper":
        return a[:16] + b"\x00" * 16
    if name == "op_pipeline":
        return ref_pipeline(a, b)
    raise ValueError(f"unknown export {name}")


# --- Input vectors ---

def general_pairs():
    rng = random.Random(2026)
    distinct_a = bytes(range(0x01, 0x21))
    distinct_b = bytes(range(0x81, 0xA1))
    # IEEE-754 float32 special bit patterns: +0, -0, subnormal, +1, -1, +inf, sNaN, qNaN.
    float32_specials = pack_u32s((
        0x00000000, 0x80000000, 0x00000001, 0x3F800000,
        0xBF800000, 0x7F800000, 0x7F800001, 0x7FC00001,
    ))
    # IEEE-754 float64 special bit patterns: -0.0, sNaN, qNaN, subnormal.
    float64_specials = pack_u64s((
        0x8000000000000000, 0x7FF0000000000001, 0x7FF8000000000001, 0x0000000000000001,
    ))
    pairs = [
        (distinct_a, distinct_b),
        (b"\x00" * 32, b"\xFF" * 32),
        (b"\xFF" * 32, b"\x00" * 32),
        (float32_specials, distinct_b),
        (float64_specials, distinct_a),
        (pack_u32s((0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF, 0, 1, 0x12345678, 0xFEDCBA98)),
         pack_u32s((0, 7, 3, 4, 0xFFFFFFF8, 0x80000007, 0x00000005, 0xDEAD0002))),
    ]
    for _ in range(10):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def perm32_pairs():
    """Index/data pairs for vpermd/vpermps covering boundary indices 0..7 and high-bit noise."""
    data_tagged = pack_u32s(0x11000000 + i * 0x1111111 for i in range(8))
    data_fp = pack_u32s((
        0x00000000, 0x80000000, 0x00000001, 0x7F7FFFFF,
        0xFF7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00001,
    ))
    index_vecs = [
        pack_u32s((0, 1, 2, 3, 4, 5, 6, 7)),
        pack_u32s((7, 6, 5, 4, 3, 2, 1, 0)),
        pack_u32s((0, 0, 7, 7, 3, 4, 4, 3)),
        pack_u32s((0xFFFFFFF8 | i for i in (7, 0, 5, 2, 6, 1, 4, 3))),
        pack_u32s((0x80000000, 0x80000007, 0x12345678, 0x1234567F, 8, 15, 16, 31)),
    ]
    return [(idx, data) for idx in index_vecs for data in (data_tagged, data_fp)]


def shift32_pairs():
    """Values and boundary shift counts (0, 1, 15, 31, 32, 33, 63, 64, >64, max) for 32-bit shifts."""
    values = [
        pack_u32s((0x80000001, 0x7FFFFFFF, 0xFFFFFFFF, 0x00000001,
                   0x80000000, 0x00000000, 0xA5A55A5A, 0x12345678)),
        pack_u32s((0xFFFFFFFE, 0x7FFFFFFE, 0x8000FFFF, 0x00010000,
                   0xF000000F, 0x0F0000F0, 0x55555555, 0xAAAAAAAA)),
    ]
    counts = [
        pack_u32s((0, 1, 15, 31, 32, 33, 63, 64)),
        pack_u32s((65, 128, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 31, 32, 0)),
        pack_u32s((32, 32, 32, 32, 31, 31, 31, 31)),
        pack_u32s((0, 0, 1, 1, 64, 64, 0x10000001, 0xFFFFFFE0)),
    ]
    return [(v, c) for v in values for c in counts]


def shift64_pairs():
    """Values and boundary shift counts (0, 1, 31, 32, 63, 64, 65, >64, max) for 64-bit shifts."""
    values = [
        pack_u64s((0x8000000000000001, 0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0123456789ABCDEF)),
        pack_u64s((0x0000000000000001, 0x8000000000000000, 0xF0F0F0F00F0F0F0F, 0x0000000000000000)),
    ]
    counts = [
        pack_u64s((0, 1, 31, 32)),
        pack_u64s((63, 64, 65, 128)),
        pack_u64s((0x100000001, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF)),
        pack_u64s((64, 63, 0, 0x4000000000000000)),
    ]
    return [(v, c) for v in values for c in counts]


def inputs_for(name):
    if name in ("op_permd", "op_permps"):
        return perm32_pairs() + general_pairs()
    if name in ("op_psllvd_xmm", "op_psllvd_ymm", "op_psravd_xmm", "op_psravd_ymm",
                "op_psrlvd_xmm", "op_psrlvd_ymm"):
        return shift32_pairs() + general_pairs()
    if name in ("op_psllvq_xmm", "op_psllvq_ymm", "op_psrlvq_xmm", "op_psrlvq_ymm"):
        return shift64_pairs() + general_pairs()
    return general_pairs()


# --- Execution & comparison ---

def invoke(fn, a, b):
    buf_a = ctypes.create_string_buffer(a, 32)
    buf_b = ctypes.create_string_buffer(b, 32)
    guarded = ctypes.create_string_buffer(GUARD_PRE + FILL + GUARD_POST, 64)
    out_ptr = ctypes.addressof(guarded) + len(GUARD_PRE)
    fn(ctypes.addressof(buf_a), ctypes.addressof(buf_b), out_ptr)
    raw = guarded.raw
    if raw[:16] != GUARD_PRE or raw[48:] != GUARD_POST:
        raise AssertionError("guard bytes corrupted around output buffer")
    return raw[16:48]


def assert_equal_output(label, name, idx, got, want):
    if got != want:
        raise AssertionError(
            f"{label} {name} case {idx}: got {got.hex()} != expected {want.hex()}")


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        funcs[name] = fn
    return funcs


def check_differential(native_path, lifted_path, altered_path):
    native = bind_exports(native_path)
    lifted = bind_exports(lifted_path, "lifted_")
    altered = bind_exports(altered_path, "lifted_")

    total = 0
    for name in EXPORTS:
        for idx, (a, b) in enumerate(inputs_for(name)):
            want = reference(name, a, b, altered=False)
            got_native = invoke(native[name], a, b)
            got_lifted = invoke(lifted[name], a, b)
            assert_equal_output("native", name, idx, got_native, want)
            assert_equal_output("lifted", name, idx, got_lifted, want)
            total += 1

    # Altered build negative control: must match altered reference and fail unmodified reference.
    altered_exports = ("op_perm2i128_mix", "op_permq", "op_pblendd_ymm")
    changed = {}
    for name in altered_exports:
        diffs = 0
        for idx, (a, b) in enumerate(inputs_for(name)):
            want_alt = reference(name, a, b, altered=True)
            want_orig = reference(name, a, b, altered=False)
            got_alt = invoke(altered[name], a, b)
            assert_equal_output("altered", name, idx, got_alt, want_alt)
            if got_alt != want_orig:
                diffs += 1
        if not diffs:
            raise AssertionError(f"altered control {name} did not diverge from original")
        changed[name] = diffs

    # Explicit comparator mutation control: flipping one bit must fail assert_equal_output.
    sample_a, sample_b = general_pairs()[0]
    expected = reference("op_pipeline", sample_a, sample_b)
    mutated = bytearray(expected)
    mutated[17] ^= 0x01
    try:
        assert_equal_output("mutation", "op_pipeline", 0, bytes(mutated), expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")

    print(f"{total} native/lifted/reference AVX2 cases across {len(EXPORTS)} exports matched bit-for-bit; "
          f"altered controls diverged on {changed}; output mutation control caught")


# --- Translator rejection controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    """Verify every memory-domain AVX2 op, memory-operand register-domain op, and invalid case rejects."""
    ld_ymm0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))
    ld_ymm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX))
    ld_ymm2 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM2, MemoryOperand(R.RDX))
    ret_ins = Instruction.create(Code.RETNQ)

    # 1) All 10 memory-domain AVX2 forms must be rejected explicitly.
    memory_domain_cases = [
        ("vgatherdpd", Instruction.create_reg_mem_reg(
            Code.VEX_VGATHERDPD_YMM_VM32X_YMM, R.YMM0, MemoryOperand(R.RCX, R.XMM1, 8), R.YMM2)),
        ("vgatherdps", Instruction.create_reg_mem_reg(
            Code.VEX_VGATHERDPS_YMM_VM32Y_YMM, R.YMM0, MemoryOperand(R.RCX, R.YMM1, 4), R.YMM2)),
        ("vgatherqpd", Instruction.create_reg_mem_reg(
            Code.VEX_VGATHERQPD_YMM_VM64Y_YMM, R.YMM0, MemoryOperand(R.RCX, R.YMM1, 8), R.YMM2)),
        ("vgatherqps", Instruction.create_reg_mem_reg(
            Code.VEX_VGATHERQPS_XMM_VM64Y_XMM, R.XMM0, MemoryOperand(R.RCX, R.YMM1, 4), R.XMM2)),
        ("vpgatherdd", Instruction.create_reg_mem_reg(
            Code.VEX_VPGATHERDD_YMM_VM32Y_YMM, R.YMM0, MemoryOperand(R.RCX, R.YMM1, 4), R.YMM2)),
        ("vpgatherdq", Instruction.create_reg_mem_reg(
            Code.VEX_VPGATHERDQ_YMM_VM32X_YMM, R.YMM0, MemoryOperand(R.RCX, R.XMM1, 8), R.YMM2)),
        ("vpgatherqd", Instruction.create_reg_mem_reg(
            Code.VEX_VPGATHERQD_XMM_VM64Y_XMM, R.XMM0, MemoryOperand(R.RCX, R.YMM1, 4), R.XMM2)),
        ("vpgatherqq", Instruction.create_reg_mem_reg(
            Code.VEX_VPGATHERQQ_YMM_VM64Y_YMM, R.YMM0, MemoryOperand(R.RCX, R.YMM1, 8), R.YMM2)),
        ("vpmaskmovd_load", Instruction.create_reg_reg_mem(
            Code.VEX_VPMASKMOVD_YMM_YMM_M256, R.YMM0, R.YMM1, MemoryOperand(R.RCX))),
        ("vpmaskmovd_store", Instruction.create_mem_reg_reg(
            Code.VEX_VPMASKMOVD_M256_YMM_YMM, MemoryOperand(R.R8), R.YMM1, R.YMM0)),
        ("vpmaskmovq_load", Instruction.create_reg_reg_mem(
            Code.VEX_VPMASKMOVQ_YMM_YMM_M256, R.YMM0, R.YMM1, MemoryOperand(R.RCX))),
        ("vpmaskmovq_store", Instruction.create_mem_reg_reg(
            Code.VEX_VPMASKMOVQ_M256_YMM_YMM, MemoryOperand(R.R8), R.YMM1, R.YMM0)),
    ]

    # 2) Memory-operand encodings of all 17 register-only AVX2 forms must be rejected explicitly.
    reg_only_mem_cases = [
        ("vextracti128_mem", Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8, MemoryOperand(R.R8), R.YMM0, 1)),
        ("vinserti128_mem", Instruction.create_reg_reg_mem_u32(
            Code.VEX_VINSERTI128_YMM_YMM_XMMM128_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX), 1)),
        ("vperm2i128_mem", Instruction.create_reg_reg_mem_u32(
            Code.VEX_VPERM2I128_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX), 0x20)),
        ("vpbroadcastb_mem", Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTB_YMM_XMMM8, R.YMM0, MemoryOperand(R.RCX))),
        ("vpbroadcastw_mem", Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTW_YMM_XMMM16, R.YMM0, MemoryOperand(R.RCX))),
        ("vpbroadcastd_mem", Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTD_YMM_XMMM32, R.YMM0, MemoryOperand(R.RCX))),
        ("vpbroadcastq_mem", Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTQ_YMM_XMMM64, R.YMM0, MemoryOperand(R.RCX))),
        ("vpermd_mem", Instruction.create_reg_reg_mem(
            Code.VEX_VPERMD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX))),
        ("vpermpd_mem", Instruction.create_reg_mem_u32(
            Code.VEX_VPERMPD_YMM_YMMM256_IMM8, R.YMM0, MemoryOperand(R.RCX), 0x1B)),
        ("vpermps_mem", Instruction.create_reg_reg_mem(
            Code.VEX_VPERMPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX))),
        ("vpermq_mem", Instruction.create_reg_mem_u32(
            Code.VEX_VPERMQ_YMM_YMMM256_IMM8, R.YMM0, MemoryOperand(R.RCX), 0x1B)),
        ("vpblendd_mem", Instruction.create_reg_reg_mem_u32(
            Code.VEX_VPBLENDD_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX), 0xAA)),
        ("vpsllvd_mem", Instruction.create_reg_reg_mem(
            Code.VEX_VPSLLVD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX))),
        ("vpsllvq_mem", Instruction.create_reg_reg_mem(
            Code.VEX_VPSLLVQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX))),
        ("vpsravd_mem", Instruction.create_reg_reg_mem(
            Code.VEX_VPSRAVD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX))),
        ("vpsrlvd_mem", Instruction.create_reg_reg_mem(
            Code.VEX_VPSRLVD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX))),
        ("vpsrlvq_mem", Instruction.create_reg_reg_mem(
            Code.VEX_VPSRLVQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX))),
    ]

    # 3) Out-of-scope instructions.
    out_of_scope_cases = [
        ("vpaddd", Instruction.create_reg_reg_reg(Code.VEX_VPADDD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
        ("vperm2f128", Instruction.create_reg_reg_reg_u32(
            Code.VEX_VPERM2F128_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, R.YMM2, 0x20)),
        ("vpxor", Instruction.create_reg_reg_reg(Code.VEX_VPXOR_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
        ("vmovdqa", Instruction.create_reg_mem(Code.VEX_VMOVDQA_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))),
        ("addps", Instruction.create_reg_reg(Code.ADDPS_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("add_gpr", Instruction.create_reg_reg(Code.ADD_R32_RM32, R.EAX, R.ECX)),
        ("ud2", Instruction.create(Code.UD2)),
    ]

    # 4) ABI, register-state and addressing violations.
    seg_ins = Instruction.create_reg_mem(Code.VEX_VBROADCASTI128_YMM_M128, R.YMM0, MemoryOperand(R.RCX))
    seg_ins.segment_prefix = R.GS
    evex_ins = Instruction.create_reg_reg_reg(
        Code.EVEX_VPERMD_YMM_K1Z_YMM_YMMM256B32, R.YMM0, R.YMM1, R.YMM2)
    evex_ins.op_mask = R.K1
    other_cases = [
        ("evex_masked_vpermd", (ld_ymm0, ld_ymm1, ld_ymm2, evex_ins, ret_ins),
         "unsupported EVEX instruction"),
        ("nonvolatile_ymm6", (Instruction.create_reg_mem(
            Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM6, MemoryOperand(R.RCX)), ret_ins),
         "nonvolatile vector register"),
        ("nonvolatile_xmm6", (ld_ymm0, Instruction.create_reg_reg_u32(
            Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8, R.XMM6, R.YMM0, 0), ret_ins),
         "nonvolatile vector register"),
        ("undefined_ymm3", (ld_ymm0, Instruction.create_reg_reg_u32(
            Code.VEX_VPERMQ_YMM_YMMM256_IMM8, R.YMM0, R.YMM3, 0x1B), ret_ins),
         "undefined vector register ymm3"),
        ("rip_rel_mem", (Instruction.create_reg_mem(
            Code.VEX_VBROADCASTI128_YMM_M128, R.YMM0, MemoryOperand(R.RIP, displ=0x1010, displ_size=1)), ret_ins),
         "unsupported memory operand"),
        ("rsp_mem", (Instruction.create_reg_mem(
            Code.VEX_VBROADCASTI128_YMM_M128, R.YMM0, MemoryOperand(R.RSP)), ret_ins),
         "unsupported memory operand"),
        ("addr32_mem", (Instruction.create_reg_mem(
            Code.VEX_VBROADCASTI128_YMM_M128, R.YMM0, MemoryOperand(R.ECX)), ret_ins),
         "unsupported memory operand"),
        ("indexed_mem", (Instruction.create_reg_mem(
            Code.VEX_VBROADCASTI128_YMM_M128, R.YMM0, MemoryOperand(R.RCX, R.RDX, 1)), ret_ins),
         "unsupported memory operand"),
        ("non_arg_base_rax", (Instruction.create_reg_mem(
            Code.VEX_VBROADCASTI128_YMM_M128, R.YMM0, MemoryOperand(R.RAX)), ret_ins),
         "unsupported memory operand"),
        ("segment_prefix", (seg_ins, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_ymm0,), "missing terminal ret"),
        ("after_ret", (ret_ins, ld_ymm0), "unreachable instruction after ret"),
    ]

    all_unit = [
        *((lbl, (ld_ymm0, ld_ymm1, ld_ymm2, ins, ret_ins), "unsupported memory-domain AVX2 instruction")
          for lbl, ins in memory_domain_cases),
        *((lbl, (ld_ymm0, ld_ymm1, ld_ymm2, ins, ret_ins),
           "unsupported memory operand for register-domain AVX2 instruction")
          for lbl, ins in reg_only_mem_cases),
        *((lbl, (ld_ymm0, ld_ymm1, ld_ymm2, ins, ret_ins), "unsupported instruction")
          for lbl, ins in out_of_scope_cases),
        *other_cases,
    ]

    for label, seq, expected_reason in all_unit:
        raw = encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    # End-to-end PE patching controls on a copy of the built native DLL.
    # Each patch replaces op_pipeline's bytes and updates its .pdata EndAddress so the entire
    # runtime-function interval is cleanly decoded before hitting the expected rejection.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_pipeline")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12)
                       if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    pe_patches = [
        ("pe_vgatherdps", encode_seq(ld_ymm0, ld_ymm1, ld_ymm2, memory_domain_cases[1][1], ret_ins),
         "unsupported memory-domain AVX2 instruction"),
        ("pe_vpgatherdd", encode_seq(ld_ymm0, ld_ymm1, ld_ymm2, memory_domain_cases[4][1], ret_ins),
         "unsupported memory-domain AVX2 instruction"),
        ("pe_vpmaskmovd", encode_seq(ld_ymm0, ld_ymm1, memory_domain_cases[8][1], ret_ins),
         "unsupported memory-domain AVX2 instruction"),
        ("pe_vperm2i128_mem", encode_seq(ld_ymm0, ld_ymm1, reg_only_mem_cases[2][1], ret_ins),
         "unsupported memory operand for register-domain AVX2 instruction"),
        ("pe_vpaddd", encode_seq(ld_ymm0, ld_ymm1, ld_ymm2, out_of_scope_cases[0][1], ret_ins),
         "unsupported instruction"),
        ("pe_nonvolatile_ymm6", encode_seq(other_cases[1][1][0], ret_ins),
         "nonvolatile vector register"),
        ("pe_undef_ymm3", encode_seq(ld_ymm0, other_cases[3][1][1], ret_ins),
         "undefined vector register ymm3"),
        ("pe_missing_ret", encode_seq(ld_ymm0), "missing terminal ret"),
        ("pe_after_ret", encode_seq(ret_ins, ld_ymm0), "unreachable instruction after ret"),
    ]
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds op_pipeline ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_pipeline"])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

    print(f"{len(all_unit)} unit rejection controls (12 gather/maskmov, 17 register-domain memory, "
          f"7 out-of-scope, 12 ABI/state) and {len(pe_patches)} PE-patched rejection controls passed")


if __name__ == "__main__":
    check_differential(*sys.argv[1:4])
    check_rejections(sys.argv[1])
