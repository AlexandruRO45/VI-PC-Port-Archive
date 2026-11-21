"""Lift bounded Windows x64 leaf functions with register-domain AVX2 instructions to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry flat host pointers to caller-owned buffers. YMM0..YMM5 are
modeled as portable 256-bit value structs (four 64-bit lanes); every VEX write
to an XMM destination zeroes bits 128..255 of the corresponding YMM register.

Supported AVX2 instructions (18 forms from docs/coverage/x64_zen2.tsv group AVX2):
  - 128-bit lane broadcast/permute: vbroadcasti128 (m128), vextracti128,
    vinserti128, vperm2i128
  - Element broadcasts: vpbroadcastb, vpbroadcastw, vpbroadcastd, vpbroadcastq
  - Lane permutes: vpermd, vpermpd, vpermps, vpermq
  - Blend: vpblendd
  - Variable shifts: vpsllvd, vpsllvq, vpsravd, vpsrlvd, vpsrlvq

Transport/ABI glue: vmovdqu (128/256-bit flat load, store and register copy),
vzeroupper and terminal ret.

Memory-domain AVX2 forms (vgather*, vpgather*, vpmaskmov*), memory operands on
the 17 register-domain AVX2 forms, nonvolatile vector registers (XMM6..XMM15 /
YMM6..YMM15), EVEX encodings, branches and all other instructions are rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import Code, Decoder, Mnemonic as M, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = ({getattr(R, f"XMM{i}") for i in range(6, 16)}
                   | {getattr(R, f"YMM{i}") for i in range(6, 16)})

MEMORY_DOMAIN_AVX2 = {
    M.VGATHERDPD, M.VGATHERDPS, M.VGATHERQPD, M.VGATHERQPS,
    M.VPGATHERDD, M.VPGATHERDQ, M.VPGATHERQD, M.VPGATHERQQ,
    M.VPMASKMOVD, M.VPMASKMOVQ,
}

REGISTER_ONLY_AVX2 = {
    M.VEXTRACTI128, M.VINSERTI128, M.VPERM2I128,
    M.VPBROADCASTB, M.VPBROADCASTW, M.VPBROADCASTD, M.VPBROADCASTQ,
    M.VPERMD, M.VPERMPD, M.VPERMPS, M.VPERMQ,
    M.VPBLENDD,
    M.VPSLLVD, M.VPSLLVQ, M.VPSRAVD, M.VPSRLVD, M.VPSRLVQ,
}

ALLOWED_CODES = {
    Code.VEX_VMOVDQU_XMM_XMMM128,
    Code.VEX_VMOVDQU_XMMM128_XMM,
    Code.VEX_VMOVDQU_YMM_YMMM256,
    Code.VEX_VMOVDQU_YMMM256_YMM,
    Code.VEX_VZEROUPPER,
    Code.RETNQ,
    Code.VEX_VBROADCASTI128_YMM_M128,
    Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8,
    Code.VEX_VINSERTI128_YMM_YMM_XMMM128_IMM8,
    Code.VEX_VPERM2I128_YMM_YMM_YMMM256_IMM8,
    Code.VEX_VPBROADCASTB_XMM_XMMM8,
    Code.VEX_VPBROADCASTB_YMM_XMMM8,
    Code.VEX_VPBROADCASTW_XMM_XMMM16,
    Code.VEX_VPBROADCASTW_YMM_XMMM16,
    Code.VEX_VPBROADCASTD_XMM_XMMM32,
    Code.VEX_VPBROADCASTD_YMM_XMMM32,
    Code.VEX_VPBROADCASTQ_XMM_XMMM64,
    Code.VEX_VPBROADCASTQ_YMM_XMMM64,
    Code.VEX_VPERMD_YMM_YMM_YMMM256,
    Code.VEX_VPERMPD_YMM_YMMM256_IMM8,
    Code.VEX_VPERMPS_YMM_YMM_YMMM256,
    Code.VEX_VPERMQ_YMM_YMMM256_IMM8,
    Code.VEX_VPBLENDD_XMM_XMM_XMMM128_IMM8,
    Code.VEX_VPBLENDD_YMM_YMM_YMMM256_IMM8,
    Code.VEX_VPSLLVD_XMM_XMM_XMMM128,
    Code.VEX_VPSLLVD_YMM_YMM_YMMM256,
    Code.VEX_VPSLLVQ_XMM_XMM_XMMM128,
    Code.VEX_VPSLLVQ_YMM_YMM_YMMM256,
    Code.VEX_VPSRAVD_XMM_XMM_XMMM128,
    Code.VEX_VPSRAVD_YMM_YMM_YMMM256,
    Code.VEX_VPSRLVD_XMM_XMM_XMMM128,
    Code.VEX_VPSRLVD_YMM_YMM_YMMM256,
    Code.VEX_VPSRLVQ_XMM_XMM_XMMM128,
    Code.VEX_VPSRLVQ_YMM_YMM_YMMM256,
}

BROADCAST_HELPERS = {
    M.VPBROADCASTB: "op_vpbroadcastb",
    M.VPBROADCASTW: "op_vpbroadcastw",
    M.VPBROADCASTD: "op_vpbroadcastd",
    M.VPBROADCASTQ: "op_vpbroadcastq",
}

SHIFT_HELPERS = {
    M.VPSLLVD: "op_vpsllvd",
    M.VPSLLVQ: "op_vpsllvq",
    M.VPSRAVD: "op_vpsravd",
    M.VPSRLVD: "op_vpsrlvd",
    M.VPSRLVQ: "op_vpsrlvq",
}

PRELUDE = """#include <cstdint>
#include <cstring>

using word = std::uint64_t;

struct Vec256 {
    std::uint64_t q[4];
};

static Vec256 load128(word address) {
    Vec256 v{};
    std::memcpy(v.q, reinterpret_cast<const void*>(address), 16);
    return v;
}

static Vec256 load256(word address) {
    Vec256 v{};
    std::memcpy(v.q, reinterpret_cast<const void*>(address), 32);
    return v;
}

static void store128(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
}

static void store256(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
}

static Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

static std::uint32_t lane32(const Vec256& v, unsigned i) {
    return static_cast<std::uint32_t>(v.q[i >> 1] >> ((i & 1u) * 32u));
}

static void set_lane32(Vec256& v, unsigned i, std::uint32_t val) {
    const unsigned shift = (i & 1u) * 32u;
    v.q[i >> 1] = (v.q[i >> 1] & ~(0xffffffffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

static Vec256 op_vbroadcasti128(word address) {
    Vec256 v = load128(address);
    v.q[2] = v.q[0];
    v.q[3] = v.q[1];
    return v;
}

static Vec256 op_vextracti128(const Vec256& src, std::uint8_t imm8) {
    if (imm8 & 1u) return {{src.q[2], src.q[3], 0ull, 0ull}};
    return {{src.q[0], src.q[1], 0ull, 0ull}};
}

static Vec256 op_vinserti128(const Vec256& src1, const Vec256& src2, std::uint8_t imm8) {
    if (imm8 & 1u) return {{src1.q[0], src1.q[1], src2.q[0], src2.q[1]}};
    return {{src2.q[0], src2.q[1], src1.q[2], src1.q[3]}};
}

static void pick_lane128(const Vec256& src1, const Vec256& src2, unsigned sel,
                         std::uint64_t& lo, std::uint64_t& hi) {
    if (sel & 8u) {
        lo = 0ull;
        hi = 0ull;
        return;
    }
    switch (sel & 3u) {
        case 0u: lo = src1.q[0]; hi = src1.q[1]; break;
        case 1u: lo = src1.q[2]; hi = src1.q[3]; break;
        case 2u: lo = src2.q[0]; hi = src2.q[1]; break;
        default: lo = src2.q[2]; hi = src2.q[3]; break;
    }
}

static Vec256 op_vperm2i128(const Vec256& src1, const Vec256& src2, std::uint8_t imm8) {
    Vec256 out{};
    pick_lane128(src1, src2, imm8 & 0xfu, out.q[0], out.q[1]);
    pick_lane128(src1, src2, (imm8 >> 4) & 0xfu, out.q[2], out.q[3]);
    return out;
}

static Vec256 op_vpbroadcastb(const Vec256& src, bool wide) {
    const std::uint64_t b = src.q[0] & 0xffull;
    const std::uint64_t q = b * 0x0101010101010101ull;
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vpbroadcastw(const Vec256& src, bool wide) {
    const std::uint64_t w = src.q[0] & 0xffffull;
    const std::uint64_t q = w * 0x0001000100010001ull;
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vpbroadcastd(const Vec256& src, bool wide) {
    const std::uint64_t d = src.q[0] & 0xffffffffull;
    const std::uint64_t q = d | (d << 32);
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vpbroadcastq(const Vec256& src, bool wide) {
    const std::uint64_t q = src.q[0];
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vpermd(const Vec256& idx, const Vec256& src) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        set_lane32(out, i, lane32(src, lane32(idx, i) & 7u));
    }
    return out;
}

static Vec256 op_vpermq(const Vec256& src, std::uint8_t imm8) {
    return {{
        src.q[(imm8 >> 0) & 3u],
        src.q[(imm8 >> 2) & 3u],
        src.q[(imm8 >> 4) & 3u],
        src.q[(imm8 >> 6) & 3u],
    }};
}

static Vec256 op_vpblendd(const Vec256& src1, const Vec256& src2, std::uint8_t imm8, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        set_lane32(out, i, ((imm8 >> i) & 1u) ? lane32(src2, i) : lane32(src1, i));
    }
    return out;
}

static Vec256 op_vpsllvd(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint32_t c = lane32(src2, i);
        set_lane32(out, i, c >= 32u ? 0u : (lane32(src1, i) << c));
    }
    return out;
}

static Vec256 op_vpsrlvd(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint32_t c = lane32(src2, i);
        set_lane32(out, i, c >= 32u ? 0u : (lane32(src1, i) >> c));
    }
    return out;
}

static Vec256 op_vpsravd(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint32_t v = lane32(src1, i);
        const std::uint32_t c = lane32(src2, i);
        const bool neg = (v & 0x80000000u) != 0u;
        std::uint32_t r;
        if (c >= 32u) {
            r = neg ? 0xffffffffu : 0u;
        } else {
            r = (v >> c) | ((neg && c != 0u) ? (~0u << (32u - c)) : 0u);
        }
        set_lane32(out, i, r);
    }
    return out;
}

static Vec256 op_vpsllvq(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint64_t c = src2.q[i];
        out.q[i] = c >= 64ull ? 0ull : (src1.q[i] << c);
    }
    return out;
}

static Vec256 op_vpsrlvq(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint64_t c = src2.q[i];
        out.q[i] = c >= 64ull ? 0ull : (src1.q[i] >> c);
    }
    return out;
}
"""


def vec_index(register, ins):
    """Map XMM0..XMM5 or YMM0..YMM5 to (index, is_ymm), rejecting all others."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register], False
    if register in VOLATILE_YMM:
        return VOLATILE_YMM[register], True
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins):
    """Lower a flat [base + disp] pointer operand rooted in an argument GPR64."""
    base = ins.memory_base
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or ins.memory_index != R.NONE or not RegisterExt.is_gpr64(base)
            or base not in ARGUMENTS):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AVX2 leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used = set()
    defined = set()
    statements = []
    returned = False

    def read_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if idx not in defined:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used.add(idx)
        return f"ymm{idx}", wide

    def write_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        used.add(idx)
        return idx, f"ymm{idx}", wide

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if ins.op_mask != R.NONE or ins.zeroing_masking or ins.is_broadcast:
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        mnemonic = ins.mnemonic
        if mnemonic in MEMORY_DOMAIN_AVX2:
            raise ValueError(f"unsupported memory-domain AVX2 instruction at RVA {ins.ip:#x}: {ins}")
        if mnemonic in REGISTER_ONLY_AVX2 and any(
                ins.op_kind(i) == OpKind.MEMORY for i in range(ins.op_count)):
            raise ValueError(
                f"unsupported memory operand for register-domain AVX2 instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")
        elif mnemonic == M.VMOVDQU:
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_idx, dst, wide = write_vec(0, ins)
                addr = flat_address(ins)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src, wide = read_vec(1, ins)
                addr = flat_address(ins)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src, _ = read_vec(1, ins)
                dst_idx, dst, wide = write_vec(0, ins)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined.add(dst_idx)
            else:
                raise ValueError(f"unsupported vmovdqu operands at RVA {ins.ip:#x}: {ins}")
        elif mnemonic == M.VBROADCASTI128:
            dst_idx, dst, _ = write_vec(0, ins)
            addr = flat_address(ins)
            statements.append(f"{dst} = op_vbroadcasti128({addr});")
            defined.add(dst_idx)
        elif mnemonic == M.VEXTRACTI128:
            src, _ = read_vec(1, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vextracti128({src}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic == M.VINSERTI128:
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vinserti128({src1}, {src2}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic == M.VPERM2I128:
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vperm2i128({src1}, {src2}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic in BROADCAST_HELPERS:
            src, _ = read_vec(1, ins)
            dst_idx, dst, wide = write_vec(0, ins)
            helper = BROADCAST_HELPERS[mnemonic]
            statements.append(f"{dst} = {helper}({src}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in (M.VPERMD, M.VPERMPS):
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            statements.append(f"{dst} = op_vpermd({src1}, {src2});")
            defined.add(dst_idx)
        elif mnemonic in (M.VPERMQ, M.VPERMPD):
            src, _ = read_vec(1, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vpermq({src}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic == M.VPBLENDD:
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            dst_idx, dst, wide = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vpblendd({src1}, {src2}, {imm:#x}u, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in SHIFT_HELPERS:
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            dst_idx, dst, wide = write_vec(0, ins)
            helper = SHIFT_HELPERS[mnemonic]
            statements.append(f"{dst} = {helper}({src1}, {src2}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used)]
    body = [f"    {stmt}" for stmt in statements]
    lines = [
        f'extern "C" __declspec(dllexport) void lifted_{name}(word rcx, word rdx, word r8) {{',
        "    (void)rcx; (void)rdx; (void)r8;",
        *decls,
        *body,
        "}",
    ]
    return "\n".join(lines), len(instructions)


def lift(pe, name):
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw)
    return source, entry, count


def translate_pe(pe, names):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name) for name in names]
    source = PRELUDE + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image, names):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        source, meta = translate(args.image, args.exports.split(","))
        args.output.write_text(source, encoding="utf-8")
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
