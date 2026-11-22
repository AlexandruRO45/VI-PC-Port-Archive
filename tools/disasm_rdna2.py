#!/usr/bin/env python3
"""
AMD RDNA 2 (GFX1013 / Scarlett) Shader Disassembler & Inspector.

Decodes raw binary machine code (.bin) from Xbox Scarlett shaders,
prints human-readable assembly with byte offsets, raw hex words, register
tracking, and explanations mapped to the AMD RDNA 2 ISA manual.
"""

import argparse
from collections import Counter
import os
from pathlib import Path
import struct
import sys

# Register decode helpers
def decode_s_reg(reg_id: int) -> str:
    """Decode scalar register or scalar constant code."""
    if reg_id <= 105:
        return f"s{reg_id}"
    if reg_id == 106:
        return "vcc_lo"
    if reg_id == 107:
        return "vcc_hi"
    if 108 <= reg_id <= 123:
        return f"ttmp{reg_id - 108}"
    if reg_id == 124:
        return "m0"
    if reg_id == 125:
        return "null"
    if reg_id == 126:
        return "exec_lo"
    if reg_id == 127:
        return "exec_hi"
    if 128 <= reg_id <= 192:
        return str(reg_id - 128)
    if 193 <= reg_id <= 208:
        return str(-(reg_id - 192))
    if reg_id == 240:
        return "0.5"
    if reg_id == 241:
        return "-0.5"
    if reg_id == 242:
        return "1.0"
    if reg_id == 243:
        return "-1.0"
    if reg_id == 244:
        return "2.0"
    if reg_id == 245:
        return "-2.0"
    if reg_id == 246:
        return "4.0"
    if reg_id == 247:
        return "-4.0"
    if reg_id == 255:
        return "literal"
    return f"s{reg_id}"


def decode_s_reg64(reg_id: int) -> str:
    """Decode a 64-bit scalar operand as an SGPR pair or a named 64-bit special."""
    if reg_id <= 105 and reg_id % 2 == 0:
        return f"s[{reg_id}:{reg_id + 1}]"
    if reg_id == 106:
        return "vcc"
    if 108 <= reg_id <= 122 and (reg_id - 108) % 2 == 0:
        n = reg_id - 108
        return f"ttmp[{n}:{n + 1}]"
    if reg_id == 125:
        return "null"
    if reg_id == 126:
        return "exec"
    return decode_s_reg(reg_id)


def decode_v_reg(reg_id: int) -> str:
    """Decode a 9-bit VOP SRC0 operand: 0..255 scalar/constant, 256..511 VGPR."""
    if 256 <= reg_id <= 511:
        return f"v{reg_id - 256}"
    return decode_s_reg(reg_id)


# Instruction Opcode Tables
SOPP_OPS = {
    0x00: ("s_nop", "No operation (wait cycles)"),
    0x01: ("s_endpgm", "Terminate shader execution"),
    0x02: ("s_branch", "Unconditional branch"),
    0x04: ("s_cbranch_scc0", "Branch if SCC == 0"),
    0x05: ("s_cbranch_scc1", "Branch if SCC == 1"),
    0x06: ("s_cbranch_vccz", "Branch if VCC is zero"),
    0x07: ("s_cbranch_vccnz", "Branch if VCC is non-zero"),
    0x08: ("s_cbranch_execz", "Branch if EXEC is zero"),
    0x09: ("s_cbranch_execnz", "Branch if EXEC is non-zero"),
    0x0a: ("s_barrier", "Synchronize wavefronts in workgroup"),
    0x0c: ("s_waitcnt", "Wait for memory operations"),
    0x20: ("s_inst_prefetch", "Instruction cache prefetch"),
    0x21: ("s_clause", "Begin hard clause"),
}

SOP1_OPS = {
    0x03: ("s_mov_b32", "Scalar move 32-bit"),
    0x04: ("s_mov_b64", "Scalar move 64-bit"),
    0x05: ("s_cmov_b32", "Scalar conditional move 32-bit"),
    0x06: ("s_cmov_b64", "Scalar conditional move 64-bit"),
    0x07: ("s_not_b32", "Scalar bitwise NOT 32-bit"),
    0x08: ("s_not_b64", "Scalar bitwise NOT 64-bit"),
    0x09: ("s_wqm_b32", "Scalar whole-quad-mode 32-bit"),
    0x0a: ("s_wqm_b64", "Scalar whole-quad-mode 64-bit"),
    0x20: ("s_setpc_b64", "Set program counter 64-bit"),
    0x21: ("s_swappc_b64", "Swap program counter 64-bit"),
}

SOP2_OPS = {
    0x00: ("s_add_u32", "Scalar add unsigned 32-bit"),
    0x01: ("s_sub_u32", "Scalar sub unsigned 32-bit"),
    0x02: ("s_add_i32", "Scalar add signed 32-bit"),
    0x03: ("s_sub_i32", "Scalar sub signed 32-bit"),
    0x06: ("s_min_i32", "Scalar minimum signed 32-bit"),
    0x07: ("s_min_u32", "Scalar minimum unsigned 32-bit"),
    0x08: ("s_max_i32", "Scalar maximum signed 32-bit"),
    0x09: ("s_max_u32", "Scalar maximum unsigned 32-bit"),
    0x0a: ("s_cselect_b32", "Scalar conditional select 32-bit"),
    0x0e: ("s_and_b32", "Scalar bitwise AND 32-bit"),
    0x0f: ("s_and_b64", "Scalar bitwise AND 64-bit"),
    0x10: ("s_or_b32", "Scalar bitwise OR 32-bit"),
    0x11: ("s_or_b64", "Scalar bitwise OR 64-bit"),
    0x12: ("s_xor_b32", "Scalar bitwise XOR 32-bit"),
    0x13: ("s_xor_b64", "Scalar bitwise XOR 64-bit"),
    0x14: ("s_andn2_b32", "Scalar AND with NOT 32-bit"),
    0x15: ("s_andn2_b64", "Scalar AND with NOT 64-bit"),
    0x1e: ("s_lshl_b32", "Scalar logical shift left 32-bit"),
    0x1f: ("s_lshl_b64", "Scalar logical shift left 64-bit"),
    0x20: ("s_lshr_b32", "Scalar logical shift right 32-bit"),
    0x21: ("s_lshr_b64", "Scalar logical shift right 64-bit"),
    0x22: ("s_ashr_i32", "Scalar arithmetic shift right 32-bit"),
    0x26: ("s_mul_i32", "Scalar multiply signed 32-bit"),
}

SOPK_OPS = {
    0x00: ("s_movk_i32", "Scalar move 16-bit sign-extended immediate"),
    0x02: ("s_cmovk_i32", "Scalar conditional move immediate"),
    0x03: ("s_cmpk_eq_i32", "Scalar compare equal immediate"),
    0x04: ("s_cmpk_lg_i32", "Scalar compare not-equal immediate"),
    0x05: ("s_cmpk_gt_i32", "Scalar compare greater immediate"),
    0x0f: ("s_addk_i32", "Scalar add immediate"),
    0x10: ("s_mulk_i32", "Scalar multiply immediate"),
    0x17: ("s_waitcnt_vscnt", "Wait for vector store counter"),
    0x18: ("s_waitcnt_vmcnt", "Wait for vector memory counter"),
    0x19: ("s_waitcnt_expcnt", "Wait for export counter"),
    0x1a: ("s_waitcnt_lgkmcnt", "Wait for LDS/GDS/K/M counter"),
}

SOPC_OPS = {
    # GFX10 / RDNA 2 opcode numbers, as in LLVM SOPC_Real_gfx10 definitions
    0x00: ("s_cmp_eq_i32", "Scalar compare equal signed 32-bit"),
    0x01: ("s_cmp_lg_i32", "Scalar compare not-equal signed 32-bit"),
    0x02: ("s_cmp_gt_i32", "Scalar compare greater signed 32-bit"),
    0x03: ("s_cmp_ge_i32", "Scalar compare greater-or-equal signed 32-bit"),
    0x04: ("s_cmp_lt_i32", "Scalar compare less signed 32-bit"),
    0x05: ("s_cmp_le_i32", "Scalar compare less-or-equal signed 32-bit"),
    0x06: ("s_cmp_eq_u32", "Scalar compare equal unsigned 32-bit"),
    0x07: ("s_cmp_lg_u32", "Scalar compare not-equal unsigned 32-bit"),
    0x08: ("s_cmp_gt_u32", "Scalar compare greater unsigned 32-bit"),
    0x09: ("s_cmp_ge_u32", "Scalar compare greater-or-equal unsigned 32-bit"),
    0x0a: ("s_cmp_lt_u32", "Scalar compare less unsigned 32-bit"),
    0x0b: ("s_cmp_le_u32", "Scalar compare less-or-equal unsigned 32-bit"),
    0x0c: ("s_bitcmp0_b32", "Scalar test that bit src1 of src0 is zero"),
    0x0d: ("s_bitcmp1_b32", "Scalar test that bit src1 of src0 is one"),
    0x0e: ("s_bitcmp0_b64", "Scalar test that bit src1 of 64-bit src0 is zero"),
    0x0f: ("s_bitcmp1_b64", "Scalar test that bit src1 of 64-bit src0 is one"),
    0x12: ("s_cmp_eq_u64", "Scalar compare equal unsigned 64-bit"),
    0x13: ("s_cmp_lg_u64", "Scalar compare not-equal unsigned 64-bit"),
}

SMEM_OPS = {
    0x00: "s_load_dword",
    0x01: "s_load_dwordx2",
    0x02: "s_load_dwordx4",
    0x03: "s_load_dwordx8",
    0x04: "s_load_dwordx16",
    0x08: "s_buffer_load_dword",
    0x09: "s_buffer_load_dwordx2",
    0x0a: "s_buffer_load_dwordx4",
    0x0b: "s_buffer_load_dwordx8",
    0x0c: "s_buffer_load_dwordx16",
}

SMEM_WIDTHS = {
    0x00: (1, 2), 0x01: (2, 2), 0x02: (4, 2), 0x03: (8, 2), 0x04: (16, 2),
    0x08: (1, 4), 0x09: (2, 4), 0x0a: (4, 4), 0x0b: (8, 4), 0x0c: (16, 4),
}

MIMG_OPS = {
    0x00: ("image_load", False, 0),
    0x01: ("image_load_mip", False, 1),
    0x08: ("image_store", False, 0),
    0x09: ("image_store_mip", False, 1),
    0x0e: ("image_get_resinfo", False, 0),
    0x20: ("image_sample", True, 0),
    0x21: ("image_sample_cl", True, 1),
    0x22: ("image_sample_d", True, 0),
    0x24: ("image_sample_l", True, 1),
    0x25: ("image_sample_b", True, 1),
    0x26: ("image_sample_b_cl", True, 2),
    0x27: ("image_sample_lz", True, 0),
    0x28: ("image_sample_c", True, 1),
    0x2c: ("image_sample_c_l", True, 2),
    0x2d: ("image_sample_c_b", True, 2),
    0x2f: ("image_sample_c_lz", True, 1),
    0x30: ("image_sample_o", True, 1),
    0x34: ("image_sample_l_o", True, 2),
    0x35: ("image_sample_b_o", True, 2),
    0x37: ("image_sample_lz_o", True, 1),
    0x40: ("image_gather4", True, 0),
    0x44: ("image_gather4_l", True, 1),
    0x47: ("image_gather4_lz", True, 0),
    0x60: ("image_get_lod", True, 0),
}

MIMG_DIMS = {
    0: ("SQ_RSRC_IMG_1D", 1, 2),
    1: ("SQ_RSRC_IMG_2D", 2, 4),
    2: ("SQ_RSRC_IMG_3D", 3, 6),
    3: ("SQ_RSRC_IMG_CUBE", 3, 4),
    4: ("SQ_RSRC_IMG_1D_ARRAY", 2, 2),
    5: ("SQ_RSRC_IMG_2D_ARRAY", 3, 4),
    6: ("SQ_RSRC_IMG_2D_MSAA", 3, 4),
    7: ("SQ_RSRC_IMG_2D_MSAA_ARRAY", 4, 4),
}

VOP1_OPS = {
    0x00: ("v_nop", "Vector no-op"),
    0x01: ("v_mov_b32_e32", "Vector copy 32-bit"),
    0x02: ("v_readfirstlane_b32", "Read first active lane into SGPR"),
    0x05: ("v_cvt_f32_i32_e32", "Convert int32 to float32"),
    0x06: ("v_cvt_f32_u32_e32", "Convert uint32 to float32"),
    0x07: ("v_cvt_u32_f32_e32", "Convert float32 to uint32"),
    0x08: ("v_cvt_i32_f32_e32", "Convert float32 to int32"),
    0x0a: ("v_cvt_f16_f32_e32", "Convert float32 to float16"),
    0x0b: ("v_cvt_f32_f16_e32", "Convert float16 to float32"),
    0x20: ("v_fract_f32_e32", "Fractional part float32"),
    0x21: ("v_trunc_f32_e32", "Truncate toward zero float32"),
    0x22: ("v_ceil_f32_e32", "Ceiling float32"),
    0x23: ("v_rndne_f32_e32", "Round to nearest even float32"),
    0x24: ("v_floor_f32_e32", "Floor float32"),
    0x25: ("v_exp_f32_e32", "Exponent 2^x float32"),
    0x27: ("v_log_f32_e32", "Logarithm log2(x) float32"),
    0x2a: ("v_rcp_f32_e32", "Reciprocal 1/x float32"),
    0x2e: ("v_rsq_f32_e32", "Reciprocal square root 1/sqrt(x)"),
    0x33: ("v_sqrt_f32_e32", "Square root sqrt(x) float32"),
    0x34: ("v_frexp_mant_f32_e32", "Vector extract float32 mantissa"),
    0x35: ("v_sin_f32_e32", "Sine float32"),
    0x36: ("v_cos_f32_e32", "Cosine float32"),
    0x37: ("v_not_b32_e32", "Vector bitwise NOT"),
    0x38: ("v_bfrev_b32_e32", "Vector bit-reverse"),
}

VOPC_OPS = {
    0x00: ("v_cmp_f_f32_e32", "Vector compare v_cmp_f_f32"),
    0x01: ("v_cmp_lt_f32_e32", "Vector compare v_cmp_lt_f32"),
    0x02: ("v_cmp_eq_f32_e32", "Vector compare v_cmp_eq_f32"),
    0x03: ("v_cmp_le_f32_e32", "Vector compare v_cmp_le_f32"),
    0x04: ("v_cmp_gt_f32_e32", "Vector compare v_cmp_gt_f32"),
    0x05: ("v_cmp_lg_f32_e32", "Vector compare v_cmp_lg_f32"),
    0x06: ("v_cmp_ge_f32_e32", "Vector compare v_cmp_ge_f32"),
    0x07: ("v_cmp_o_f32_e32", "Vector compare v_cmp_o_f32"),
    0x08: ("v_cmp_u_f32_e32", "Vector compare v_cmp_u_f32"),
    0x09: ("v_cmp_nge_f32_e32", "Vector compare v_cmp_nge_f32"),
    0x0a: ("v_cmp_nlg_f32_e32", "Vector compare v_cmp_nlg_f32"),
    0x0b: ("v_cmp_ngt_f32_e32", "Vector compare v_cmp_ngt_f32"),
    0x0c: ("v_cmp_nle_f32_e32", "Vector compare v_cmp_nle_f32"),
    0x0d: ("v_cmp_neq_f32_e32", "Vector compare v_cmp_neq_f32"),
    0x0e: ("v_cmp_nlt_f32_e32", "Vector compare v_cmp_nlt_f32"),
    0x0f: ("v_cmp_tru_f32_e32", "Vector compare v_cmp_tru_f32"),
    0x10: ("v_cmpx_f_f32_e32", "Vector compare v_cmpx_f_f32"),
    0x11: ("v_cmpx_lt_f32_e32", "Vector compare v_cmpx_lt_f32"),
    0x12: ("v_cmpx_eq_f32_e32", "Vector compare v_cmpx_eq_f32"),
    0x13: ("v_cmpx_le_f32_e32", "Vector compare v_cmpx_le_f32"),
    0x14: ("v_cmpx_gt_f32_e32", "Vector compare v_cmpx_gt_f32"),
    0x15: ("v_cmpx_lg_f32_e32", "Vector compare v_cmpx_lg_f32"),
    0x16: ("v_cmpx_ge_f32_e32", "Vector compare v_cmpx_ge_f32"),
    0x17: ("v_cmpx_o_f32_e32", "Vector compare v_cmpx_o_f32"),
    0x18: ("v_cmpx_u_f32_e32", "Vector compare v_cmpx_u_f32"),
    0x19: ("v_cmpx_nge_f32_e32", "Vector compare v_cmpx_nge_f32"),
    0x1a: ("v_cmpx_nlg_f32_e32", "Vector compare v_cmpx_nlg_f32"),
    0x1b: ("v_cmpx_ngt_f32_e32", "Vector compare v_cmpx_ngt_f32"),
    0x1c: ("v_cmpx_nle_f32_e32", "Vector compare v_cmpx_nle_f32"),
    0x1d: ("v_cmpx_neq_f32_e32", "Vector compare v_cmpx_neq_f32"),
    0x1e: ("v_cmpx_nlt_f32_e32", "Vector compare v_cmpx_nlt_f32"),
    0x1f: ("v_cmpx_tru_f32_e32", "Vector compare v_cmpx_tru_f32"),
    0x20: ("v_cmp_f_f64_e32", "Vector compare v_cmp_f_f64"),
    0x21: ("v_cmp_lt_f64_e32", "Vector compare v_cmp_lt_f64"),
    0x22: ("v_cmp_eq_f64_e32", "Vector compare v_cmp_eq_f64"),
    0x23: ("v_cmp_le_f64_e32", "Vector compare v_cmp_le_f64"),
    0x24: ("v_cmp_gt_f64_e32", "Vector compare v_cmp_gt_f64"),
    0x25: ("v_cmp_lg_f64_e32", "Vector compare v_cmp_lg_f64"),
    0x26: ("v_cmp_ge_f64_e32", "Vector compare v_cmp_ge_f64"),
    0x27: ("v_cmp_o_f64_e32", "Vector compare v_cmp_o_f64"),
    0x28: ("v_cmp_u_f64_e32", "Vector compare v_cmp_u_f64"),
    0x29: ("v_cmp_nge_f64_e32", "Vector compare v_cmp_nge_f64"),
    0x2a: ("v_cmp_nlg_f64_e32", "Vector compare v_cmp_nlg_f64"),
    0x2b: ("v_cmp_ngt_f64_e32", "Vector compare v_cmp_ngt_f64"),
    0x2c: ("v_cmp_nle_f64_e32", "Vector compare v_cmp_nle_f64"),
    0x2d: ("v_cmp_neq_f64_e32", "Vector compare v_cmp_neq_f64"),
    0x2e: ("v_cmp_nlt_f64_e32", "Vector compare v_cmp_nlt_f64"),
    0x2f: ("v_cmp_tru_f64_e32", "Vector compare v_cmp_tru_f64"),
    0x30: ("v_cmpx_f_f64_e32", "Vector compare v_cmpx_f_f64"),
    0x31: ("v_cmpx_lt_f64_e32", "Vector compare v_cmpx_lt_f64"),
    0x32: ("v_cmpx_eq_f64_e32", "Vector compare v_cmpx_eq_f64"),
    0x33: ("v_cmpx_le_f64_e32", "Vector compare v_cmpx_le_f64"),
    0x34: ("v_cmpx_gt_f64_e32", "Vector compare v_cmpx_gt_f64"),
    0x35: ("v_cmpx_lg_f64_e32", "Vector compare v_cmpx_lg_f64"),
    0x36: ("v_cmpx_ge_f64_e32", "Vector compare v_cmpx_ge_f64"),
    0x37: ("v_cmpx_o_f64_e32", "Vector compare v_cmpx_o_f64"),
    0x38: ("v_cmpx_u_f64_e32", "Vector compare v_cmpx_u_f64"),
    0x39: ("v_cmpx_nge_f64_e32", "Vector compare v_cmpx_nge_f64"),
    0x3a: ("v_cmpx_nlg_f64_e32", "Vector compare v_cmpx_nlg_f64"),
    0x3b: ("v_cmpx_ngt_f64_e32", "Vector compare v_cmpx_ngt_f64"),
    0x3c: ("v_cmpx_nle_f64_e32", "Vector compare v_cmpx_nle_f64"),
    0x3d: ("v_cmpx_neq_f64_e32", "Vector compare v_cmpx_neq_f64"),
    0x3e: ("v_cmpx_nlt_f64_e32", "Vector compare v_cmpx_nlt_f64"),
    0x3f: ("v_cmpx_tru_f64_e32", "Vector compare v_cmpx_tru_f64"),
    0x80: ("v_cmp_f_i32_e32", "Vector compare v_cmp_f_i32"),
    0x81: ("v_cmp_lt_i32_e32", "Vector compare v_cmp_lt_i32"),
    0x82: ("v_cmp_eq_i32_e32", "Vector compare v_cmp_eq_i32"),
    0x83: ("v_cmp_le_i32_e32", "Vector compare v_cmp_le_i32"),
    0x84: ("v_cmp_gt_i32_e32", "Vector compare v_cmp_gt_i32"),
    0x85: ("v_cmp_ne_i32_e32", "Vector compare v_cmp_ne_i32"),
    0x86: ("v_cmp_ge_i32_e32", "Vector compare v_cmp_ge_i32"),
    0x87: ("v_cmp_t_i32_e32", "Vector compare v_cmp_t_i32"),
    0x88: ("v_cmp_class_f32_e32", "Vector compare v_cmp_class_f32"),
    0x89: ("v_cmp_lt_i16_e32", "Vector compare v_cmp_lt_i16"),
    0x8a: ("v_cmp_eq_i16_e32", "Vector compare v_cmp_eq_i16"),
    0x8b: ("v_cmp_le_i16_e32", "Vector compare v_cmp_le_i16"),
    0x8c: ("v_cmp_gt_i16_e32", "Vector compare v_cmp_gt_i16"),
    0x8d: ("v_cmp_ne_i16_e32", "Vector compare v_cmp_ne_i16"),
    0x8e: ("v_cmp_ge_i16_e32", "Vector compare v_cmp_ge_i16"),
    0x8f: ("v_cmp_class_f16_e32", "Vector compare v_cmp_class_f16"),
    0x90: ("v_cmpx_f_i32_e32", "Vector compare v_cmpx_f_i32"),
    0x91: ("v_cmpx_lt_i32_e32", "Vector compare v_cmpx_lt_i32"),
    0x92: ("v_cmpx_eq_i32_e32", "Vector compare v_cmpx_eq_i32"),
    0x93: ("v_cmpx_le_i32_e32", "Vector compare v_cmpx_le_i32"),
    0x94: ("v_cmpx_gt_i32_e32", "Vector compare v_cmpx_gt_i32"),
    0x95: ("v_cmpx_ne_i32_e32", "Vector compare v_cmpx_ne_i32"),
    0x96: ("v_cmpx_ge_i32_e32", "Vector compare v_cmpx_ge_i32"),
    0x97: ("v_cmpx_t_i32_e32", "Vector compare v_cmpx_t_i32"),
    0x98: ("v_cmpx_class_f32_e32", "Vector compare v_cmpx_class_f32"),
    0x99: ("v_cmpx_lt_i16_e32", "Vector compare v_cmpx_lt_i16"),
    0x9a: ("v_cmpx_eq_i16_e32", "Vector compare v_cmpx_eq_i16"),
    0x9b: ("v_cmpx_le_i16_e32", "Vector compare v_cmpx_le_i16"),
    0x9c: ("v_cmpx_gt_i16_e32", "Vector compare v_cmpx_gt_i16"),
    0x9d: ("v_cmpx_ne_i16_e32", "Vector compare v_cmpx_ne_i16"),
    0x9e: ("v_cmpx_ge_i16_e32", "Vector compare v_cmpx_ge_i16"),
    0x9f: ("v_cmpx_class_f16_e32", "Vector compare v_cmpx_class_f16"),
    0xa0: ("v_cmp_f_i64_e32", "Vector compare v_cmp_f_i64"),
    0xa1: ("v_cmp_lt_i64_e32", "Vector compare v_cmp_lt_i64"),
    0xa2: ("v_cmp_eq_i64_e32", "Vector compare v_cmp_eq_i64"),
    0xa3: ("v_cmp_le_i64_e32", "Vector compare v_cmp_le_i64"),
    0xa4: ("v_cmp_gt_i64_e32", "Vector compare v_cmp_gt_i64"),
    0xa5: ("v_cmp_ne_i64_e32", "Vector compare v_cmp_ne_i64"),
    0xa6: ("v_cmp_ge_i64_e32", "Vector compare v_cmp_ge_i64"),
    0xa7: ("v_cmp_t_i64_e32", "Vector compare v_cmp_t_i64"),
    0xa8: ("v_cmp_class_f64_e32", "Vector compare v_cmp_class_f64"),
    0xa9: ("v_cmp_lt_u16_e32", "Vector compare v_cmp_lt_u16"),
    0xaa: ("v_cmp_eq_u16_e32", "Vector compare v_cmp_eq_u16"),
    0xab: ("v_cmp_le_u16_e32", "Vector compare v_cmp_le_u16"),
    0xac: ("v_cmp_gt_u16_e32", "Vector compare v_cmp_gt_u16"),
    0xad: ("v_cmp_ne_u16_e32", "Vector compare v_cmp_ne_u16"),
    0xae: ("v_cmp_ge_u16_e32", "Vector compare v_cmp_ge_u16"),
    0xb0: ("v_cmpx_f_i64_e32", "Vector compare v_cmpx_f_i64"),
    0xb1: ("v_cmpx_lt_i64_e32", "Vector compare v_cmpx_lt_i64"),
    0xb2: ("v_cmpx_eq_i64_e32", "Vector compare v_cmpx_eq_i64"),
    0xb3: ("v_cmpx_le_i64_e32", "Vector compare v_cmpx_le_i64"),
    0xb4: ("v_cmpx_gt_i64_e32", "Vector compare v_cmpx_gt_i64"),
    0xb5: ("v_cmpx_ne_i64_e32", "Vector compare v_cmpx_ne_i64"),
    0xb6: ("v_cmpx_ge_i64_e32", "Vector compare v_cmpx_ge_i64"),
    0xb7: ("v_cmpx_t_i64_e32", "Vector compare v_cmpx_t_i64"),
    0xb8: ("v_cmpx_class_f64_e32", "Vector compare v_cmpx_class_f64"),
    0xb9: ("v_cmpx_lt_u16_e32", "Vector compare v_cmpx_lt_u16"),
    0xba: ("v_cmpx_eq_u16_e32", "Vector compare v_cmpx_eq_u16"),
    0xbb: ("v_cmpx_le_u16_e32", "Vector compare v_cmpx_le_u16"),
    0xbc: ("v_cmpx_gt_u16_e32", "Vector compare v_cmpx_gt_u16"),
    0xbd: ("v_cmpx_ne_u16_e32", "Vector compare v_cmpx_ne_u16"),
    0xbe: ("v_cmpx_ge_u16_e32", "Vector compare v_cmpx_ge_u16"),
    0xc0: ("v_cmp_f_u32_e32", "Vector compare v_cmp_f_u32"),
    0xc1: ("v_cmp_lt_u32_e32", "Vector compare v_cmp_lt_u32"),
    0xc2: ("v_cmp_eq_u32_e32", "Vector compare v_cmp_eq_u32"),
    0xc3: ("v_cmp_le_u32_e32", "Vector compare v_cmp_le_u32"),
    0xc4: ("v_cmp_gt_u32_e32", "Vector compare v_cmp_gt_u32"),
    0xc5: ("v_cmp_ne_u32_e32", "Vector compare v_cmp_ne_u32"),
    0xc6: ("v_cmp_ge_u32_e32", "Vector compare v_cmp_ge_u32"),
    0xc7: ("v_cmp_t_u32_e32", "Vector compare v_cmp_t_u32"),
    0xc8: ("v_cmp_f_f16_e32", "Vector compare v_cmp_f_f16"),
    0xc9: ("v_cmp_lt_f16_e32", "Vector compare v_cmp_lt_f16"),
    0xca: ("v_cmp_eq_f16_e32", "Vector compare v_cmp_eq_f16"),
    0xcb: ("v_cmp_le_f16_e32", "Vector compare v_cmp_le_f16"),
    0xcc: ("v_cmp_gt_f16_e32", "Vector compare v_cmp_gt_f16"),
    0xcd: ("v_cmp_lg_f16_e32", "Vector compare v_cmp_lg_f16"),
    0xce: ("v_cmp_ge_f16_e32", "Vector compare v_cmp_ge_f16"),
    0xcf: ("v_cmp_o_f16_e32", "Vector compare v_cmp_o_f16"),
    0xd0: ("v_cmpx_f_u32_e32", "Vector compare v_cmpx_f_u32"),
    0xd1: ("v_cmpx_lt_u32_e32", "Vector compare v_cmpx_lt_u32"),
    0xd2: ("v_cmpx_eq_u32_e32", "Vector compare v_cmpx_eq_u32"),
    0xd3: ("v_cmpx_le_u32_e32", "Vector compare v_cmpx_le_u32"),
    0xd4: ("v_cmpx_gt_u32_e32", "Vector compare v_cmpx_gt_u32"),
    0xd5: ("v_cmpx_ne_u32_e32", "Vector compare v_cmpx_ne_u32"),
    0xd6: ("v_cmpx_ge_u32_e32", "Vector compare v_cmpx_ge_u32"),
    0xd7: ("v_cmpx_t_u32_e32", "Vector compare v_cmpx_t_u32"),
    0xd8: ("v_cmpx_f_f16_e32", "Vector compare v_cmpx_f_f16"),
    0xd9: ("v_cmpx_lt_f16_e32", "Vector compare v_cmpx_lt_f16"),
    0xda: ("v_cmpx_eq_f16_e32", "Vector compare v_cmpx_eq_f16"),
    0xdb: ("v_cmpx_le_f16_e32", "Vector compare v_cmpx_le_f16"),
    0xdc: ("v_cmpx_gt_f16_e32", "Vector compare v_cmpx_gt_f16"),
    0xdd: ("v_cmpx_lg_f16_e32", "Vector compare v_cmpx_lg_f16"),
    0xde: ("v_cmpx_ge_f16_e32", "Vector compare v_cmpx_ge_f16"),
    0xdf: ("v_cmpx_o_f16_e32", "Vector compare v_cmpx_o_f16"),
    0xe0: ("v_cmp_f_u64_e32", "Vector compare v_cmp_f_u64"),
    0xe1: ("v_cmp_lt_u64_e32", "Vector compare v_cmp_lt_u64"),
    0xe2: ("v_cmp_eq_u64_e32", "Vector compare v_cmp_eq_u64"),
    0xe3: ("v_cmp_le_u64_e32", "Vector compare v_cmp_le_u64"),
    0xe4: ("v_cmp_gt_u64_e32", "Vector compare v_cmp_gt_u64"),
    0xe5: ("v_cmp_ne_u64_e32", "Vector compare v_cmp_ne_u64"),
    0xe6: ("v_cmp_ge_u64_e32", "Vector compare v_cmp_ge_u64"),
    0xe7: ("v_cmp_t_u64_e32", "Vector compare v_cmp_t_u64"),
    0xe8: ("v_cmp_u_f16_e32", "Vector compare v_cmp_u_f16"),
    0xe9: ("v_cmp_nge_f16_e32", "Vector compare v_cmp_nge_f16"),
    0xea: ("v_cmp_nlg_f16_e32", "Vector compare v_cmp_nlg_f16"),
    0xeb: ("v_cmp_ngt_f16_e32", "Vector compare v_cmp_ngt_f16"),
    0xec: ("v_cmp_nle_f16_e32", "Vector compare v_cmp_nle_f16"),
    0xed: ("v_cmp_neq_f16_e32", "Vector compare v_cmp_neq_f16"),
    0xee: ("v_cmp_nlt_f16_e32", "Vector compare v_cmp_nlt_f16"),
    0xef: ("v_cmp_tru_f16_e32", "Vector compare v_cmp_tru_f16"),
    0xf0: ("v_cmpx_f_u64_e32", "Vector compare v_cmpx_f_u64"),
    0xf1: ("v_cmpx_lt_u64_e32", "Vector compare v_cmpx_lt_u64"),
    0xf2: ("v_cmpx_eq_u64_e32", "Vector compare v_cmpx_eq_u64"),
    0xf3: ("v_cmpx_le_u64_e32", "Vector compare v_cmpx_le_u64"),
    0xf4: ("v_cmpx_gt_u64_e32", "Vector compare v_cmpx_gt_u64"),
    0xf5: ("v_cmpx_ne_u64_e32", "Vector compare v_cmpx_ne_u64"),
    0xf6: ("v_cmpx_ge_u64_e32", "Vector compare v_cmpx_ge_u64"),
    0xf7: ("v_cmpx_t_u64_e32", "Vector compare v_cmpx_t_u64"),
    0xf8: ("v_cmpx_u_f16_e32", "Vector compare v_cmpx_u_f16"),
    0xf9: ("v_cmpx_nge_f16_e32", "Vector compare v_cmpx_nge_f16"),
    0xfa: ("v_cmpx_nlg_f16_e32", "Vector compare v_cmpx_nlg_f16"),
    0xfb: ("v_cmpx_ngt_f16_e32", "Vector compare v_cmpx_ngt_f16"),
    0xfc: ("v_cmpx_nle_f16_e32", "Vector compare v_cmpx_nle_f16"),
    0xfd: ("v_cmpx_neq_f16_e32", "Vector compare v_cmpx_neq_f16"),
    0xfe: ("v_cmpx_nlt_f16_e32", "Vector compare v_cmpx_nlt_f16"),
    0xff: ("v_cmpx_tru_f16_e32", "Vector compare v_cmpx_tru_f16"),
}


VOP2_OPS = {
    0x01: ("v_cndmask_b32_e32", "Vector conditional mask select"),
    0x03: ("v_add_f32_e32", "Vector add float32"),
    0x04: ("v_sub_f32_e32", "Vector subtract float32"),
    0x05: ("v_subrev_f32_e32", "Vector reverse subtract float32"),
    0x07: ("v_mul_legacy_f32_e32", "Vector multiply zero-preserving float32"),
    0x08: ("v_mul_f32_e32", "Vector multiply float32"),
    0x0f: ("v_min_f32_e32", "Vector minimum float32"),
    0x10: ("v_max_f32_e32", "Vector maximum float32"),
    0x11: ("v_min_i32_e32", "Vector minimum int32"),
    0x12: ("v_max_i32_e32", "Vector maximum int32"),
    0x13: ("v_min_u32_e32", "Vector minimum uint32"),
    0x14: ("v_max_u32_e32", "Vector maximum uint32"),
    0x16: ("v_lshrrev_b32_e32", "Vector shift right logical"),
    0x18: ("v_ashrrev_i32_e32", "Vector shift right arithmetic"),
    0x1a: ("v_lshlrev_b32_e32", "Vector shift left logical"),
    0x1b: ("v_and_b32_e32", "Vector bitwise AND"),
    0x1c: ("v_or_b32_e32", "Vector bitwise OR"),
    0x1d: ("v_xor_b32_e32", "Vector bitwise XOR"),
    0x25: ("v_add_nc_u32_e32", "Vector add uint32 without carry"),
    0x26: ("v_sub_nc_u32_e32", "Vector sub uint32 without carry"),
    0x27: ("v_subrev_nc_u32_e32", "Vector reverse sub uint32 without carry"),
    0x2b: ("v_fmac_f32_e32", "Vector fused multiply-accumulate float32"),
    0x2c: ("v_fmamk_f32", "Vector fused multiply-add with literal multiplier"),
    0x2d: ("v_fmaak_f32", "Vector fused multiply-add with literal addend"),
    0x2f: ("v_cvt_pkrtz_f16_f32_e32", "Convert two float32 to packed float16 (round to zero)"),
}

VOP3P_OPS = {
    0x00: ("v_pk_mad_i16", "Multiply-add signed 16-bit packed"),
    0x01: ("v_pk_mul_lo_u16", "Multiply low unsigned 16-bit packed"),
    0x02: ("v_pk_add_i16", "Add signed 16-bit packed"),
    0x03: ("v_pk_sub_i16", "Subtract signed 16-bit packed"),
    0x04: ("v_pk_lshlrev_b16", "Logical shift left reverse 16-bit packed"),
    0x05: ("v_pk_lshrrev_b16", "Logical shift right reverse 16-bit packed"),
    0x06: ("v_pk_ashrrev_i16", "Arithmetic shift right reverse 16-bit packed"),
    0x07: ("v_pk_max_i16", "Maximum signed 16-bit packed"),
    0x08: ("v_pk_min_i16", "Minimum signed 16-bit packed"),
    0x09: ("v_pk_mad_u16", "Multiply-add unsigned 16-bit packed"),
    0x0a: ("v_pk_add_u16", "Add unsigned 16-bit packed"),
    0x0b: ("v_pk_sub_u16", "Subtract unsigned 16-bit packed"),
    0x0c: ("v_pk_max_u16", "Maximum unsigned 16-bit packed"),
    0x0d: ("v_pk_min_u16", "Minimum unsigned 16-bit packed"),
    0x0e: ("v_pk_fma_f16", "Fused multiply-add float16 packed"),
    0x0f: ("v_pk_add_f16", "Add float16 packed"),
    0x10: ("v_pk_mul_f16", "Multiply float16 packed"),
    0x11: ("v_pk_min_f16", "Minimum float16 packed"),
    0x12: ("v_pk_max_f16", "Maximum float16 packed"),
    0x13: ("v_dot2_f32_f16", "Dot product 2-element float16 accumulate to float32"),
    0x14: ("v_dot2_i32_i16", "Dot product 2-element signed 16-bit accumulate to int32"),
    0x15: ("v_dot2_u32_u16", "Dot product 2-element unsigned 16-bit accumulate to uint32"),
    0x16: ("v_dot4_i32_i8", "Dot product 4-element signed 8-bit accumulate to int32"),
    0x17: ("v_dot4_u32_u8", "Dot product 4-element unsigned 8-bit accumulate to uint32"),
    0x18: ("v_dot8_i32_i4", "Dot product 8-element signed 4-bit accumulate to int32"),
    0x19: ("v_dot8_u32_u4", "Dot product 8-element unsigned 4-bit accumulate to uint32"),
    0x20: ("v_fma_mix_f32", "Fused multiply-add mixed precision to float32"),
    0x21: ("v_fma_mixlo_f16", "Fused multiply-add mixed precision to float16 low"),
    0x22: ("v_fma_mixhi_f16", "Fused multiply-add mixed precision to float16 high"),
}

VOP3P_3SRC = {
    0x00, 0x09, 0x0e, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22,
}

EXP_TARGETS = {
    0: "mrt0", 1: "mrt1", 2: "mrt2", 3: "mrt3",
    4: "mrt4", 5: "mrt5", 6: "mrt6", 7: "mrt7",
    8: "mrtz", 9: "null",
    12: "pos0", 13: "pos1", 14: "pos2", 15: "pos3",
    **{32 + i: f"param{i}" for i in range(32)},
}


def decode_instruction(data: bytes, offset: int) -> tuple[int, str, str, str]:
    """
    Decode one RDNA 2 instruction at given offset.
    Returns: (instruction_byte_length, opcode_name, assembly_string, comment)
    """
    if offset + 4 > len(data):
        return (len(data) - offset, "data", f".byte {data[offset:].hex()}", "Trailing bytes")

    w0, = struct.unpack_from("<I", data, offset)

    # 1. SOPP: [31:23] == 0b101111111 (0xbf800000)
    if (w0 & 0xff800000) == 0xbf800000:
        op = (w0 >> 16) & 0x7f
        simm = w0 & 0xffff
        signed_simm = simm if simm < 0x8000 else simm - 0x10000
        op_info = SOPP_OPS.get(op, (f"s_sopp_{op:02x}", "Unknown SOPP"))
        op_name, comment = op_info

        if op_name == "s_endpgm":
            asm = "s_endpgm"
        elif op_name == "s_nop":
            asm = f"s_nop {simm}"
        elif op_name == "s_waitcnt":
            vm = (simm >> 4) & 0xf
            lgkm = (simm >> 8) & 0x3f
            asm = f"s_waitcnt vmcnt({vm}) lgkmcnt({lgkm})"
        elif op_name.startswith("s_cbranch") or op_name == "s_branch":
            target = offset + 4 + signed_simm * 4
            asm = f"{op_name} label_0x{target:x}"
        else:
            asm = f"{op_name} 0x{simm:x}"
        return (4, op_name, asm, comment)

    # 2. SOP1: [31:23] == 0b101111101 (0xbe800000)
    if (w0 & 0xff800000) == 0xbe800000:
        sdst = (w0 >> 16) & 0x7f
        op = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOP1_OPS.get(op, (f"s_sop1_{op:02x}", "Unknown SOP1"))
        op_name, comment = op_info
        length = 4
        if ssrc0 == 255:
            if offset + 8 > len(data):
                return (4, "sop1_truncated", f"{op_name} <truncated>", "Incomplete SOP1 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            src0_str = f"0x{literal:x}"
            length = 8
        else:
            src0_str = decode_s_reg(ssrc0)
        asm = f"{op_name} {decode_s_reg(sdst)}, {src0_str}"
        return (length, op_name, asm, comment)

    # SOPC: [31:23] == 0b101111110 (0xbf000000)
    # Must run before the SOPK prefix check, which would otherwise consume these words.
    if (w0 & 0xff800000) == 0xbf000000:
        op = (w0 >> 16) & 0x7f
        ssrc1 = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOPC_OPS.get(op, (f"s_sopc_{op:02x}", "Unknown SOPC"))
        op_name, comment = op_info
        length = 4
        literal = None
        if ssrc0 == 255 or ssrc1 == 255:
            if offset + 8 > len(data):
                return (4, "sopc_truncated", f"{op_name} <truncated>", "Incomplete SOPC literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8

        def sopc_operand(reg_id: int, wide: bool) -> str:
            if reg_id == 255 and literal is not None:
                return f"0x{literal:x}"
            if wide:
                return decode_s_reg64(reg_id)
            return decode_s_reg(reg_id)

        src0_wide = op_name in {"s_cmp_eq_u64", "s_cmp_lg_u64", "s_bitcmp0_b64", "s_bitcmp1_b64"}
        src1_wide = op_name in {"s_cmp_eq_u64", "s_cmp_lg_u64"}
        asm = f"{op_name} {sopc_operand(ssrc0, src0_wide)}, {sopc_operand(ssrc1, src1_wide)}"
        return (length, op_name, asm, comment)

    # 3. SOPK: [31:28] == 0b1011 (0xb0000000)
    if (w0 & 0xf0000000) == 0xb0000000 and (w0 & 0x0f800000) not in (0x0f800000, 0x0e800000, 0x0f000000):
        op = (w0 >> 23) & 0x1f
        sdst = (w0 >> 16) & 0x7f
        simm = w0 & 0xffff
        signed_simm = simm if simm < 0x8000 else simm - 0x10000
        op_info = SOPK_OPS.get(op, (f"s_sopk_{op:02x}", "Unknown SOPK"))
        op_name, comment = op_info
        asm = f"{op_name} {decode_s_reg(sdst)}, 0x{simm:x}"
        return (4, op_name, asm, comment)

    # 4. SOP2: [31:30] == 0b10 (0x80000000)
    if (w0 & 0xc0000000) == 0x80000000 and (w0 & 0x30000000) != 0x30000000:
        op = (w0 >> 23) & 0x7f
        sdst = (w0 >> 16) & 0x7f
        ssrc1 = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOP2_OPS.get(op, (f"s_sop2_{op:02x}", "Unknown SOP2"))
        op_name, comment = op_info
        length = 4
        literal = None
        if ssrc0 == 255 or ssrc1 == 255:
            if offset + 8 > len(data):
                return (4, "sop2_truncated", f"{op_name} <truncated>", "Incomplete SOP2 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        src0_str = f"0x{literal:x}" if ssrc0 == 255 and literal is not None else decode_s_reg(ssrc0)
        src1_str = f"0x{literal:x}" if ssrc1 == 255 and literal is not None else decode_s_reg(ssrc1)
        asm = f"{op_name} {decode_s_reg(sdst)}, {src0_str}, {src1_str}"
        return (length, op_name, asm, comment)

    # 5. VOP1: [31:25] == 0b0111111 (0x7e000000)
    if (w0 & 0xfe000000) == 0x7e000000:
        vdst = (w0 >> 17) & 0xff
        op = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOP1_OPS.get(op, (f"v_vop1_{op:02x}", "Unknown VOP1"))
        op_name, comment = op_info
        length = 4
        if src0 == 255:
            if offset + 8 > len(data):
                return (4, "vop1_truncated", f"{op_name} <truncated>", "Incomplete VOP1 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            src0_str = f"0x{literal:x}"
            length = 8
        else:
            src0_str = decode_v_reg(src0)
        if op_name == "v_nop":
            asm = "v_nop"
        else:
            asm = f"{op_name} v{vdst}, {src0_str}"
        return (length, op_name, asm, comment)

    # 6. VOPC: [31:25] == 0b0111110 (0x7c000000)
    if (w0 & 0xfe000000) == 0x7c000000:
        op = (w0 >> 17) & 0xff
        vsrc1 = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOPC_OPS.get(op, (f"v_vopc_{op:02x}", "Unknown VOPC"))
        op_name, comment = op_info
        has_literal = src0 == 255
        length = 4
        literal = None
        if has_literal:
            if offset + 8 > len(data):
                return (4, "vopc_truncated", f"{op_name} <truncated>", "Incomplete VOPC literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        src0_str = f"0x{literal:x}" if src0 == 255 and literal is not None else decode_v_reg(src0)
        if op_name.startswith("v_cmpx_"):
            asm = f"{op_name} {src0_str}, v{vsrc1}"
        else:
            asm = f"{op_name} vcc_lo, {src0_str}, v{vsrc1}"
        return (length, op_name, asm, comment)

    # 7. VOP2: [31] == 0
    if (w0 & 0x80000000) == 0:
        op = (w0 >> 25) & 0x3f
        vdst = (w0 >> 17) & 0xff
        vsrc1 = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOP2_OPS.get(op, (f"v_vop2_{op:02x}", "Unknown VOP2"))
        op_name, comment = op_info
        has_literal = src0 == 255 or op in (0x2c, 0x2d)
        length = 4
        literal = None
        if has_literal:
            if offset + 8 > len(data):
                return (4, "vop2_truncated", f"{op_name} <truncated>", "Incomplete VOP2 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        src0_str = f"0x{literal:x}" if src0 == 255 and literal is not None else decode_v_reg(src0)
        if op_name == "v_cndmask_b32_e32":
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}, vcc_lo"
        elif op == 0x2c:
            asm = f"{op_name} v{vdst}, {src0_str}, 0x{literal:x}, v{vsrc1}"
        elif op == 0x2d:
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}, 0x{literal:x}"
        else:
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}"
        return (length, op_name, asm, comment)

    # 8. VINTRP: [31:26] == 0b110010 (0xc8000000)
    if (w0 & 0xfc000000) == 0xc8000000:
        vdst = (w0 >> 18) & 0xff
        op = (w0 >> 16) & 0x3
        attr = (w0 >> 10) & 0x3f
        chan = "xyzw"[(w0 >> 8) & 0x3]
        vsrc = w0 & 0xff
        op_names = {0: "v_interp_p1_f32_e32", 1: "v_interp_p2_f32_e32", 2: "v_interp_mov_f32_e32"}
        op_name = op_names.get(op, f"v_interp_{op}")
        asm = f"{op_name} v{vdst}, v{vsrc}, attr{attr}.{chan}"
        comment = f"Parameter interpolation stage {op+1} for attribute {attr}.{chan}"
        return (4, op_name, asm, comment)

    # 9. EXP: 64-bit [31:26] == 0b111110 (0xf8000000)
    if (w0 & 0xfc000000) == 0xf8000000:
        if offset + 8 > len(data):
            return (4, "exp_truncated", "exp <truncated>", "Incomplete 64-bit export instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        en = w0 & 0xf
        target = (w0 >> 4) & 0x3f
        compr = (w0 >> 10) & 1
        done = (w0 >> 11) & 1
        vm = (w0 >> 12) & 1
        vs = [(w1 >> (8 * k)) & 0xff for k in range(4)]
        if compr:
            lo = f"v{vs[0]}" if en & 0x1 else "off"
            hi = f"v{vs[1]}" if en & 0x4 else "off"
            regs = [lo, lo, hi, hi]
        else:
            regs = [f"v{vs[k]}" if (en & (1 << k)) else "off" for k in range(4)]
        target_name = EXP_TARGETS.get(target, f"target_{target}")
        flags = []
        if done:
            flags.append("done")
        if compr:
            flags.append("compr")
        if vm:
            flags.append("vm")
        # The printed compressed operands use the first bit of each pair.
        # Keep a non-paired raw mask visible so it cannot masquerade as a supported export.
        if compr and en not in (0x0, 0x3, 0xc, 0xf):
            flags.append(f"en:{en:#x}")
        flags_str = " ".join(flags)
        asm = f"exp {target_name} {', '.join(regs)} {flags_str}".strip()
        comment = f"Export output pixels/attributes to {target_name.upper()}"
        return (8, "exp", asm, comment)

    # 10. SMEM: 64-bit [31:26] == 0b111101 (0xf4000000)
    if (w0 & 0xfc000000) == 0xf4000000:
        if offset + 8 > len(data):
            return (4, "smem_truncated", "smem <truncated>", "Incomplete 64-bit SMEM instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 18) & 0xff
        sdata = (w0 >> 6) & 0x7f
        sbase = (w0 & 0x3f) * 2
        op_name = SMEM_OPS.get(op, f"s_smem_{op:02x}")
        data_width, base_width = SMEM_WIDTHS.get(op, (4, 2))
        sdata_str = decode_s_reg(sdata) if data_width == 1 else f"s[{sdata}:{sdata + data_width - 1}]"
        if base_width == 2 and sbase == 106:
            sbase_str = "vcc"
        else:
            sbase_str = f"s[{sbase}:{sbase + base_width - 1}]"
        soffset = (w1 >> 25) & 0x7f
        imm_offset = w1 & 0x1fffff
        if soffset == 125:
            off_str = f"0x{imm_offset:x}"
        else:
            off_str = decode_s_reg(soffset)
            if imm_offset:
                off_str += f" offset:0x{imm_offset:x}"
        mods = []
        if (w0 >> 16) & 1:
            mods.append("glc")
        if (w0 >> 14) & 1:
            mods.append("dlc")
        mod_str = (" " + " ".join(mods)) if mods else ""
        asm = f"{op_name} {sdata_str}, {sbase_str}, {off_str}{mod_str}"
        comment = "Scalar memory load from constant buffer/resource descriptor"
        return (8, op_name, asm, comment)

    # 11. MIMG: 64-bit + optional NSA dwords [31:26] == 0b111100 (0xf0000000)
    if (w0 & 0xfc000000) == 0xf0000000:
        nsize = (w0 >> 1) & 0x3
        length = 8 + 4 * nsize
        if offset + length > len(data):
            return (len(data) - offset, "mimg_truncated", "mimg <truncated>", "Incomplete MIMG instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 18) & 0x7f
        op_name, has_sampler, extra_addrs = MIMG_OPS.get(op, (f"image_mimg_{op:02x}", False, 0))
        dmask = (w0 >> 8) & 0xf
        dim_code = (w0 >> 3) & 0x7
        dim_name, base_addrs, deriv_addrs = MIMG_DIMS[dim_code]
        addr_count = base_addrs + extra_addrs + (deriv_addrs if op == 0x22 else 0)
        d16 = (w1 >> 31) & 1
        n_data = max(1, bin(dmask).count("1"))
        if d16:
            n_data = (n_data + 1) // 2
        vaddr0 = w1 & 0xff
        vdata = (w1 >> 8) & 0xff
        srsrc = ((w1 >> 16) & 0x1f) * 4
        ssamp = ((w1 >> 21) & 0x1f) * 4
        r128 = (w0 >> 15) & 1
        vdata_str = f"v{vdata}" if n_data == 1 else f"v[{vdata}:{vdata + n_data - 1}]"
        if nsize > 0:
            extra_bytes = list(data[offset + 8:offset + length])
            addrs = [vaddr0] + extra_bytes[:max(0, addr_count - 1)]
            vaddr_str = "[" + ", ".join(f"v{r}" for r in addrs) + "]"
        else:
            vaddr_str = f"v{vaddr0}" if addr_count == 1 else f"v[{vaddr0}:{vaddr0 + addr_count - 1}]"
        srsrc_width = 4 if r128 else 8
        srsrc_str = f"s[{srsrc}:{srsrc + srsrc_width - 1}]"
        operands = [vdata_str, vaddr_str, srsrc_str]
        if has_sampler:
            operands.append(f"s[{ssamp}:{ssamp + 3}]")
        mods = [f"dmask:0x{dmask:x}", f"dim:{dim_name}"]
        if (w0 >> 12) & 1:
            mods.append("unorm")
        if (w0 >> 13) & 1:
            mods.append("glc")
        if (w0 >> 25) & 1:
            mods.append("slc")
        if r128:
            mods.append("r128")
        if (w0 >> 14) & 1:
            mods.append("lwe")
        if (w0 >> 7) & 1:
            mods.append("dlc")
        if d16:
            mods.append("d16")
        asm = f"{op_name} {', '.join(operands)} {' '.join(mods)}"
        comment = "Image memory load/store/sample operation"
        return (length, op_name, asm, comment)

    # 12. VOP3P: 64-bit [31:23] == 0b110011000 (0xcc000000)
    if (w0 & 0xff800000) == 0xcc000000:
        if offset + 8 > len(data):
            return (4, "vop3p_truncated", "vop3p <truncated>", "Incomplete 64-bit VOP3P instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 16) & 0x7f
        clamp = (w0 >> 15) & 1
        vdst = w0 & 0xff
        src0 = w1 & 0x1ff
        src1 = (w1 >> 9) & 0x1ff
        src2 = (w1 >> 18) & 0x1ff

        op_info = VOP3P_OPS.get(op, (f"v_vop3p_{op:02x}", "Unknown VOP3P"))
        op_name, comment = op_info
        is_3src = op in VOP3P_3SRC
        arity = 3 if is_3src else 2

        length = 8
        literal = None
        if src0 == 255 or src1 == 255 or (is_3src and src2 == 255):
            if offset + 12 > len(data):
                return (8, "vop3p_truncated", f"{op_name} <truncated>", "Incomplete VOP3P literal")
            literal, = struct.unpack_from("<I", data, offset + 8)
            length = 12

        # Check for unmodeled / reserved bits in 2-source VOP3P encodings.
        # Note: bit 14 of w0 is op_sel_hi[2], which LLVM's MC assembler defaults to 1 (op_sel_hi = 0b111)
        # for all packed VOP3P instructions regardless of arity, so bit 14 is ignored on 2-source forms.
        if not is_3src and (src2 != 0 or ((w0 >> 10) & 1) or ((w0 >> 13) & 1) or ((w1 >> 31) & 1)):
            return (length, f"{op_name}_unsupported", f"{op_name} <unsupported_reserved_bits>", "Reserved bitfields set in 2-source VOP3P encoding")

        def vop3p_operand(reg_id: int) -> str:
            if reg_id == 255 and literal is not None:
                return f"0x{literal:x}"
            return decode_v_reg(reg_id)

        if is_3src:
            asm = f"{op_name} v{vdst}, {vop3p_operand(src0)}, {vop3p_operand(src1)}, {vop3p_operand(src2)}"
            op_sel = [(w0 >> 11) & 1, (w0 >> 12) & 1, (w0 >> 13) & 1]
            op_sel_hi = [(w1 >> 27) & 1, (w1 >> 28) & 1, (w0 >> 14) & 1]
            neg_lo = [(w1 >> 29) & 1, (w1 >> 30) & 1, (w1 >> 31) & 1]
            neg_hi = [(w0 >> 8) & 1, (w0 >> 9) & 1, (w0 >> 10) & 1]
        else:
            asm = f"{op_name} v{vdst}, {vop3p_operand(src0)}, {vop3p_operand(src1)}"
            op_sel = [(w0 >> 11) & 1, (w0 >> 12) & 1]
            op_sel_hi = [(w1 >> 27) & 1, (w1 >> 28) & 1]
            neg_lo = [(w1 >> 29) & 1, (w1 >> 30) & 1]
            neg_hi = [(w0 >> 8) & 1, (w0 >> 9) & 1]

        # Architectural defaults:
        # Standard packed instructions (v_pk_*, v_dot*): default op_sel is all 0, op_sel_hi is all 1, neg_lo/neg_hi all 0.
        # Mixed-precision FMA (v_fma_mix*): default op_sel and op_sel_hi are all 0 (f32, low half).
        if op_name.startswith("v_fma_mix"):
            default_op_sel = [0] * arity
            default_op_sel_hi = [0] * arity
        else:
            default_op_sel = [0] * arity
            default_op_sel_hi = [1] * arity
        default_neg_lo = [0] * arity
        default_neg_hi = [0] * arity

        if op_sel != default_op_sel:
            asm += f" op_sel:[{','.join(str(b) for b in op_sel)}]"
        if op_sel_hi != default_op_sel_hi:
            asm += f" op_sel_hi:[{','.join(str(b) for b in op_sel_hi)}]"
        if neg_lo != default_neg_lo:
            asm += f" neg_lo:[{','.join(str(b) for b in neg_lo)}]"
        if neg_hi != default_neg_hi:
            asm += f" neg_hi:[{','.join(str(b) for b in neg_hi)}]"

        if clamp:
            asm += " clamp"

        return (length, op_name, asm, comment)

    # Default fallback: unknown 32-bit word
    return (4, f"word_0x{w0:08x}", f".long 0x{w0:08x}", "Unrecognized RDNA 2 instruction encoding")


def disassemble_binary(data: bytes, shader_name: str = "shader") -> list[dict]:
    """Disassemble raw binary RDNA 2 shader into structured instruction records."""
    instructions = []
    offset = 0
    while offset < len(data):
        length, op_name, asm, comment = decode_instruction(data, offset)
        hex_bytes = data[offset:offset + length].hex()
        instructions.append({
            "offset": offset,
            "length": length,
            "hex": hex_bytes,
            "opcode": op_name,
            "assembly": asm,
            "comment": comment,
        })
        offset += length
        if op_name == "s_endpgm":
            # Some shaders have 4 trailing zero padding bytes after endpgm
            break
    return instructions


# Alias for library usage
disasm_rdna2 = disassemble_binary


def print_disassembly(shader_name: str, instructions: list[dict], show_comments: bool = True):
    """Print beautifully formatted disassembly with offsets and annotations."""
    print("=" * 80)
    print(f" RDNA 2 DISASSEMBLY: {shader_name} ")
    print("=" * 80)
    print(f"{'Offset':<8} {'Raw Hex':<18} {'Assembly':<42} {'Notes'}")
    print("-" * 80)

    for item in instructions:
        off_str = f"+0x{item['offset']:04x}"
        hex_str = item["hex"]
        asm_str = item["assembly"]
        comment_str = f"// {item['comment']}" if (show_comments and item["comment"]) else ""
        print(f"{off_str:<8} {hex_str:<18} {asm_str:<42} {comment_str}")

    print("=" * 80)
    total_bytes = sum(i["length"] for i in instructions)
    opcodes = Counter(i["opcode"] for i in instructions)
    print(f"Summary: {len(instructions)} instructions | {total_bytes} bytes | Top opcodes: {', '.join(f'{k}:{v}' for k, v in opcodes.most_common(5))}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", type=Path, default=[], help="Path to shader binary .bin file(s) or folder")
    parser.add_argument("--hex", default=None, help="Direct hex string to decode (e.g. '0300a0bf8002047e')")
    parser.add_argument("--output", type=Path, default=None, help="Save disassembly listing to output file")

    args = parser.parse_args()

    # Hex decode mode
    if args.hex:
        raw_bytes = bytes.fromhex(args.hex.replace(" ", ""))
        insns = disassemble_binary(raw_bytes, "hex_input")
        print_disassembly("hex_input", insns)
        return 0

    if not args.targets:
        parser.error("Specify targets or provide hex via --hex")

    all_listings = []
    for target in args.targets:
        if target.is_file():
            data = target.read_bytes()
            insns = disassemble_binary(data, target.name)
            print_disassembly(target.name, insns)
            all_listings.append((target.name, insns))
        elif target.is_dir():
            files = sorted(target.glob("*.bin"))
            print(f"[*] Found {len(files)} shader binary files in {target.name}...")
            for f in files[:20]:  # Limit preview to 20 when folder passed
                data = f.read_bytes()
                insns = disassemble_binary(data, f.name)
                print_disassembly(f.name, insns)
                all_listings.append((f.name, insns))

    if args.output and all_listings:
        with args.output.open("w", encoding="utf-8") as out_f:
            for name, insns in all_listings:
                out_f.write(f"; Disassembly: {name}\n")
                for item in insns:
                    out_f.write(f"    {item['assembly']:<40} ; +0x{item['offset']:04x} // {item['comment']}\n")
                out_f.write("\n")
        print(f"[+] Disassembly saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
