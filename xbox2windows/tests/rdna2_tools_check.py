"""Automated checks with source-owned inputs for RDNA 2 disassembly, extraction, and scanning tools."""
import argparse
from io import StringIO
import json
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from disasm_rdna2 import disasm_rdna2, decode_instruction, disassemble_binary
from extract_shaders import extract_shaders, extract_native_rdna2_code, sanitize_filename
from pixel_aot import VOP3_INT16, VOP3_INT16_TERNARY, fma_mix_operand, lower, native_program, parse_fma_mix
from scan_unsupported_shaders import (
    analyze_shader_instructions,
    is_instruction_supported,
    is_opcode_recognized,
    classify_isa_category,
    generate_reports,
    scan_directory_or_files,
    detect_shader_stage,
    generate_progress_dashboard,
    disassemble_binary_shader,
    make_failed_shader_record,
    KNOWN_RDNA2_BASE_OPCODES,
)
try:
    from pixel_aot import disassemble as llvm_disassemble
except ImportError:
    llvm_disassemble = None


def resolve_llvm_lib(name_or_path: str) -> str:
    """Resolve an LLVM shared library name or path for ctypes.CDLL loading."""
    p = Path(name_or_path)
    if p.is_file():
        return str(p.resolve())
    import ctypes.util
    found = ctypes.util.find_library(name_or_path)
    if found and Path(found).is_file():
        return str(Path(found).resolve())
    for directory in [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
    ]:
        candidate = Path(directory) / name_or_path
        if candidate.is_file():
            return str(candidate.resolve())
    return name_or_path


def resolve_tool(env_var: str, default_names: list[str]) -> str | None:
    """Resolve a command-line tool path via environment variable or system search."""
    val = os.environ.get(env_var)
    if val:
        w = shutil.which(val)
        if w:
            return w
        p = Path(val)
        if p.is_file():
            return str(p.resolve())
    for name in default_names:
        w = shutil.which(name)
        if w:
            return w
    return None


def assemble_with_llvm_mc(instructions: list[str], assembler: str, objcopy: str) -> bytes:
    """Assemble AMDGPU instructions for gfx1013 using llvm-mc and extract .text bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "fixture.s"
        obj = tmp_dir / "fixture.o"
        binary = tmp_dir / "fixture.bin"
        src.write_text("\n".join(instructions) + "\n", encoding="ascii")
        subprocess.run(
            [assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", str(src), "-o", str(obj)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [objcopy, "-O", "binary", "--only-section=.text", str(obj), str(binary)],
            check=True,
            capture_output=True,
        )
        return binary.read_bytes()


def envelope(code: bytes, name: bytes = b"fixture_ps") -> bytes:
    """Create a source-owned DXBC container wrapping native GFX1013 machine code."""
    program_start = 256
    image = bytearray(program_start) + code
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 192, program_start - 40)
    native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + zlib.compress(image)
    program = (bytes(24) + b"XDAT" + struct.pack("<II", 1, 0)
               + struct.pack("<III", 13, 48, len(native)) + native)
    blob = (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 44 + len(program), 1, 36)
            + b"DXIL" + struct.pack("<I", len(program)) + program)
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(blob) + encoder.flush()
    return (bytes([len(name) + 1]) + name + b"\0\0"
            + struct.pack("<4I", len(payload), 0, len(payload), len(blob)) + payload)


class TestRdna2Disasm(unittest.TestCase):
    """Test pure-Python RDNA 2 disassembler and verify against golden vectors and LLVM."""

    # GFX1013 e32 encodings from llvm-mc 19 -show-encoding.
    VOP1_VOP2_E32_VECTORS = (
        ("050f066c", "v_fmac_f16_e32 v3, v5, v7"),
        ("0d231676", "v_ldexp_f16_e32 v11, v13, v17"),
        ("1785267e", "v_movreld_b32_e32 v19, v23"),
        ("1f873a7e", "v_movrels_b32_e32 v29, v31"),
        ("29894a7e", "v_movrelsd_b32_e32 v37, v41"),
        ("2f91567e", "v_movrelsd_2_b32_e32 v43, v47"),
    )

    def test_vop1_vop2_e32_gfx1013(self):
        for encoding, assembly in self.VOP1_VOP2_E32_VECTORS:
            with self.subTest(assembly=assembly):
                record, = disasm_rdna2(bytes.fromhex(encoding))
                self.assertEqual(record["assembly"], assembly)

    def test_vop1_vop2_e32_llvm_round_trip(self):
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])
        if not assembler or not objcopy:
            self.fail(f"VOP1/VOP2 round trip requires llvm-mc and llvm-objcopy (assembler={assembler}, objcopy={objcopy})")
        source = [assembly for _, assembly in self.VOP1_VOP2_E32_VECTORS]
        expected = bytes.fromhex("".join(encoding for encoding, _ in self.VOP1_VOP2_E32_VECTORS))
        emitted = assemble_with_llvm_mc(source, assembler, objcopy)
        self.assertEqual(emitted, expected)
        decoded = [record["assembly"] for record in disasm_rdna2(emitted)]
        self.assertEqual(decoded, llvm_disassemble(emitted, resolved_lib))
        self.assertEqual(decoded, source)
        self.assertEqual(assemble_with_llvm_mc(decoded, assembler, objcopy), emitted)

    def test_golden_vectors(self):
        vectors = [
            # (hex_bytes, expected_opcode, expected_mnemonic)
            ("000080bf", "s_nop", "s_nop 0"),
            ("000081bf", "s_endpgm", "s_endpgm"),
            ("00009fbf", "s_code_end", "s_code_end"),
            ("00000000", "word_0x00000000", ".long 0x00000000"),
            ("000082bf", "s_branch", "s_branch label_0x4"),
            ("7f008cbf", "s_waitcnt", "s_waitcnt vmcnt(15) lgkmcnt(0)"),
            ("000382be", "s_mov_b32", "s_mov_b32 s2, s0"),
            ("000482be", "s_mov_b64", "s_mov_b64 s[2:3], s[0:1]"),
            ("000882be", "s_not_b64", "s_not_b64 s[2:3], s[0:1]"),
            ("003c82be", "s_and_saveexec_b32", "s_and_saveexec_b32 s2, s0"),
            ("02048087", "s_and_b64", "s_and_b64 s[0:1], s[2:3], s[4:5]"),
            ("0204808f", "s_lshl_b64", "s_lshl_b64 s[0:1], s[2:3], s4"),
            ("7e04b2be", "s_mov_b64", "s_mov_b64 s[50:51], exec"),
            ("7e0afebe", "s_wqm_b64", "s_wqm_b64 exec, exec"),
            ("7e32fe87", "s_and_b64", "s_and_b64 exec, exec, s[50:51]"),
            ("000b82be", "s_brev_b32", "s_brev_b32 s2, s0"),
            ("000c82be", "s_brev_b64", "s_brev_b64 s[2:3], s[0:1]"),
            ("000d82be", "s_bcnt0_i32_b32", "s_bcnt0_i32_b32 s2, s0"),
            ("000e82be", "s_bcnt0_i32_b64", "s_bcnt0_i32_b64 s2, s[0:1]"),
            ("000f82be", "s_bcnt1_i32_b32", "s_bcnt1_i32_b32 s2, s0"),
            ("001082be", "s_bcnt1_i32_b64", "s_bcnt1_i32_b64 s2, s[0:1]"),
            ("001182be", "s_ff0_i32_b32", "s_ff0_i32_b32 s2, s0"),
            ("001282be", "s_ff0_i32_b64", "s_ff0_i32_b64 s2, s[0:1]"),
            ("001382be", "s_ff1_i32_b32", "s_ff1_i32_b32 s2, s0"),
            ("001482be", "s_ff1_i32_b64", "s_ff1_i32_b64 s2, s[0:1]"),
            ("001582be", "s_flbit_i32_b32", "s_flbit_i32_b32 s2, s0"),
            ("001682be", "s_flbit_i32_b64", "s_flbit_i32_b64 s2, s[0:1]"),
            ("001f80be", "s_getpc_b64", "s_getpc_b64 s[0:1]"),
            ("002080be", "s_setpc_b64", "s_setpc_b64 s[0:1]"),
            ("002280be", "s_rfe_b64", "s_rfe_b64 s[0:1]"),
            ("0224febe", "s_and_saveexec_b64", "s_and_saveexec_b64 exec, s[2:3]"),
            ("023afebe", "s_andn2_wrexec_b64", "s_andn2_wrexec_b64 exec, s[2:3]"),
            ("013dfebe", "s_or_saveexec_b32", "s_or_saveexec_b32 exec_lo, s1"),
            ("0147febe", "s_andn2_wrexec_b32", "s_andn2_wrexec_b32 exec_lo, s1"),
            ("00008080", "s_sub_u32", "s_sub_u32 s0, s0, s0"),
            # GFX10 scalar operand codes: 108-123 are ttmp0-15, 125 is null (SOP2 ssrc1)
            ("007d8080", "s_sub_u32", "s_sub_u32 s0, s0, null"),
            ("006c8080", "s_sub_u32", "s_sub_u32 s0, s0, ttmp0"),
            # GFX10 / RDNA 2 opcode numbers, encodings as in LLVM's gfx10 MC tests
            ("01020080", "s_add_u32", "s_add_u32 s0, s1, s2"),
            ("01028080", "s_sub_u32", "s_sub_u32 s0, s1, s2"),
            ("01020081", "s_add_i32", "s_add_i32 s0, s1, s2"),
            ("01028081", "s_sub_i32", "s_sub_i32 s0, s1, s2"),
            ("01020083", "s_min_i32", "s_min_i32 s0, s1, s2"),
            ("01028083", "s_min_u32", "s_min_u32 s0, s1, s2"),
            ("01020084", "s_max_i32", "s_max_i32 s0, s1, s2"),
            ("01028084", "s_max_u32", "s_max_u32 s0, s1, s2"),
            ("01020085", "s_cselect_b32", "s_cselect_b32 s0, s1, s2"),
            ("01020087", "s_and_b32", "s_and_b32 s0, s1, s2"),
            ("01020088", "s_or_b32", "s_or_b32 s0, s1, s2"),
            ("01020089", "s_xor_b32", "s_xor_b32 s0, s1, s2"),
            ("0102008a", "s_andn2_b32", "s_andn2_b32 s0, s1, s2"),
            ("0102008f", "s_lshl_b32", "s_lshl_b32 s0, s1, s2"),
            ("01020090", "s_lshr_b32", "s_lshr_b32 s0, s1, s2"),
            ("01020091", "s_ashr_i32", "s_ashr_i32 s0, s1, s2"),
            ("01020093", "s_mul_i32", "s_mul_i32 s0, s1, s2"),
            ("01028093", "s_bfe_u32", "s_bfe_u32 s0, s1, s2"),
            ("01020094", "s_bfe_i32", "s_bfe_i32 s0, s1, s2"),
            ("02048094", "s_bfe_u64", "s_bfe_u64 s[0:1], s[2:3], s4"),
            ("02040095", "s_bfe_i64", "s_bfe_i64 s[0:1], s[2:3], s4"),
            ("0102809a", "s_mul_hi_u32", "s_mul_hi_u32 s0, s1, s2"),
            ("0102009b", "s_mul_hi_i32", "s_mul_hi_i32 s0, s1, s2"),
            ("341281b1", "s_cmpk_eq_i32", "s_cmpk_eq_i32 s1, 0x1234"),
            # SOPC: [31:23] == 0b101111110. These encodings were previously
            # consumed by the SOPK prefix check (e.g. 000100bf as s_sopk_1e).
            ("000100bf", "s_cmp_eq_i32", "s_cmp_eq_i32 s0, s1"),
            ("000106bf", "s_cmp_eq_u32", "s_cmp_eq_u32 s0, s1"),
            ("020302bf", "s_cmp_gt_i32", "s_cmp_gt_i32 s2, s3"),
            ("007d00bf", "s_cmp_eq_i32", "s_cmp_eq_i32 s0, null"),
            ("00010cbf", "s_bitcmp0_b32", "s_bitcmp0_b32 s0, s1"),
            ("000212bf", "s_cmp_eq_u64", "s_cmp_eq_u64 s[0:1], s[2:3]"),
            ("04060ebf", "s_bitcmp0_b64", "s_bitcmp0_b64 s[4:5], s6"),
            ("000080bb", "s_waitcnt_vscnt", "s_waitcnt_vscnt s0, 0x0"),
            ("0000fdbc", "s_waitcnt_expcnt", "s_waitcnt_expcnt null, 0x0"),
            ("0300fdbc", "s_waitcnt_expcnt", "s_waitcnt_expcnt null, 0x3"),
            ("0100a0bf", "s_inst_prefetch", "s_inst_prefetch 0x1"),
            ("0255027e", "v_rcp_f32_e32", "v_rcp_f32_e32 v1, v2"),
            ("010d007e", "v_cvt_f32_u32_e32", "v_cvt_f32_u32_e32 v0, v1"),
            ("0000027e", "v_nop", "v_nop"),
            # VOP1 9-bit SRC0: 0..255 scalar/constant, 256..511 VGPR
            ("0103007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, v1"),
            ("0102007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, s1"),
            ("f202047e", "v_mov_b32_e32", "v_mov_b32_e32 v2, 1.0"),
            ("f002067e", "v_mov_b32_e32", "v_mov_b32_e32 v3, 0.5"),
            ("8002087e", "v_mov_b32_e32", "v_mov_b32_e32 v4, 0"),
            # Special source operands 235-239 and 251-254, and the 1/(2*pi) inline
            # constant 248. Encodings and spellings from llvm-mc -mcpu=gfx1013.
            ("eb02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_shared_base"),
            ("ec02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_shared_limit"),
            ("ed02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_private_base"),
            ("ee02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_private_limit"),
            ("ef02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_pops_exiting_wave_id"),
            ("fb02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_vccz"),
            ("fc02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_execz"),
            ("fd02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_scc"),
            ("fe02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_lds_direct"),
            ("f8040606", "v_add_f32_e32", "v_add_f32_e32 v3, 0.15915494, v2"),
            ("fd0382be", "s_mov_b32", "s_mov_b32 s2, src_scc"),
            ("eb0482be", "s_mov_b64", "s_mov_b64 s[2:3], src_shared_base"),
            ("f80482be", "s_mov_b64", "s_mov_b64 s[2:3], 0.15915494309189532"),
            ("000064d5f8040200", "v_add_f64", "v_add_f64 v[0:1], 0.15915494309189532, v[2:3]"),
            ("000064d5ed040200", "v_add_f64", "v_add_f64 v[0:1], src_private_base, v[2:3]"),
            ("0105007e", "v_readfirstlane_b32", "v_readfirstlane_b32 s0, v1"),
            ("0c05587e", "v_readfirstlane_b32", "v_readfirstlane_b32 s44, v12"),
            # VOP2 9-bit SRC0 (VGPR vs SGPR vs inline constant) with GFX10/RDNA 2 opcodes
            ("01050602", "v_cndmask_b32_e32", "v_cndmask_b32_e32 v3, v1, v2, vcc_lo"),
            ("01050606", "v_add_f32_e32", "v_add_f32_e32 v3, v1, v2"),
            ("01040606", "v_add_f32_e32", "v_add_f32_e32 v3, s1, v2"),
            ("f0040606", "v_add_f32_e32", "v_add_f32_e32 v3, 0.5, v2"),
            ("01050610", "v_mul_f32_e32", "v_mul_f32_e32 v3, v1, v2"),
            ("0105062c", "v_lshrrev_b32_e32", "v_lshrrev_b32_e32 v3, v1, v2"),
            ("01050630", "v_ashrrev_i32_e32", "v_ashrrev_i32_e32 v3, v1, v2"),
            ("01050634", "v_lshlrev_b32_e32", "v_lshlrev_b32_e32 v3, v1, v2"),
            ("01050636", "v_and_b32_e32", "v_and_b32_e32 v3, v1, v2"),
            ("01050638", "v_or_b32_e32", "v_or_b32_e32 v3, v1, v2"),
            ("0105063a", "v_xor_b32_e32", "v_xor_b32_e32 v3, v1, v2"),
            ("0105064a", "v_add_nc_u32_e32", "v_add_nc_u32_e32 v3, v1, v2"),
            ("0105063e", "v_mac_f32_e32", "v_mac_f32_e32 v3, v1, v2"),
            ("01050656", "v_fmac_f32_e32", "v_fmac_f32_e32 v3, v1, v2"),
            ("01050a12", "v_mul_i32_i24_e32", "v_mul_i32_i24_e32 v5, v1, v2"),
            ("01050a14", "v_mul_hi_i32_i24_e32", "v_mul_hi_i32_i24_e32 v5, v1, v2"),
            ("01050a16", "v_mul_u32_u24_e32", "v_mul_u32_u24_e32 v5, v1, v2"),
            ("01050a18", "v_mul_hi_u32_u24_e32", "v_mul_hi_u32_u24_e32 v5, v1, v2"),
            ("01050a3c", "v_xnor_b32_e32", "v_xnor_b32_e32 v5, v1, v2"),
            # VOP1 rounding and byte conversion ops (GFX10 / RDNA 2)
            ("01230a7e", "v_cvt_f32_ubyte0_e32", "v_cvt_f32_ubyte0_e32 v5, v1"),
            ("01250a7e", "v_cvt_f32_ubyte1_e32", "v_cvt_f32_ubyte1_e32 v5, v1"),
            ("01270a7e", "v_cvt_f32_ubyte2_e32", "v_cvt_f32_ubyte2_e32 v5, v1"),
            ("01290a7e", "v_cvt_f32_ubyte3_e32", "v_cvt_f32_ubyte3_e32 v5, v1"),
            ("01410a7e", "v_fract_f32_e32", "v_fract_f32_e32 v5, v1"),
            ("01430a7e", "v_trunc_f32_e32", "v_trunc_f32_e32 v5, v1"),
            ("01450a7e", "v_ceil_f32_e32", "v_ceil_f32_e32 v5, v1"),
            ("01470a7e", "v_rndne_f32_e32", "v_rndne_f32_e32 v5, v1"),
            ("01490a7e", "v_floor_f32_e32", "v_floor_f32_e32 v5, v1"),
            # VOP2 mandatory literal FMA forms
            ("01070a5841312111", "v_fmamk_f32", "v_fmamk_f32 v5, v1, 0x11213141, v3"),
            ("01050a5a41312111", "v_fmaak_f32", "v_fmaak_f32 v5, v1, v2, 0x11213141"),
            # Non-fused literal forms (encodings from LLVM 15 llvm-mc -mcpu=gfx1013 -show-encoding)
            ("01070a4041312111", "v_madmk_f32", "v_madmk_f32 v5, v1, 0x11213141, v3"),
            ("01050a4241312111", "v_madak_f32", "v_madak_f32 v5, v1, v2, 0x11213141"),
            # VINTRP: attr is [15:10], chan is [9:8]
            ("000004c8", "v_interp_p1_f32_e32", "v_interp_p1_f32_e32 v1, v0, attr0.x"),
            ("000108c8", "v_interp_p1_f32_e32", "v_interp_p1_f32_e32 v2, v0, attr0.y"),
            ("00020cc8", "v_interp_p1_f32_e32", "v_interp_p1_f32_e32 v3, v0, attr0.z"),
            ("000310c8", "v_interp_p1_f32_e32", "v_interp_p1_f32_e32 v4, v0, attr0.w"),
            ("01050dc8", "v_interp_p2_f32_e32", "v_interp_p2_f32_e32 v3, v1, attr1.y"),
            # EXP: en=[3:0], target=[9:4], compr=10, done=11, vm=12
            ("000000f800000000", "exp", "exp mrt0 off, off, off, off"),
            ("010800f804000000", "exp", "exp mrt0 v4, off, off, off done"),
            ("8f0800f804030201", "exp", "exp mrtz v4, v3, v2, v1 done"),
            ("cf0000f804030201", "exp", "exp pos0 v4, v3, v2, v1"),
            ("0f1c00f810110000", "exp", "exp mrt0 v16, v16, v17, v17 done compr vm"),
            # SMEM: register widths per op and soffset vs immediate offset
            ("01050cf4000000fa", "s_load_dwordx8", "s_load_dwordx8 s[20:27], s[2:3], 0x0"),
            ("01050cf445230100", "s_load_dwordx8", "s_load_dwordx8 s[20:27], s[2:3], s0 offset:0x12345"),
            ("420120f400000000", "s_buffer_load_dword", "s_buffer_load_dword s5, s[4:7], s0"),
            ("020528f4200000fa", "s_buffer_load_dwordx4", "s_buffer_load_dwordx4 s[20:23], s[4:7], 0x20"),
            # MIMG: contiguous and NSA encodings
            ("001f00f000000000", "image_load", "image_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm"),
            ("080100f002000000", "image_load", "image_load v0, v[2:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D"),
            ("000780f020402103", "image_sample", "image_sample v[64:66], v32, s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D"),
            ("220790f02040210310080000", "image_sample_l", "image_sample_l v[64:66], [v32, v16, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY"),
            ("220794f02040210310080000", "image_sample_b", "image_sample_b v[64:66], [v32, v16, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY"),
            ("22079cf02040210310000000", "image_sample_lz", "image_sample_lz v[64:66], [v32, v16], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY"),
            # GFX10 VOPC vector comparisons
            ("0105027c", "v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc_lo, v1, v2"),
            ("0104027c", "v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc_lo, s1, v2"),
            ("2510027c", "v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc_lo, s37, v8"),
            ("ff04027c0000803f", "v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v2"),
            ("0105227c", "v_cmpx_lt_f32_e32", "v_cmpx_lt_f32_e32 v1, v2"),
            # GFX10 VOP3A and VOP3B 64-bit vector ALU instructions
            ("000041d501050e04", "v_mad_f32", "v_mad_f32 v0, v1, v2, v3"),
            ("00004bd501050e04", "v_fma_f32", "v_fma_f32 v0, v1, v2, v3"),
            ("0c8008d50ce50100", "v_mul_f32_e64", "v_mul_f32_e64 v12, v12, 1.0 clamp"),
            ("000001d42c240260", "v_cmp_lt_f32_e64", "v_cmp_lt_f32_e64 s0, -s44, -v18"),
            ("0e0001d580e40100", "v_cndmask_b32_e64", "v_cndmask_b32_e64 v14, 0, 1.0, s0"),
            ("01000fd702070200", "v_add_co_u32", "v_add_co_u32 v1, s0, v2, v3"),
            ("05004bd501ff09040000803f", "v_fma_f32", "v_fma_f32 v5, v1, 0x3f800000, v2"),
            ("030103d501050208", "v_add_f32_e64", "v_add_f32_e64 v3, |v1|, v2 mul:2"),
            ("340889d42c240200", "v_cmp_lt_i16_e64", "v_cmp_lt_i16_e64 s52, s44, v18 op_sel:[1,0,0]"),
            ("fe0064d501050202", "v_add_f64", "v_add_f64 v[254:255], v[1:2], v[2:3]"),
            ("0200ffd683080200", "v_lshlrev_b64", "v_lshlrev_b64 v[2:3], 3, v[4:5]"),
            ("000076d502061204", "v_mad_u64_u32", "v_mad_u64_u32 v[0:1], s0, s2, v3, v[4:5]"),
            ("000090d502010000", "v_cvt_f64_f32_e64", "v_cvt_f64_f32_e64 v[0:1], v2"),
            ("00008fd502010000", "v_cvt_f32_f64_e64", "v_cvt_f32_f64_e64 v0, v[2:3]"),
            ("000080d502091a04", "v_nop", "v_nop"),
            ("0000abd502091a04", "v_rcp_iflag_f32_e64", "v_rcp_iflag_f32_e64 v0, v2"),
            ("0000afd502091a04", "v_rcp_f64_e64", "v_rcp_f64_e64 v[0:1], v[2:3]"),
            ("0000b1d502091a04", "v_rsq_f64_e64", "v_rsq_f64_e64 v[0:1], v[2:3]"),
            ("0000b4d502091a04", "v_sqrt_f64_e64", "v_sqrt_f64_e64 v[0:1], v[2:3]"),
            ("0000b9d502091a04", "v_ffbh_u32_e64", "v_ffbh_u32_e64 v0, v2"),
            ("0000bad502091a04", "v_ffbl_b32_e64", "v_ffbl_b32_e64 v0, v2"),
            ("0000bbd502091a04", "v_ffbh_i32_e64", "v_ffbh_i32_e64 v0, v2"),
            ("000040d502091a04", "v_mad_legacy_f32", "v_mad_legacy_f32 v0, v2, v4, v6"),
            ("000050d502091a04", "v_mullit_f32", "v_mullit_f32 v0, v2, v4, v6"),
            ("010028d502060000", "v_add_co_ci_u32_e64", "v_add_co_ci_u32_e64 v1, s0, s2, s3, s0"),
            # GFX10 VOP3P packed float16, integer16, dot-products, and mixed-precision FMA
            ("04400acc03030118", "v_pk_add_u16", "v_pk_add_u16 v4, v3, 1"),
            ("04000acc03030118", "v_pk_add_u16", "v_pk_add_u16 v4, v3, 1"),
            ("1a8012cc18e10118", "v_pk_max_f16", "v_pk_max_f16 v26, v24, 0.5 clamp"),
            ("0c4013cc12275a1c", "v_dot2_f32_f16", "v_dot2_f32_f16 v12, v18, v19, v22"),
            ("0c0020cc2c265204", "v_fma_mix_f32", "v_fma_mix_f32 v12, s44, v19, v20"),
            # MUBUF: untyped, format, d16, atomics, cache invalidation, and buffer modifiers
            ("000030e000050203", "buffer_load_dword", "buffer_load_dword v5, off, s[8:11], s3"),
            ("40d030e000054203", "buffer_load_dword", "buffer_load_dword v5, v0, s[8:11], s3 offen offset:64 glc slc dlc"),
            ("ff3f30e000050280", "buffer_load_dword", "buffer_load_dword v5, v[0:1], s[8:11], 0 idxen offen offset:4095"),
            ("000031e000000203", "buffer_load_dword", "buffer_load_dword off, s[8:11], s3 lds"),
            ("000030e00005827d", "buffer_load_dword", "buffer_load_dword v5, off, s[8:11], null tfe"),
            ("000030e000051b03", "buffer_load_dword", "buffer_load_dword v5, off, ttmp[0:3], s3"),
            ("102038e00008037c", "buffer_load_dwordx4", "buffer_load_dwordx4 v[8:11], v0, s[12:15], m0 idxen offset:16"),
            ("00000ce200050280", "buffer_load_format_d16_xyzw", "buffer_load_format_d16_xyzw v[5:6], off, s[8:11], 0"),
            ("0c1070e000050203", "buffer_store_dword", "buffer_store_dword v5, v0, s[8:11], s3 offen offset:12"),
            ("001078e004000280", "buffer_store_dwordx4", "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
            ("000010e200050203", "buffer_store_format_d16_x", "buffer_store_format_d16_x v5, off, s[8:11], s3"),
            ("0050c8e000050203", "buffer_atomic_add", "buffer_atomic_add v5, v0, s[8:11], s3 offen glc"),
            ("0010c4e000050203", "buffer_atomic_cmpswap", "buffer_atomic_cmpswap v[5:6], v0, s[8:11], s3 offen"),
            ("005044e100050280", "buffer_atomic_cmpswap_x2", "buffer_atomic_cmpswap_x2 v[5:8], v0, s[8:11], 0 offen glc"),
            ("0000c4e100000000", "buffer_gl0_inv", "buffer_gl0_inv"),
            ("0000c8e100000000", "buffer_gl1_inv", "buffer_gl1_inv"),
            # MTBUF: typed buffer loads/stores and format field decoding
            ("0000b0e800050203", "tbuffer_load_format_x", "tbuffer_load_format_x v5, off, s[8:11], s3 format:[BUF_FMT_32_FLOAT]"),
            ("40d06bea00054203", "tbuffer_load_format_xyzw", "tbuffer_load_format_xyzw v[5:8], v0, s[8:11], s3 format:[BUF_FMT_32_32_32_32_FLOAT] offen offset:64 glc slc dlc"),
            ("0030ede800052280", "tbuffer_store_format_d16_xy", "tbuffer_store_format_d16_xy v5, v[0:1], s[8:11], 0 format:[BUF_FMT_16_16_FLOAT] idxen offen"),
            ("000008e800050203", "tbuffer_load_format_x", "tbuffer_load_format_x v5, off, s[8:11], s3"),
            ("0000f8eb00050203", "tbuffer_load_format_x", "tbuffer_load_format_x v5, off, s[8:11], s3 format:127"),
            # FLAT, GLOBAL, and SCRATCH memory instructions
            ("000030dc02007d04", "flat_load_dword", "flat_load_dword v4, v[2:3]"),
            ("ff173bdc02007d04", "flat_load_dwordx4", "flat_load_dwordx4 v[4:7], v[2:3] offset:2047 glc slc dlc"),
            ("100088dc02007d04", "flat_load_sbyte_d16", "flat_load_sbyte_d16 v4, v[2:3] offset:16"),
            ("100070dc02087d00", "flat_store_dword", "flat_store_dword v[2:3], v8 offset:16"),
            ("00007cdc02087d00", "flat_store_dwordx3", "flat_store_dwordx3 v[2:3], v[8:10]"),
            ("0000c8dc02087d00", "flat_atomic_add", "flat_atomic_add v[2:3], v8"),
            ("0000c9dc02087d04", "flat_atomic_add", "flat_atomic_add v4, v[2:3], v8 glc"),
            ("2000c5dc02087d04", "flat_atomic_cmpswap", "flat_atomic_cmpswap v4, v[2:3], v[8:9] offset:32 glc"),
            ("000045dd02087d04", "flat_atomic_cmpswap_x2", "flat_atomic_cmpswap_x2 v[4:5], v[2:3], v[8:11] glc"),
            ("008030dc02007d04", "global_load_dword", "global_load_dword v4, v[2:3], off"),
            ("f09f33dc01000204", "global_load_dword", "global_load_dword v4, v1, s[2:3] offset:-16 glc slc dlc"),
            ("00a030dc02007d00", "global_load_dword", "global_load_dword v[2:3], off lds"),
            ("00a030dc01000200", "global_load_dword", "global_load_dword v1, s[2:3] lds"),
            ("ff8770dc02087d00", "global_store_dword", "global_store_dword v[2:3], v8, off offset:2047"),
            ("008864dc01080200", "global_store_byte_d16_hi", "global_store_byte_d16_hi v1, v8, s[2:3] offset:-2048"),
            ("0080c8dc01080200", "global_atomic_add", "global_atomic_add v1, v8, s[2:3]"),
            ("0080c9dc01080204", "global_atomic_add", "global_atomic_add v4, v1, v8, s[2:3] glc"),
            ("008045dd02087d04", "global_atomic_cmpswap_x2", "global_atomic_cmpswap_x2 v[4:5], v[2:3], v[8:11], off glc"),
            ("104030dc01007d04", "scratch_load_dword", "scratch_load_dword v4, v1, off offset:16"),
            ("f05f33dc00000204", "scratch_load_dword", "scratch_load_dword v4, off, s2 offset:-16 glc slc dlc"),
            ("106030dc00000200", "scratch_load_dword", "scratch_load_dword off, s2 offset:16 lds"),
            ("004020dc00007c04", "scratch_load_ubyte", "scratch_load_ubyte v4, off, m0"),
            ("104070dc01087d00", "scratch_store_dword", "scratch_store_dword v1, v8, off offset:16"),
            ("e04f74dc00080200", "scratch_store_dwordx2", "scratch_store_dwordx2 off, v[8:9], s2 offset:-32"),
            ("000050d800000000", "ds_nop", "ds_nop"),
            ("0000d8d802000018", "ds_read_b32", "ds_read_b32 v24, v2"),
            ("4000d8d802000018", "ds_read_b32", "ds_read_b32 v24, v2 offset:64"),
            ("0400e8d802000018", "ds_read_u8", "ds_read_u8 v24, v2 offset:4"),
            ("1000d8d902000018", "ds_read_b64", "ds_read_b64 v[24:25], v2 offset:16"),
            ("0000fcdb02000018", "ds_read_b128", "ds_read_b128 v[24:27], v2"),
            ("0408dcd802000018", "ds_read2_b32", "ds_read2_b32 v[24:25], v2 offset0:4 offset1:8"),
            ("0102e0d902000018", "ds_read2st64_b64", "ds_read2st64_b64 v[24:27], v2 offset0:1 offset1:2"),
            ("000034d802080000", "ds_write_b32", "ds_write_b32 v2, v8"),
            ("800034d802080000", "ds_write_b32", "ds_write_b32 v2, v8 offset:128"),
            ("040838d802081000", "ds_write2_b32", "ds_write2_b32 v2, v8, v16 offset0:4 offset1:8"),
            ("20007cdb02080000", "ds_write_b128", "ds_write_b128 v2, v[8:11] offset:32"),
            ("080000d802080000", "ds_add_u32", "ds_add_u32 v2, v8 offset:8"),
            ("080082d802080018", "ds_add_rtn_u32", "ds_add_rtn_u32 v24, v2, v8 offset:8 gds"),
            ("100030d802081000", "ds_mskor_b32", "ds_mskor_b32 v2, v8, v16 offset:16"),
            ("0102b8d802081018", "ds_wrxchg2_rtn_b32", "ds_wrxchg2_rtn_b32 v[24:25], v2, v8, v16 offset0:1 offset1:2"),
            ("0000d4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2"),
            ("1b80d4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2 offset:swizzle(QUAD_PERM,3,2,1,0)"),
            ("1f40d4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2 offset:swizzle(SWAP,16)"),
            ("1f1cd4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2 offset:swizzle(REVERSE,8)"),
            ("5c00d4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2 offset:swizzle(BROADCAST,4,2)"),
            ("0103d4d802000018", "ds_swizzle_b32", 'ds_swizzle_b32 v24, v2 offset:swizzle(BITMASK_PERM,"1100p")'),
            ("0c00fad800000018", "ds_append", "ds_append v24 offset:12 gds"),
            ("0000f4d800000018", "ds_consume", "ds_consume v24"),
            ("0400fed802000018", "ds_ordered_count", "ds_ordered_count v24, v2 offset:4 gds"),
            ("000062d800000000", "ds_gws_sema_release_all", "ds_gws_sema_release_all gds"),
            ("080076d802000000", "ds_gws_barrier", "ds_gws_barrier v2 offset:8 gds"),
            ("0400c8da02080018", "ds_permute_b32", "ds_permute_b32 v24, v2, v8 offset:4"),
            ("0000ccda02080018", "ds_bpermute_b32", "ds_bpermute_b32 v24, v2, v8"),
            ("1000c0da00080000", "ds_write_addtid_b32", "ds_write_addtid_b32 v8 offset:16"),
            ("1000c4da00000018", "ds_read_addtid_b32", "ds_read_addtid_b32 v24 offset:16"),
            ("200054da02000000", "ds_add_src2_f32", "ds_add_src2_f32 v2 offset:32"),
            ("0000f8d902080018", "ds_condxchg32_rtn_b64", "ds_condxchg32_rtn_b64 v[24:25], v2, v[8:9]"),
            ("050060d701050002", "v_readlane_b32", "v_readlane_b32 s5, v1, s2"),
            ("050061d780040002", "v_writelane_b32", "v_writelane_b32 v5, 0, s2"),
        ]
        for hex_str, expected_op, expected_asm in vectors:
            raw = bytes.fromhex(hex_str)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed for {hex_str}")
            self.assertEqual(insns[0]["opcode"], expected_op, f"Mismatched opcode for {hex_str}")
            self.assertEqual(insns[0]["assembly"], expected_asm, f"Mismatched assembly for {hex_str}")

    def test_compressed_export_keeps_unpaired_mask(self):
        base = bytes.fromhex("0f1c00f810110000")
        operands = {
            0x2: "off, off, off, off",
            0xa: "off, off, off, off",
            0x5: "v16, v16, v17, v17",
        }
        for mask, registers in operands.items():
            with self.subTest(mask=mask):
                raw = bytes([(base[0] & 0xf0) | mask]) + base[1:]
                records = disasm_rdna2(raw)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["assembly"],
                                 f"exp mrt0 {registers} done compr vm en:{mask:#x}")

    def test_waitcnt_uses_gfx10_counter_fields(self):
        # GFX10 s_waitcnt simm16: vmcnt [3:0] and [15:14], expcnt [6:4], lgkmcnt [13:8].
        vectors = [
            # Printed by LLVM 15.0.7 (MC/Disassembler/AMDGPU/gfx10_dasm_all.txt).
            ("00008cbf", "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"),
            ("34128cbf", "s_waitcnt vmcnt(4) expcnt(3) lgkmcnt(18)"),
            ("51c18cbf", "s_waitcnt vmcnt(49) expcnt(5) lgkmcnt(1)"),
            # Encoded by LLVM 15.0.7 (MC/AMDGPU/gfx10_asm_sop.s); bit 7 is unused.
            ("d1c18cbf", "s_waitcnt vmcnt(49) expcnt(5) lgkmcnt(1)"),
            # One counter each; the others at their maximum are omitted.
            ("703f8cbf", "s_waitcnt vmcnt(0)"),
            ("707f8cbf", "s_waitcnt vmcnt(16)"),
            ("7f3f8cbf", "s_waitcnt vmcnt(15)"),
            ("0fff8cbf", "s_waitcnt expcnt(0)"),
            ("7fc08cbf", "s_waitcnt lgkmcnt(0)"),
            ("7fff8cbf", "s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(63)"),
        ]
        for hex_str, expected in vectors:
            with self.subTest(encoding=hex_str):
                records = disasm_rdna2(bytes.fromhex(hex_str + "000081bf"))
                self.assertEqual([(r["opcode"], r["length"]) for r in records],
                                 [("s_waitcnt", 4), ("s_endpgm", 4)])
                self.assertEqual(records[0]["assembly"], expected)
                self.assertTrue(is_instruction_supported(expected))

    def test_literal_operands_keep_stream_aligned(self):
        # SOP1, SOP2, VOP1, VOP2 (src0 == 255) and v_fmaak_f32 each consume a trailing 32-bit literal.
        stream = bytes.fromhex(
            "ff0382be 78563412"  # s_mov_b32 s2, 0x12345678
            "02ff0380 00010000"  # s_add_u32 s3, s2, 0x100
            "ff02087e 89674523"  # v_mov_b32_e32 v4, 0x23456789
            "ff040606 0000803f"  # v_add_f32_e32 v3, 0x3f800000, v2
            "01050a5a 41312111"  # v_fmaak_f32 v5, v1, v2, 0x11213141
            "ff04027c 0000803f"  # v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v2
            "05004bd5 01ff0904 0000803f"  # v_fma_f32 v5, v1, 0x3f800000, v2
            "000081bf"           # s_endpgm
        )
        insns = disasm_rdna2(stream)
        self.assertEqual(
            [(i["opcode"], i["length"], i["assembly"]) for i in insns],
            [
                ("s_mov_b32", 8, "s_mov_b32 s2, 0x12345678"),
                ("s_add_u32", 8, "s_add_u32 s3, s2, 0x100"),
                ("v_mov_b32_e32", 8, "v_mov_b32_e32 v4, 0x23456789"),
                ("v_add_f32_e32", 8, "v_add_f32_e32 v3, 0x3f800000, v2"),
                ("v_fmaak_f32", 8, "v_fmaak_f32 v5, v1, v2, 0x11213141"),
                ("v_cmp_lt_f32_e32", 8, "v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v2"),
                ("v_fma_f32", 12, "v_fma_f32 v5, v1, 0x3f800000, v2"),
                ("s_endpgm", 4, "s_endpgm"),
            ],
        )
        # Truncated literals and truncated NSA MIMG must report *_truncated instead of crashing.
        for bad_hex, expected_op in (
            ("ff0382be", "sop1_truncated"),
            ("02ff0380", "sop2_truncated"),
            ("ff02087e", "vop1_truncated"),
            ("01050a5a", "vop2_truncated"),
            ("ff04027c", "vopc_truncated"),
            ("05004bd501ff0904", "vop3_truncated"),
            ("00004bd5", "vop3_truncated"),
            ("220790f020402103", "mimg_truncated"),
        ):
            insns = disasm_rdna2(bytes.fromhex(bad_hex))
            self.assertEqual(len(insns), 1)
            self.assertEqual(insns[0]["opcode"], expected_op)
            self.assertIn("<truncated>", insns[0]["assembly"])

    def test_vop1_readfirstlane_b32_destination_is_an_sgpr(self):
        # Issue #746: in VOP1 the vdst field of v_readfirstlane_b32 names an SGPR
        # even though the mnemonic starts with "v_". LLVM for gfx1013 (triple
        # amdgcn--amdpal) assembles `v_readfirstlane_b32 s2, v0` to this encoding
        # and disassembles those bytes back to `v_readfirstlane_b32 s2, v0`.
        record = disasm_rdna2(bytes.fromhex("0005047e"))[0]
        self.assertEqual(record["opcode"], "v_readfirstlane_b32")
        self.assertEqual(record["length"], 4)
        self.assertEqual(record["assembly"], "v_readfirstlane_b32 s2, v0")

        # The plain register form is the only shape the target assembles:
        # `v_readfirstlane_b32 v2, v0` and `v_readfirstlane_b32 s2, s0` are both
        # rejected with "invalid operand for instruction", so the source stays a
        # VGPR and only the destination class is corrected.

        # Keep a following s_endpgm in both controls.  The VOP1 control checks
        # that the 32-bit instruction still ends at +4.
        control = bytes.fromhex(
            "0005047e"  # v_readfirstlane_b32 s2, v0
            "000081bf"  # s_endpgm
        )
        self.assertEqual(
            [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in disasm_rdna2(control)],
            [
                (0, 4, "v_readfirstlane_b32", "v_readfirstlane_b32 s2, v0"),
                (4, 4, "s_endpgm", "s_endpgm"),
            ],
        )

        # An intervening 32-bit instruction is an alignment control as well.
        stream = bytes.fromhex(
            "0005047e"  # v_readfirstlane_b32 s2, v0
            "0103007e"  # v_mov_b32_e32 v0, v1
            "000081bf"  # s_endpgm
        )
        self.assertEqual(
            [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in disasm_rdna2(stream)],
            [
                (0, 4, "v_readfirstlane_b32", "v_readfirstlane_b32 s2, v0"),
                (4, 4, "v_mov_b32_e32", "v_mov_b32_e32 v0, v1"),
                (8, 4, "s_endpgm", "s_endpgm"),
            ],
        )

        # Negative control: vdst == 2 in an ordinary VOP1 is still a VGPR.
        self.assertEqual(
            disasm_rdna2(bytes.fromhex("0103047e"))[0]["assembly"],
            "v_mov_b32_e32 v2, v1",
        )

        # VOP3A control: gfx1013 does not assemble a VOP3A shape of this
        # operation (`v_readfirstlane_b32_e64 s2, v0` fails with "e64 variant of
        # this instruction is not supported") and llvm-mc reports the issue's
        # bytes 020082d500010000 as an invalid instruction encoding.  The record
        # keeps a distinct unsupported-encoding diagnostic, consumes exactly
        # eight bytes, is neither recognized nor supported by the scanner (so a
        # lowerer does not receive an instruction), and the following s_endpgm
        # still decodes at +8.
        vop3a = disasm_rdna2(bytes.fromhex(
            "020082d500010000"  # proposed VOP3A word from issue #746 (invalid on gfx1013)
            "000081bf"          # s_endpgm
        ))
        self.assertEqual(len(vop3a), 2)
        self.assertEqual((vop3a[0]["offset"], vop3a[0]["length"]), (0, 8))
        self.assertEqual(vop3a[0]["opcode"], "v_readfirstlane_b32_unsupported")
        self.assertEqual(vop3a[0]["assembly"],
                         "v_readfirstlane_b32 <unsupported VOP3A encoding>")
        self.assertNotEqual(vop3a[0]["opcode"], "v_readfirstlane_b32")
        self.assertFalse(is_opcode_recognized(vop3a[0]["opcode"]))
        self.assertFalse(is_instruction_supported(vop3a[0]["assembly"]))
        self.assertEqual(
            (vop3a[1]["offset"], vop3a[1]["length"], vop3a[1]["opcode"]),
            (8, 4, "s_endpgm"),
        )

    def test_sopc_literal_keeps_the_following_instruction_aligned(self):
        # src1 == literal (255) makes the SOPC encoding 8 bytes. Without
        # consuming the literal dword, the following s_endpgm would be misread.
        raw = bytes.fromhex("00ff06bf 78563412 000081bf")
        insns = disasm_rdna2(raw)
        self.assertEqual(
            [(i["opcode"], i["length"], i["assembly"]) for i in insns],
            [
                ("s_cmp_eq_u32", 8, "s_cmp_eq_u32 s0, 0x12345678"),
                ("s_endpgm", 4, "s_endpgm"),
            ],
        )

    def test_interp_f16_uses_llvm_spelling(self):
        """Issue #938 step 2: the VOP3 binary16 interpolation forms decode as LLVM 15 prints them for gfx1013.

        Each expected string is llvm-mc 15.0.7's text for the same bytes. Encodings LLVM 15 refuses keep their
        eight-byte record and are reported as unsupported instead of as the instruction.
        """
        endpgm = bytes.fromhex("000081bf")
        cases = (
            ("010042d700040200", "v_interp_p1ll_f16 v1, v2, attr0.x"),
            ("010042d7c5050200", "v_interp_p1ll_f16 v1, v2, attr5.w high"),
            ("018242d75f040058", "v_interp_p1ll_f16 v1, -|s2|, attr31.y clamp div:2"),
            ("010043d741040e94", "v_interp_p1lv_f16 v1, v2, attr1.y, -v3 mul:4"),
            ("01825ad7ff05fe07", "v_interp_p2_f16 v1, |v2|, attr63.w, v255 high clamp"),
            ("ff005ad7a0d2f001", "v_interp_p2_f16 v255, s105, attr32.z, m0"),
            # An inline constant source: LLVM 15 prints it as an invalid immediate, and a lone negation as neg().
            ("010043d741020d04", "v_interp_p1lv_f16 v1, /*invalid immediate*/, attr1.y, v3"),
            ("010043d741020d44", "v_interp_p1lv_f16 v1, neg(/*invalid immediate*/), attr1.y, v3"),
        )
        for raw, assembly in cases:
            with self.subTest(raw=raw):
                insns = disasm_rdna2(bytes.fromhex(raw) + endpgm)
                self.assertEqual([(i["length"], i["assembly"]) for i in insns], [(8, assembly), (4, "s_endpgm")])
                self.assertEqual(insns[0]["opcode"], assembly.split()[0])
        refused = (
            ("010042d700040e00", "v_interp_p1ll_f16"),  # p1ll with a third source
            ("010142d700040200", "v_interp_p1ll_f16"),  # abs on the attribute operand
            ("010043d741040e24", "v_interp_p1lv_f16"),  # neg on the attribute operand
            ("01005ad782040e0c", "v_interp_p2_f16"),    # output modifier on p2
            ("01005ad782fe0d04", "v_interp_p2_f16"),    # literal source
        )
        for raw, name in refused:
            with self.subTest(refused=raw):
                insns = disasm_rdna2(bytes.fromhex(raw) + endpgm)
                self.assertEqual([(i["offset"], i["length"], i["opcode"]) for i in insns],
                                 [(0, 8, f"{name}_unsupported"), (8, 4, "s_endpgm")])

    def test_hwreg_operands_use_llvm_spelling(self):
        """Issue #938: hwreg operands and s_setreg_imm32_b32 literals print as LLVM 15 prints them for gfx1013.

        Each expected string is LLVM 15.0.7's disassembly (pixel_aot.disassemble, gfx1013) of the same bytes.
        """
        endpgm = bytes.fromhex("000081bf")
        cases = (
            ("810880ba03000000", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"),
            ("810880ba00000000", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0"),
            ("14f880ba00000000", "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0"),
            ("15f880ba00040000", "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_HI), 0x400"),
            ("011980baffffffff", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), -1"),
            ("04f880ba41000000", "s_setreg_imm32_b32 hwreg(4), 0x41"),
            ("810880baf0ffffff", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), -16"),
            ("810880baefffffff", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0xffffffef"),
            ("810880ba40000000", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 64"),
            ("810880ba00000080", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0x80000000"),
            ("810880ba0000803f", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1.0"),
            ("810880ba83f9223e", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0.15915494"),
            ("810880ba000080c0", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), -4.0"),
            ("01f804b9", "s_getreg_b32 s4, hwreg(HW_REG_MODE)"),
            ("171a05b9", "s_getreg_b32 s5, hwreg(HW_REG_HW_ID1, 8, 4)"),
            ("810884b9", "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s4"),
            ("ff07fcb9", "s_setreg_b32 hwreg(63, 31, 1), m0"),
        )
        for raw, assembly in cases:
            with self.subTest(raw=raw):
                insns = disasm_rdna2(bytes.fromhex(raw) + endpgm)
                self.assertEqual([i["assembly"] for i in insns], [assembly, "s_endpgm"])

        # The shader scan recognizes the MODE rounding and FLAT_SCRATCH writes only in LLVM's spelling, so the
        # decoded forms must reach is_instruction_supported in that spelling.
        for raw in ("810880ba03000000", "810880ba00000000", "14f880ba00000000", "15f880ba00040000"):
            with self.subTest(supported=raw):
                line = disasm_rdna2(bytes.fromhex(raw))[0]["assembly"]
                self.assertTrue(is_instruction_supported(line), line)
        # Controls: the previous raw spelling, and a directed rounding mode the translator refuses.
        self.assertFalse(is_instruction_supported("s_setreg_imm32_b32 0x881, 0x3"))
        directed = disasm_rdna2(bytes.fromhex("810880ba01000000"))[0]["assembly"]
        self.assertEqual(directed, "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1")
        self.assertFalse(is_instruction_supported(directed), directed)

    def test_hwreg_truncated_literal_and_endpgm_boundaries(self):
        """Issue #938 follow-up: SOPK hwreg truncation and s_endpgm boundary controls."""
        # 4-byte truncated s_setreg_imm32_b32 (missing literal dword) emits sopk_truncated cleanly.
        trunc4 = disasm_rdna2(bytes.fromhex("810880ba"))
        self.assertEqual(
            [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in trunc4],
            [(0, 4, "sopk_truncated", "s_setreg_imm32_b32 <truncated>")],
        )
        # 6-byte stream (partial second dword) still flags sopk_truncated and emits the remaining 2 bytes as data.
        trunc6 = disasm_rdna2(bytes.fromhex("810880ba0300"))
        self.assertEqual(
            [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in trunc6],
            [
                (0, 4, "sopk_truncated", "s_setreg_imm32_b32 <truncated>"),
                (4, 2, "data", ".byte 0300"),
            ],
        )
        # Mid-stream truncation after a valid 4-byte s_getreg_b32 keeps the first record intact.
        mid_trunc = disasm_rdna2(bytes.fromhex("01f804b9" + "810880ba"))
        self.assertEqual(
            [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in mid_trunc],
            [
                (0, 4, "s_getreg_b32", "s_getreg_b32 s4, hwreg(HW_REG_MODE)"),
                (4, 4, "sopk_truncated", "s_setreg_imm32_b32 <truncated>"),
            ],
        )
        # Boundary control: a 32-bit literal equal to s_endpgm (0xbf810000) belongs to
        # s_setreg_imm32_b32.  Trailing bytes after s_endpgm are still decoded and marked
        # with after_endpgm=True (matches LLVM 15 decode-through-end behaviour).
        stream = bytes.fromhex(
            "810880ba000081bf"  # s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0xbf810000
            "01f804b9"          # s_getreg_b32 s4, hwreg(HW_REG_MODE)
            "000081bf"          # s_endpgm
            "810884b9"          # trailing instruction decoded past s_endpgm
        )
        insns = disasm_rdna2(stream)
        self.assertEqual(
            [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in insns],
            [
                (0, 8, "s_setreg_imm32_b32", "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0xbf810000"),
                (8, 4, "s_getreg_b32", "s_getreg_b32 s4, hwreg(HW_REG_MODE)"),
                (12, 4, "s_endpgm", "s_endpgm"),
                (16, 4, "s_setreg_b32", "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s4"),
            ],
        )

    def test_s_code_end_is_recognized_but_not_translated(self):
        instruction = disasm_rdna2(bytes.fromhex("00009fbf"))[0]
        self.assertEqual((instruction["opcode"], instruction["assembly"]), ("s_code_end", "s_code_end"))
        self.assertTrue(is_opcode_recognized(instruction["opcode"]))
        self.assertFalse(is_instruction_supported(instruction["assembly"]))

    def test_binary_scan_counts_records_after_s_endpgm(self):
        code = bytes.fromhex("000081bf0403067e000081bf")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ps_after_end.bin"
            path.write_bytes(code)
            result, = scan_directory_or_files([path])
        self.assertEqual(result["total_instructions"], 3)
        self.assertEqual(result["opcode_distribution"],
                         {"s_endpgm": 2, "v_mov_b32_e32": 1})
        self.assertTrue(result["opcodes_recognized"])
        self.assertFalse(result["translator_accepted"])

    def test_source_built_records_after_s_endpgm(self):
        """Decode and scan the complete supplied span, including records after the first program end."""
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])
        if not assembler or not objcopy:
            self.fail(
                f"post-endpgm source fixture requires llvm-mc and llvm-objcopy "
                f"(assembler={assembler}, objcopy={objcopy})"
            )

        closed = assemble_with_llvm_mc(["v_mov_b32_e32 v1, v2", "s_endpgm"], assembler, objcopy)
        closed_records = disasm_rdna2(closed)
        self.assertEqual(
            [(record["offset"], record["length"], record["assembly"]) for record in closed_records],
            [(0, 4, "v_mov_b32_e32 v1, v2"), (4, 4, "s_endpgm")],
        )
        self.assertEqual(llvm_disassemble(closed, resolved_lib, offsets=True),
                         [(record["offset"], record["assembly"]) for record in closed_records])

        boundary = assemble_with_llvm_mc(
            [
                "s_branch after_first_exit",
                "s_endpgm",
                "v_mov_b32_e32 v1, v2",
                "s_code_end",
                "after_first_exit:",
                "s_endpgm",
                ".long 0",
            ],
            assembler,
            objcopy,
        )
        records = disasm_rdna2(boundary)
        expected = [
            (0, 4, "s_branch", "s_branch label_0x10"),
            (4, 4, "s_endpgm", "s_endpgm"),
            (8, 4, "v_mov_b32_e32", "v_mov_b32_e32 v1, v2"),
            (12, 4, "s_code_end", "s_code_end"),
            (16, 4, "s_endpgm", "s_endpgm"),
            (20, 4, "word_0x00000000", ".long 0x00000000"),
        ]
        self.assertEqual(
            [(record["offset"], record["length"], record["opcode"], record["assembly"]) for record in records],
            expected,
        )
        llvm_prefix = llvm_disassemble(boundary[:-4], resolved_lib, offsets=True)
        self.assertEqual([offset for offset, _ in llvm_prefix], [0, 4, 8, 12, 16])
        self.assertEqual(
            [(offset, line) for offset, line in llvm_prefix if offset != 0],
            [(offset, assembly) for offset, _, _, assembly in expected[1:5]],
        )
        with self.assertRaisesRegex(ValueError, "at byte 20$"):
            llvm_disassemble(boundary, resolved_lib)

        self.assertTrue(is_opcode_recognized("s_code_end"))
        self.assertFalse(is_instruction_supported("s_code_end"))
        scan_source = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
            "v_mov_b32_e32 v1, v2",
            "s_endpgm",
        ]
        scan_code = assemble_with_llvm_mc(scan_source, assembler, objcopy)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ps_after_end.bin"
            path.write_bytes(scan_code)
            result, = scan_directory_or_files([path])
        self.assertEqual(result["total_instructions"], len(scan_source))
        self.assertEqual(result["opcode_distribution"]["v_mov_b32_e32"], 1)
        self.assertTrue(result["opcodes_recognized"])
        self.assertEqual(result["supported_instructions"], len(scan_source))
        self.assertFalse(result["translator_accepted"])
        self.assertEqual(result["lower_error"], "Instruction follows program end")

    def test_decode_continues_past_s_endpgm(self):
        """Issue #938 step 4: disasm_rdna2 must decode past s_endpgm (LLVM 15 behaviour).

        Verifies that instructions after s_endpgm are decoded, and that the
        after_endpgm field distinguishes pre/post terminator instructions.
        """
        # s_endpgm followed by a valid instruction — must NOT stop at offset 4.
        stream = bytes.fromhex("000081bf" + "0103007e")  # s_endpgm | v_mov_b32_e32 v0, v1
        insns = disasm_rdna2(stream)
        self.assertEqual(len(insns), 2)
        self.assertEqual(insns[0]["opcode"], "s_endpgm")
        self.assertEqual(insns[1]["opcode"], "v_mov_b32_e32")
        self.assertFalse(insns[0]["after_endpgm"])
        self.assertTrue(insns[1]["after_endpgm"])

        # Instruction sequence: v_mov_b32_e32 v0, v1 | s_nop 0 | s_endpgm | v_mov_b32_e32 v2, v3
        stream2 = bytes.fromhex(
            "0103007e" +  # v_mov_b32_e32 v0, v1
            "000080bf" +  # s_nop 0
            "000081bf" +  # s_endpgm
            "0307007e"    # v_mov_b32_e32 v2, v3
        )
        insns2 = disasm_rdna2(stream2)
        self.assertGreaterEqual(len(insns2), 4)
        self.assertEqual(insns2[2]["opcode"], "s_endpgm")
        self.assertEqual(insns2[3]["after_endpgm"], True)

        # A second terminator cannot move the first boundary.
        repeated = disasm_rdna2(stream + bytes.fromhex("000081bf0103007e"))
        self.assertEqual([ins["after_endpgm"] for ins in repeated], [False, True, True, True])
        unterminated = disasm_rdna2(bytes.fromhex("0103007e000080bf"))
        self.assertEqual([ins["after_endpgm"] for ins in unterminated], [False, False])

    def test_second_dword_forms_keep_the_stream_aligned(self):
        # Independent GFX10 encodings from LLVM MC 15.0.7 test/MC/AMDGPU (gfx10_asm_vop1.s,
        # gfx10_asm_vop2.s, gfx10_asm_dpp8.s, gfx10_asm_sop.s). Each is one 8-byte instruction.
        endpgm = bytes.fromhex("000081bf")
        cases = (
            ("f9160a7e01060600", "vop1_sdwa", "v_cvt_f32_f16_sdwa <unsupported SDWA>"),
            ("fa020a7e01e40000", "vop1_dpp16", "v_mov_b32_dpp <unsupported DPP16>"),
            ("e9020a7e0188c6fa", "vop1_dpp8", "v_mov_b32_dpp <unsupported DPP8>"),
            ("ea020a7e0188c6fa", "vop1_dpp8", "v_mov_b32_dpp <unsupported DPP8>"),  # fi:1
            ("f9040a0601060606", "vop2_sdwa", "v_add_f32_sdwa <unsupported SDWA>"),
            ("fa040a1001e40000", "vop2_dpp16", "v_mul_f32_dpp <unsupported DPP16>"),
            ("e9040a060188c6fa", "vop2_dpp8", "v_add_f32_dpp <unsupported DPP8>"),
            ("01070a6e21110000", "v_fmamk_f16", "v_fmamk_f16 v5, v1, 0x1121, v3"),
            ("01050a7021110000", "v_fmaak_f16", "v_fmaak_f16 v5, v1, v2, 0x1121"),
            # Issue #938: LLVM 15 prints SIMM16 0x1234 as hwreg(52, 8, 3).
            ("341280ba41312111", "s_setreg_imm32_b32", "s_setreg_imm32_b32 hwreg(52, 8, 3), 0x11213141"),
        )
        for raw, opcode, assembly in cases:
            with self.subTest(raw=raw):
                insns = disasm_rdna2(bytes.fromhex(raw) + endpgm)
                self.assertEqual([(i["offset"], i["length"], i["opcode"]) for i in insns],
                                 [(0, 8, opcode), (8, 4, "s_endpgm")])
                self.assertEqual(insns[0]["assembly"], assembly)
                truncated = disasm_rdna2(bytes.fromhex(raw)[:4])
                self.assertEqual(len(truncated), 1)
                self.assertEqual(truncated[0]["length"], 4)
                self.assertTrue(truncated[0]["opcode"].endswith("_truncated"), truncated[0]["opcode"])

        # Control: SRC0 codes next to the extension codes name an operand and stay one dword.
        for src0 in (232, 235, 248, 251, 256):
            with self.subTest(src0=src0):
                insns = disasm_rdna2(struct.pack("<I", 0x7e0a0200 | src0) + endpgm)
                self.assertEqual([(i["length"], i["opcode"]) for i in insns], [(4, "v_mov_b32_e32"), (4, "s_endpgm")])

    def test_sopc_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import SOPC_OPS
        for op, (expected_name, _) in SOPC_OPS.items():
            w0 = 0xbf000000 | (op << 16) | (2 << 8) | 0
            raw = struct.pack("<I", w0)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding SOPC op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 4)

        w0_literal = 0xbf000000 | (0x06 << 16) | (255 << 8) | 0
        truncated_raw = struct.pack("<I", w0_literal)
        insns = disasm_rdna2(truncated_raw)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "sopc_truncated")
        self.assertIn("<truncated>", insns[0]["assembly"])

    def test_vopc_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import VOPC_OPS
        for op, (expected_name, _) in VOPC_OPS.items():
            wide = expected_name.endswith(("_f64_e32", "_i64_e32", "_u64_e32"))
            src0 = 0 if wide else 1
            w0 = 0x7c000000 | (op << 17) | (2 << 9) | src0
            insns = disasm_rdna2(struct.pack("<I", w0))
            self.assertEqual(len(insns), 1, f"Failed decoding VOPC op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 4)
            lhs = "s[0:1]" if wide else "s1"
            rhs = "v[2:3]" if wide and "_class_" not in expected_name else "v2"
            dest = "" if expected_name.startswith("v_cmpx_") else "vcc_lo, "
            self.assertEqual(insns[0]["assembly"], f"{expected_name} {dest}{lhs}, {rhs}")

        # Independent GFX10 encodings from LLVM MC's gfx10_asm_vopc/vopcx fixtures.
        for raw, expected in (
            ("6404427c", "v_cmp_lt_f64_e32 vcc_lo, s[100:101], v[2:3]"),
            ("01fd437c", "v_cmp_lt_f64_e32 vcc_lo, v[1:2], v[254:255]"),
        ):
            self.assertEqual(disasm_rdna2(bytes.fromhex(raw))[0]["assembly"], expected)
        sdwa = bytes.fromhex("f904247c01001606")
        insns = disasm_rdna2(sdwa + bytes.fromhex("000081bf"))
        self.assertEqual([i["opcode"] for i in insns], ["v_cmpx_eq_f32_sdwa", "s_endpgm"])
        self.assertEqual(insns[0]["length"], 8)
        self.assertEqual(insns[0]["assembly"], "v_cmpx_eq_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD")
        self.assertEqual(disasm_rdna2(sdwa[:4])[0]["opcode"], "vopc_truncated")

        # Truncated VOPC literal
        w0_lit = 0x7c000000 | (0x01 << 17) | (2 << 9) | 255
        insns = disasm_rdna2(struct.pack("<I", w0_lit))
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "vopc_truncated")
        self.assertIn("<truncated>", insns[0]["assembly"])

        # Literal source in VOPC
        raw_lit = struct.pack("<2I", w0_lit, 0x3f800000)
        insns = disasm_rdna2(raw_lit)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "v_cmp_lt_f32_e32")
        self.assertEqual(insns[0]["length"], 8)
        self.assertEqual(insns[0]["assembly"], "v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v2")

        # Scanner recognizes VOPC and assigns correct category
        self.assertTrue(is_opcode_recognized("v_cmp_lt_f32_e32"))
        self.assertTrue(is_opcode_recognized("v_cmpx_gt_i32_e32"))
        self.assertEqual(classify_isa_category("v_cmp_lt_f32_e32"), "VOPC (Vector Compare)")
        self.assertEqual(classify_isa_category("v_cmpx_gt_i32_e32"), "VOPC (Vector Compare)")

    def test_vopc_sdwa_decoding(self):
        """Issue #938: decode GFX10 VOP_SDWA9Be (VOPC/VOPCX SDWA) instructions in LLVM 15 syntax."""
        # Independent GFX10 wave32 encodings from LLVM 15.0.7 gfx10_asm_vopc_sdwa.s and gfx10_asm_vopcx.s.
        golden = [
            ("f904887d01000606", "v_cmp_gt_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904887d01860606", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904887d01e40606", "v_cmp_gt_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904887d01868606", "v_cmp_gt_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904887d7c868606", "v_cmp_gt_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f9fe897d01860606", "v_cmp_gt_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD"),
            ("f904887d01860006", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD"),
            ("f904887d01860106", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD"),
            ("f904887d01860206", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD"),
            ("f904887d01860306", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD"),
            ("f904887d01860406", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD"),
            ("f904887d01860506", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD"),
            ("f904887d01860600", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0"),
            ("f904887d01860605", "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1"),
            ("f904887d01860e06", "v_cmp_gt_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904887d0186060e", "v_cmp_gt_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD"),
            ("f904887d01000506", "v_cmp_gt_u32_sdwa vcc_lo, v1, v2 src0_sel:WORD_1 src1_sel:DWORD"),
            ("f904027c01861606", "v_cmp_lt_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904027c01862606", "v_cmp_lt_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904027c01863606", "v_cmp_lt_f32_sdwa s6, -|v1|, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904027c01860616", "v_cmp_lt_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904027c01860626", "v_cmp_lt_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD"),
            ("f904927d01000606", "v_cmp_lt_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904927d01861606", "v_cmp_lt_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904107d01861606", "v_cmp_class_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904107d01862606", "v_cmp_class_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904107d0186060e", "v_cmp_class_f32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD"),
            ("f904127d01860e06", "v_cmp_lt_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904547d01000606", "v_cmp_eq_u16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904247c01001606", "v_cmpx_eq_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904247c01000626", "v_cmpx_eq_f32_sdwa v1, |v2| src0_sel:DWORD src1_sel:DWORD"),
            ("f904a87d01008606", "v_cmpx_gt_u32_sdwa s1, v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904a87d01000e06", "v_cmpx_gt_u32_sdwa sext(v1), v2 src0_sel:DWORD src1_sel:DWORD"),
            ("f904b27d01000506", "v_cmpx_lt_f16_sdwa v1, v2 src0_sel:WORD_1 src1_sel:DWORD"),
        ]
        for raw_hex, expected in golden:
            insns = disasm_rdna2(bytes.fromhex(raw_hex))
            self.assertEqual(len(insns), 1, raw_hex)
            self.assertEqual(insns[0]["length"], 8, raw_hex)
            self.assertEqual(insns[0]["opcode"], expected.split()[0], raw_hex)
            self.assertEqual(insns[0]["assembly"], expected, raw_hex)

        # src1_sgpr bit (Inst{63}) decodes src1 as a scalar operand per VOPC_SDWA9e.
        insns = disasm_rdna2(bytes.fromhex("f904887d01860686"))
        self.assertEqual(insns[0]["assembly"], "v_cmp_gt_u32_sdwa s6, v1, s2 src0_sel:DWORD src1_sel:DWORD")

        # Negative and boundary controls: encodings LLVM 15 refuses stay 8-byte vopc_sdwa records,
        # and a 4-byte SDWA prefix returns vopc_truncated.
        refused = (
            "f904427c01000606",  # v_cmp_lt_f64_e32 with src0=249 (64-bit compare)
            "f904887d01000706",  # src0_sel == 7 (invalid)
            "f904887d01000607",  # src1_sel == 7 (invalid)
            "f904027c01860e06",  # float compare with src0_sext set
            "f904027c0186060e",  # float compare with src1_sext set
            "f904887d01861606",  # integer compare with src0_neg set
            "f904887d01860626",  # integer compare with src1_abs set
            "f904107d01860616",  # v_cmp_class_f32_sdwa with src1_neg set
            "f904247c01860606",  # v_cmpx_eq_f32_sdwa with sdst_en set
            "f904887dff868606",  # src0_sgpr=1 with literal tag 255
        )
        for bad_hex in refused:
            bad = disasm_rdna2(bytes.fromhex(bad_hex) + bytes.fromhex("000081bf"))
            self.assertEqual([i["opcode"] for i in bad], ["vopc_sdwa", "s_endpgm"], bad_hex)
            self.assertEqual(bad[0]["length"], 8, bad_hex)
            self.assertIn("<unsupported SDWA>", bad[0]["assembly"], bad_hex)
        trunc = disasm_rdna2(bytes.fromhex("f904887d"))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "vopc_truncated")
        self.assertEqual(trunc[0]["length"], 4)

    # Issue #938: VOPC SDWA neg without abs on an inline constant, printed neg(c) as LLVM 15 does. The encodings
    # are llvm-mc 15.0.7 output (gfx1013, wave32) for the same text; LLVM prints float constants in decimal.
    SDWA_NEG_INLINE_DECODED = (
        ("f904027c81009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(1), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027c80009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(0), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cc0009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(64), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cc1009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(-1), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cd0009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(-16), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cf0009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(0.5), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cf1009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(-0.5), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cf2009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(1.0), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cf8009606", "v_cmp_lt_f32_sdwa vcc_lo, neg(0.15915494), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f902037c01000696", "v_cmp_lt_f32_sdwa vcc_lo, v1, neg(1) src0_sel:DWORD src1_sel:DWORD"),
        ("f9e2037c01000696", "v_cmp_lt_f32_sdwa vcc_lo, v1, neg(-0.5) src0_sel:DWORD src1_sel:DWORD"),
        ("f904927df4869606", "v_cmp_lt_f16_sdwa s6, neg(2.0), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f9ee937d01000696", "v_cmp_lt_f16_sdwa vcc_lo, v1, neg(-4.0) src0_sel:DWORD src1_sel:DWORD"),
        ("f904107dc1869606", "v_cmp_class_f32_sdwa s6, neg(-1), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904247cf0009606", "v_cmpx_eq_f32_sdwa neg(0.5), v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f90e257c01000696", "v_cmpx_eq_f32_sdwa v1, neg(7) src0_sel:DWORD src1_sel:DWORD"),
        # Spellings that already matched LLVM 15 and stay unchanged: abs, abs+neg, the inline constants
        # themselves, and neg on a register or a named special operand.
        ("f904027c8100b606", "v_cmp_lt_f32_sdwa vcc_lo, -|1|, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cc100a606", "v_cmp_lt_f32_sdwa vcc_lo, |-1|, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cf000b606", "v_cmp_lt_f32_sdwa vcc_lo, -|0.5|, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cc1008606", "v_cmp_lt_f32_sdwa vcc_lo, -1, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027cf1008606", "v_cmp_lt_f32_sdwa vcc_lo, -0.5, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027c01009606", "v_cmp_lt_f32_sdwa vcc_lo, -s1, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027ceb009606", "v_cmp_lt_f32_sdwa vcc_lo, -src_shared_base, v2 src0_sel:DWORD src1_sel:DWORD"),
    )
    # neg on an integer operand stays refused: an integer compare's src0 and a class compare's src1, each with
    # neg on the inline constant 1 (the accepted forms without neg are f904887d81868606 and f902117d01860686).
    SDWA_NEG_INLINE_REFUSED = ("f904887d81869606", "f902117d01860696")

    def test_vopc_sdwa_neg_on_inline_constant(self):
        """Issue #938: neg without abs on an inline constant is neg(c), not -c, which names another constant."""
        mov = bytes.fromhex("0403067e")  # v_mov_b32_e32 v3, v4
        for raw_hex, expected in self.SDWA_NEG_INLINE_DECODED:
            insns = disasm_rdna2(bytes.fromhex(raw_hex) + mov)
            self.assertEqual(
                [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in insns],
                [(0, 8, expected.split()[0], expected), (8, 4, "v_mov_b32_e32", "v_mov_b32_e32 v3, v4")],
                raw_hex,
            )
        for raw_hex in self.SDWA_NEG_INLINE_REFUSED:
            insns = disasm_rdna2(bytes.fromhex(raw_hex) + mov)
            self.assertEqual(
                [(i["offset"], i["length"], i["opcode"]) for i in insns],
                [(0, 8, "vopc_sdwa"), (8, 4, "v_mov_b32_e32")],
                raw_hex,
            )
            self.assertIn("<unsupported SDWA>", insns[0]["assembly"], raw_hex)
            self.assertEqual(insns[1]["assembly"], "v_mov_b32_e32 v3, v4", raw_hex)

        # neg(1) (0x80000001) and the inline constant -1 (0xffffffff) are different operands with different text.
        neg_one = disasm_rdna2(bytes.fromhex("f924027c81009606"))[0]["assembly"]
        minus_one = disasm_rdna2(bytes.fromhex("f924027cc1008606"))[0]["assembly"]
        self.assertEqual(neg_one, "v_cmp_lt_f32_sdwa vcc_lo, neg(1), v18 src0_sel:DWORD src1_sel:DWORD")
        self.assertEqual(minus_one, "v_cmp_lt_f32_sdwa vcc_lo, -1, v18 src0_sel:DWORD src1_sel:DWORD")

        # Ordinary -1 remains valid. neg(1) must never lower as -1, and spelling a decimal
        # modifier explicitly must not change its established binary32 comparison.
        from pixel_aot_check import with_constants
        from scan_unsupported_shaders import is_instruction_supported

        def lowered(compare):
            program = with_constants(["v_sub_f32_e64 v18, s45, 0", "v_sub_f32_e64 v19, s47, 0", compare,
                                      "v_cndmask_b32_e32 v12, 0, v19, vcc_lo"])
            return lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64)

        self.assertIn("asfloat(4294967295u) < alu20", lowered(minus_one))
        with self.assertRaisesRegex(ValueError, r"Unsupported arithmetic operand neg\("):
            lowered(neg_one)
        self.assertFalse(is_instruction_supported(neg_one))
        for raw_hex, decimal in (("f924027cf0009606", "0.5"), ("f924027cf1009606", "-0.5")):
            compare = disasm_rdna2(bytes.fromhex(raw_hex))[0]["assembly"]
            with self.subTest(decimal=decimal):
                self.assertIn(f"neg({decimal})", compare)
                self.assertEqual(lowered(compare), lowered(compare.replace(f"neg({decimal})", f"-{decimal}")))
                self.assertTrue(is_instruction_supported(compare))

    def test_vopc_sdwa_neg_on_inline_constant_matches_llvm(self):
        """Issue #938: the neg(c) spellings and refusals equal LLVM 15's."""
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        mov = bytes.fromhex("0403067e")
        for raw_hex, expected in self.SDWA_NEG_INLINE_DECODED:
            with self.subTest(llvm_decoded=raw_hex):
                self.assertEqual(llvm_disassemble(bytes.fromhex(raw_hex) + mov, resolved_lib, offsets=True),
                                 [(0, expected), (8, "v_mov_b32_e32 v3, v4")])
        for raw_hex in self.SDWA_NEG_INLINE_REFUSED:
            with self.subTest(llvm_refused=raw_hex):
                with self.assertRaisesRegex(ValueError, "at byte 0$"):
                    llvm_disassemble(bytes.fromhex(raw_hex) + mov, resolved_lib)

    def test_vopc_sdwa_llvm_round_trip(self):
        """Issue #938: VOPC/VOPCX SDWA forms round-trip byte-for-byte through LLVM 15."""
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])
        if not assembler or not objcopy:
            self.fail(f"VOPC SDWA round trip requires llvm-mc and llvm-objcopy (assembler={assembler}, objcopy={objcopy})")

        source = [
            "v_cmp_gt_u32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s100, v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, s1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, m0, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v255 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_0 src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_1 src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_2 src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:BYTE_3 src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:WORD_0 src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:WORD_1 src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:BYTE_0",
            "v_cmp_gt_u32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:WORD_1",
            "v_cmp_gt_u32_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_gt_u32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_lt_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_lt_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_lt_f32_sdwa s6, -|v1|, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_lt_f32_sdwa s6, v1, -v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_lt_f32_sdwa s6, v1, |v2| src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_lt_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_lt_f16_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_class_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_class_f32_sdwa s6, |v1|, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_class_f32_sdwa s6, v1, sext(v2) src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_lt_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmp_eq_u16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmpx_eq_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmpx_eq_f32_sdwa v1, |v2| src0_sel:DWORD src1_sel:DWORD",
            "v_cmpx_gt_u32_sdwa s1, v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmpx_gt_u32_sdwa sext(v1), v2 src0_sel:DWORD src1_sel:DWORD",
            "v_cmpx_lt_f16_sdwa v1, v2 src0_sel:WORD_1 src1_sel:DWORD",
            "s_endpgm",
        ]
        emitted = assemble_with_llvm_mc(source, assembler, objcopy)
        llvm_lines = llvm_disassemble(emitted, resolved_lib)
        decoded = [rec["assembly"] for rec in disasm_rdna2(emitted)]
        self.assertEqual(llvm_lines, source)
        self.assertEqual(decoded, llvm_lines)
        self.assertEqual(assemble_with_llvm_mc(decoded, assembler, objcopy), emitted)
        # All 33 SDWA lines are 8 bytes each and must produce 33 distinct encodings.
        self.assertEqual(len({emitted[i:i + 8] for i in range(0, 33 * 8, 8)}), 33)
        # Verify LLVM 15 itself refuses every reserved/wrong-type SDWA bit pattern we reject.
        # (src0_sel/src1_sel == 7 is intentionally omitted here: LLVM 15's printSDWASel hits llvm_unreachable.)
        for bad_hex in (
            "f904427c01000606",
            "f904027c01860e06",
            "f904027c0186060e",
            "f904887d01861606",
            "f904887d01860626",
            "f904107d01860616",
            "f904247c01860606",
            "f904887dff868606",
        ):
            with self.subTest(llvm_refused=bad_hex):
                with self.assertRaises(ValueError):
                    llvm_disassemble(bytes.fromhex(bad_hex), resolved_lib)

    # Issue #938: VOPC SDWA scalar source codes 209-234 lie outside the supported LLVM 15 decoder subset.
    # Accepted boundary records (codes 208 and 235), assembled by llvm-mc 15.0.7 for gfx1013 wave32.
    SDWA_SCALAR_SUBSET_ACCEPTED = (
        ("f904027cd0008606", "v_cmp_lt_f32_sdwa vcc_lo, -16, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f904027ceb008606", "v_cmp_lt_f32_sdwa vcc_lo, src_shared_base, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f9a0037c01000686", "v_cmp_lt_f32_sdwa vcc_lo, v1, -16 src0_sel:DWORD src1_sel:DWORD"),
        ("f9d6037c01000686", "v_cmp_lt_f32_sdwa vcc_lo, v1, src_shared_base src0_sel:DWORD src1_sel:DWORD"),
        ("f904887dd0868606", "v_cmp_gt_u32_sdwa s6, -16, v2 src0_sel:DWORD src1_sel:DWORD"),
        ("f9d6a97d01000686", "v_cmpx_gt_u32_sdwa v1, src_shared_base src0_sel:DWORD src1_sel:DWORD"),
    )
    # The same records with the scalar code moved to 209 or 234 (no assembler spelling exists for these codes).
    SDWA_SCALAR_SUBSET_REFUSED = (
        "f904027cd1008606",  # v_cmp_lt_f32 SDWA, src0 = scalar code 209
        "f904027cea008606",  # v_cmp_lt_f32 SDWA, src0 = scalar code 234
        "f9a2037c01000686",  # v_cmp_lt_f32 SDWA, src1 = scalar code 209
        "f9d4037c01000686",  # v_cmp_lt_f32 SDWA, src1 = scalar code 234
        "f904887dd1868606",  # v_cmp_gt_u32 SDWA (sdst s6), src0 = scalar code 209
        "f9d4a97d01000686",  # v_cmpx_gt_u32 SDWA, src1 = scalar code 234
    )

    def test_vopc_sdwa_scalar_codes_outside_llvm15_subset(self):
        """Issue #938: refuse VOPC SDWA scalar codes 209-234 as 8-byte records and resync on the next dword."""
        mov = bytes.fromhex("0403067e")  # v_mov_b32_e32 v3, v4
        for raw_hex, expected in self.SDWA_SCALAR_SUBSET_ACCEPTED:
            insns = disasm_rdna2(bytes.fromhex(raw_hex) + mov)
            self.assertEqual(
                [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in insns],
                [(0, 8, expected.split()[0], expected), (8, 4, "v_mov_b32_e32", "v_mov_b32_e32 v3, v4")],
                raw_hex,
            )
        subset_comment = "VOPC SDWA scalar operand outside the supported LLVM 15 decoder subset"
        for raw_hex in self.SDWA_SCALAR_SUBSET_REFUSED:
            insns = disasm_rdna2(bytes.fromhex(raw_hex) + mov)
            self.assertEqual(
                [(i["offset"], i["length"], i["opcode"]) for i in insns],
                [(0, 8, "vopc_sdwa"), (8, 4, "v_mov_b32_e32")],
                raw_hex,
            )
            self.assertIn("<unsupported SDWA>", insns[0]["assembly"], raw_hex)
            self.assertEqual(insns[0]["comment"], subset_comment, raw_hex)
            self.assertEqual(insns[1]["assembly"], "v_mov_b32_e32 v3, v4", raw_hex)

        # Every code in 209-234 is refused as src0 and as src1, with or without input modifiers.
        for code in range(209, 235):
            for neg in (0, 1):
                w1_src0 = code | (1 << 23) | (6 << 16) | (6 << 24) | (neg << 20)
                w1_src1 = 1 | (1 << 31) | (6 << 16) | (6 << 24) | (neg << 28)
                src0_record = struct.pack("<II", 0x7C0204F9, w1_src0)  # v_cmp_lt_f32 SDWA, vsrc1 = v2
                src1_record = struct.pack("<II", 0x7C0200F9 | (code << 9), w1_src1)  # src0 = v1
                for record in (src0_record, src1_record):
                    insns = disasm_rdna2(record + mov)
                    self.assertEqual(
                        [(i["offset"], i["opcode"], i["comment"]) for i in insns],
                        [(0, "vopc_sdwa", subset_comment), (8, "v_mov_b32_e32", insns[1]["comment"])],
                        record.hex(),
                    )
                    self.assertEqual(insns[1]["assembly"], "v_mov_b32_e32 v3, v4", record.hex())

    def test_vopc_sdwa_scalar_subset_matches_llvm(self):
        """Issue #938: LLVM 15 decodes the boundary records and refuses scalar codes 209-234."""
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        mov = bytes.fromhex("0403067e")
        for raw_hex, expected in self.SDWA_SCALAR_SUBSET_ACCEPTED:
            with self.subTest(llvm_decoded=raw_hex):
                self.assertEqual(llvm_disassemble(bytes.fromhex(raw_hex) + mov, resolved_lib, offsets=True),
                                 [(0, expected), (8, "v_mov_b32_e32 v3, v4")])
        for raw_hex in self.SDWA_SCALAR_SUBSET_REFUSED:
            with self.subTest(llvm_refused=raw_hex):
                with self.assertRaisesRegex(ValueError, "at byte 0$"):
                    llvm_disassemble(bytes.fromhex(raw_hex) + mov, resolved_lib)

    def test_sop2_absdiff_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import SOP2_OPS
        expected_name, _ = SOP2_OPS[0x2c]
        self.assertEqual(expected_name, "s_absdiff_i32")
        w0 = 0x80000000 | (0x2c << 23) | (2 << 16) | (1 << 8) | 0
        raw = struct.pack("<I", w0)
        insns = disasm_rdna2(raw)
        self.assertEqual(len(insns), 1, "Failed decoding SOP2 op 0x2c")
        self.assertEqual(insns[0]["opcode"], "s_absdiff_i32")
        self.assertEqual(insns[0]["length"], 4)
        self.assertEqual(insns[0]["assembly"], "s_absdiff_i32 s2, s0, s1")

        self.assertTrue(is_opcode_recognized("s_absdiff_i32"))
        self.assertEqual(classify_isa_category("s_absdiff_i32"), "SALU (Scalar ALU)")


    def test_vop2_carry_and_mimg_atomic_encodings(self):
        """Issue #938 step 3: VOP2 carry ops with implicit vcc_lo and GFX10 MIMG image_atomic_* match LLVM 15."""
        endpgm = bytes.fromhex("000081bf")
        # Golden encodings from LLVM 15.0.7 test/MC/AMDGPU/gfx10_asm_vop2.s and gfx10_asm_mimg.s.
        cases = (
            ("01050a50", "v_add_co_ci_u32_e32", "v_add_co_ci_u32_e32 v5, vcc_lo, v1, v2, vcc_lo"),
            ("80040a50", "v_add_co_ci_u32_e32", "v_add_co_ci_u32_e32 v5, vcc_lo, 0, v2, vcc_lo"),
            ("c1040a50", "v_add_co_ci_u32_e32", "v_add_co_ci_u32_e32 v5, vcc_lo, -1, v2, vcc_lo"),
            ("ff040a5078563412", "v_add_co_ci_u32_e32", "v_add_co_ci_u32_e32 v5, vcc_lo, 0x12345678, v2, vcc_lo"),
            ("01050a52", "v_sub_co_ci_u32_e32", "v_sub_co_ci_u32_e32 v5, vcc_lo, v1, v2, vcc_lo"),
            ("01050a54", "v_subrev_co_ci_u32_e32", "v_subrev_co_ci_u32_e32 v5, vcc_lo, v1, v2, vcc_lo"),
            ("10213cf020041800", "image_atomic_swap", "image_atomic_swap v4, v[32:34], s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_3D glc"),
            ("122340f02004180001020000", "image_atomic_cmpswap", "image_atomic_cmpswap v[4:5], [v32, v1, v2], s[96:103] dmask:0x3 dim:SQ_RSRC_IMG_3D glc"),
            ("1a2344f02004180001020000", "image_atomic_add", "image_atomic_add v[4:5], [v32, v1, v2], s[96:103] dmask:0x3 dim:SQ_RSRC_IMG_CUBE glc"),
            ("222148f02004180001000000", "image_atomic_sub", "image_atomic_sub v4, [v32, v1], s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY glc"),
            ("2a2150f02004180001020000", "image_atomic_smin", "image_atomic_smin v4, [v32, v1, v2], s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY glc"),
            ("322154f02004180001020000", "image_atomic_umin", "image_atomic_umin v4, [v32, v1, v2], s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA glc"),
            ("3a2158f02004180001020300", "image_atomic_smax", "image_atomic_smax v4, [v32, v1, v2, v3], s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY glc"),
            ("0a215cf02004180001000000", "image_atomic_umax", "image_atomic_umax v4, [v32, v1], s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_2D glc"),
            ("002160f020041800", "image_atomic_and", "image_atomic_and v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc"),
            ("002164f020041800", "image_atomic_or", "image_atomic_or v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc"),
            ("002168f020041800", "image_atomic_xor", "image_atomic_xor v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc"),
            ("00216cf020041800", "image_atomic_inc", "image_atomic_inc v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc"),
            ("002170f020041800", "image_atomic_dec", "image_atomic_dec v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc"),
            ("002374f020041800", "image_atomic_fcmpswap", "image_atomic_fcmpswap v[4:5], v32, s[96:103] dmask:0x3 dim:SQ_RSRC_IMG_1D glc"),
            ("002178f020041800", "image_atomic_fmin", "image_atomic_fmin v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc"),
            ("00217cf020041800", "image_atomic_fmax", "image_atomic_fmax v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc"),
        )
        for raw, opcode, assembly in cases:
            with self.subTest(raw=raw):
                raw_bytes = bytes.fromhex(raw)
                insns = disasm_rdna2(raw_bytes + endpgm)
                self.assertEqual(
                    [(i["offset"], i["length"], i["opcode"]) for i in insns],
                    [(0, len(raw_bytes), opcode), (len(raw_bytes), 4, "s_endpgm")],
                )
                self.assertEqual(insns[0]["assembly"], assembly)

        # Truncation controls: 4-byte truncated VOP2 carry literal and truncated MIMG atomic.
        trunc_vop2 = disasm_rdna2(bytes.fromhex("ff040a50"))
        self.assertEqual(
            [(i["opcode"], i["assembly"]) for i in trunc_vop2],
            [("vop2_truncated", "v_add_co_ci_u32_e32 <truncated>")],
        )
        trunc_mimg = disasm_rdna2(bytes.fromhex("0a215cf020041800"))  # nsize=1 expects 12 bytes
        self.assertEqual(trunc_mimg[0]["opcode"], "mimg_truncated")

    def test_vop2_carry_and_mimg_atomic_llvm_round_trip(self):
        """Issue #938 step 3: VOP2 carry ops and GFX10 image_atomic_* round-trip through LLVM 15 on gfx1013."""
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])
        if not assembler or not objcopy:
            self.fail(
                f"carry/atomic round trip requires llvm-mc and llvm-objcopy "
                f"(assembler={assembler}, objcopy={objcopy})"
            )

        source = [
            "v_add_co_ci_u32_e32 v1, vcc_lo, v2, v3, vcc_lo",
            "v_add_co_ci_u32_e32 v5, vcc_lo, 0x12345678, v2, vcc_lo",
            "v_sub_co_ci_u32_e32 v1, vcc_lo, v2, v3, vcc_lo",
            "v_subrev_co_ci_u32_e32 v1, vcc_lo, v2, v3, vcc_lo",
            "image_atomic_swap v4, v[32:34], s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_3D glc",
            "image_atomic_cmpswap v[4:5], [v32, v1, v2], s[96:103] dmask:0x3 dim:SQ_RSRC_IMG_3D glc",
            "image_atomic_add v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_sub v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_smin v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_umin v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_smax v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_umax v1, v2, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_and v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_or v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_xor v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_inc v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_dec v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_fcmpswap v[4:5], v32, s[96:103] dmask:0x3 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_fmin v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "image_atomic_fmax v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D glc",
            "s_endpgm",
        ]
        emitted = assemble_with_llvm_mc(source, assembler, objcopy)
        llvm_lines = llvm_disassemble(emitted, resolved_lib)
        decoded = [rec["assembly"] for rec in disasm_rdna2(emitted)]
        self.assertEqual(llvm_lines, source)
        self.assertEqual(decoded, llvm_lines)
        self.assertEqual(assemble_with_llvm_mc(decoded, assembler, objcopy), emitted)

    def test_mimg_missing_gfx1013_ops_golden_encodings(self):
        """Issue #938: sampling, gather, packed-load, MSAA-load and BVH ops decode to the LLVM spelling."""
        vectors = [
            ("0807a0f004102303", "image_sample_c",
             "image_sample_c v[16:18], v[4:6], s[12:19], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_2D"),
            ("080f20f104102303", "image_gather4_c",
             "image_gather4_c v[16:19], v[4:6], s[12:19], s[100:103] dmask:0xf dim:SQ_RSRC_IMG_2D"),
            ("100fa8f004102303", "image_sample_c_d",
             "image_sample_c_d v[16:19], v[4:19], s[12:19], s[100:103] dmask:0xf dim:SQ_RSRC_IMG_3D"),
            # One-hot gather: four texels of the selected component, two registers with d16.
            ("080120f104102303", "image_gather4_c",
             "image_gather4_c v[16:19], v[4:6], s[12:19], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_2D"),
            ("080120f104102383", "image_gather4_c",
             "image_gather4_c v[16:17], v[4:6], s[12:19], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_2D d16"),
            ("1103a8f004102303", "image_sample_c_d_g16",
             "image_sample_c_d_g16 v[16:17], v[4:11], s[12:19], s[100:103] dmask:0x3 dim:SQ_RSRC_IMG_3D"),
            ("0101b8f104102303", "image_sample_c_cd_o_g16",
             "image_sample_c_cd_o_g16 v16, v[4:8], s[12:19], s[100:103] dmask:0x1 dim:SQ_RSRC_IMG_1D"),
            ("080f08f004100300", "image_load_pck",
             "image_load_pck v[16:19], v[4:5], s[12:19] dmask:0xf dim:SQ_RSRC_IMG_2D"),
            ("310f00f004100300", "image_msaa_load",
             "image_msaa_load v[16:19], v[4:6], s[12:19] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA"),
            # The printed tuple is LLVM 15's (9 to 15 address dwords print as 16 registers), not the dword count.
            ("019f98f100040100", "image_bvh_intersect_ray", "image_bvh_intersect_ray v[4:7], v[0:15], s[4:7]"),
            ("019f98f100040140", "image_bvh_intersect_ray", "image_bvh_intersect_ray v[4:7], v[0:7], s[4:7] a16"),
            ("019f9cf100040100", "image_bvh64_intersect_ray", "image_bvh64_intersect_ray v[4:7], v[0:15], s[4:7]"),
            ("019f9cf100040140", "image_bvh64_intersect_ray",
             "image_bvh64_intersect_ray v[4:7], v[0:15], s[4:7] a16"),
            # Non-sequential addressing (w0[2:1] = 1): one VGPR per address dword.
            ("079f98f100040100020406080a0c0e1012140000", "image_bvh_intersect_ray",
             "image_bvh_intersect_ray v[4:7], [v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20], s[4:7]"),
        ]
        for hex_bytes, opcode, assembly in vectors:
            with self.subTest(assembly=assembly):
                records = disasm_rdna2(bytes.fromhex(hex_bytes))
                self.assertEqual([(r["opcode"], r["assembly"]) for r in records], [(opcode, assembly)])

    def test_mimg_bvh_refuses_modifier_bits_and_keeps_stream_alignment(self):
        """Issue #938: a ray-intersection word with modifiers the encoding fixes is refused, not mis-decoded."""
        # dmask 0xe instead of the fixed 0xf, then s_endpgm.
        records = disasm_rdna2(bytes.fromhex("019e98f100040100" "000081bf"))
        self.assertEqual(
            [(r["offset"], r["opcode"]) for r in records],
            [(0, "image_bvh_intersect_ray_unsupported"), (8, "s_endpgm")],
        )
        # d16 set in the second word.
        records = disasm_rdna2(bytes.fromhex("019f98f1000401" "80" "000081bf"))
        self.assertEqual(records[0]["opcode"], "image_bvh_intersect_ray_unsupported")
        self.assertEqual(records[1]["opcode"], "s_endpgm")

    def test_mimg_missing_gfx1013_ops_llvm_round_trip(self):
        """Issue #938: every listed MIMG op decodes exactly as LLVM prints it, on every dimension LLVM models."""
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        ops = {
            0x02: "image_load_pck", 0x03: "image_load_pck_sgn", 0x04: "image_load_mip_pck",
            0x05: "image_load_mip_pck_sgn", 0x0a: "image_store_pck", 0x0b: "image_store_mip_pck",
            0x23: "image_sample_d_cl", 0x29: "image_sample_c_cl", 0x2a: "image_sample_c_d",
            0x2b: "image_sample_c_d_cl", 0x2e: "image_sample_c_b_cl", 0x31: "image_sample_cl_o",
            0x32: "image_sample_d_o", 0x33: "image_sample_d_cl_o", 0x36: "image_sample_b_cl_o",
            0x38: "image_sample_c_o", 0x39: "image_sample_c_cl_o", 0x3a: "image_sample_c_d_o",
            0x3b: "image_sample_c_d_cl_o", 0x3c: "image_sample_c_l_o", 0x3d: "image_sample_c_b_o",
            0x3e: "image_sample_c_b_cl_o", 0x3f: "image_sample_c_lz_o", 0x41: "image_gather4_cl",
            0x45: "image_gather4_b", 0x46: "image_gather4_b_cl", 0x48: "image_gather4_c",
            0x49: "image_gather4_c_cl", 0x4c: "image_gather4_c_l", 0x4d: "image_gather4_c_b",
            0x4e: "image_gather4_c_b_cl", 0x4f: "image_gather4_c_lz", 0x50: "image_gather4_o",
            0x51: "image_gather4_cl_o", 0x54: "image_gather4_l_o", 0x55: "image_gather4_b_o",
            0x56: "image_gather4_b_cl_o", 0x57: "image_gather4_lz_o", 0x58: "image_gather4_c_o",
            0x59: "image_gather4_c_cl_o", 0x5c: "image_gather4_c_l_o", 0x5d: "image_gather4_c_b_o",
            0x5e: "image_gather4_c_b_cl_o", 0x5f: "image_gather4_c_lz_o", 0x68: "image_sample_cd",
            0x69: "image_sample_cd_cl", 0x6a: "image_sample_c_cd", 0x6b: "image_sample_c_cd_cl",
            0x6c: "image_sample_cd_o", 0x6d: "image_sample_cd_cl_o", 0x6e: "image_sample_c_cd_o",
            0x6f: "image_sample_c_cd_cl_o", 0x80: "image_msaa_load", 0xa2: "image_sample_d_g16",
            0xa3: "image_sample_d_cl_g16", 0xaa: "image_sample_c_d_g16", 0xab: "image_sample_c_d_cl_g16",
            0xb2: "image_sample_d_o_g16", 0xb3: "image_sample_d_cl_o_g16", 0xba: "image_sample_c_d_o_g16",
            0xbb: "image_sample_c_d_cl_o_g16", 0xe8: "image_sample_cd_g16", 0xe9: "image_sample_cd_cl_g16",
            0xea: "image_sample_c_cd_g16", 0xeb: "image_sample_c_cd_cl_g16", 0xec: "image_sample_cd_o_g16",
            0xed: "image_sample_cd_cl_o_g16", 0xee: "image_sample_c_cd_o_g16",
            0xef: "image_sample_c_cd_cl_o_g16",
        }
        has_sampler = {0x23, 0x29, 0x2a, 0x2b, 0x2e, 0x31, 0x32, 0x33, 0x36, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x41, 0x45, 0x46, 0x48, 0x49, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5c, 0x5d, 0x5e, 0x5f, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0xa2, 0xa3, 0xaa, 0xab, 0xb2, 0xb3, 0xba, 0xbb, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef}
        code = bytearray()
        names = []
        for op, name in ops.items():
            # Dimensions 0..6: LLVM decodes a five-dword address on the MSAA array dimension as a
            # different operand shape, so that dimension is left out of the comparison.
            # A gather also gets a one-hot mask, with and without d16: it returns four texels (two with d16).
            masks = [(0xf, 0)] + ([(0x1, 0), (0x1, 1)] if "gather4" in name else [])
            for dim in range(7):
                for dmask, d16 in masks:
                    word0 = (0xf0000000 | ((op & 0x7f) << 18) | ((op >> 7) & 1) | (dmask << 8) | (dim << 3))
                    word1 = ((5 << 21) if op in has_sampler else 0) | (2 << 16) | 4 | (d16 << 31)
                    code += struct.pack("<II", word0, word1)
                    names.append(name)
        # LLVM 15 (the CI pin) prints a 9..15-register packed address as a 16-register tuple and the decoder follows
        # it; newer LLVM prints the exact width, so widen its spelling to compare on either LLVM version.
        def widen(line):
            def fix(m):
                lo, hi = int(m.group(1)), int(m.group(2))
                return f"v[{lo}:{lo + 15}]" if 9 <= hi - lo + 1 <= 15 else m.group(0)
            return re.sub(r"v\[(\d+):(\d+)\]", fix, line)
        llvm_lines = [widen(line) for line in llvm_disassemble(bytes(code), resolved_lib)]
        records = disasm_rdna2(bytes(code))
        self.assertEqual(len(llvm_lines), len(names))
        self.assertEqual([r["assembly"] for r in records], llvm_lines)
        self.assertEqual([r["opcode"] for r in records], names)

        bvh = bytes.fromhex("019f98f100040100" "019f98f100040140" "019f9cf100040100" "019f9cf100040140")
        self.assertEqual([r["assembly"] for r in disasm_rdna2(bvh)],
                         [widen(line) for line in llvm_disassemble(bvh, resolved_lib)])
        nsa = bytes.fromhex("079f98f100040100020406080a0c0e1012140000")
        self.assertEqual([r["assembly"] for r in disasm_rdna2(nsa)], llvm_disassemble(nsa, resolved_lib))

    def test_binary_scan_names_newly_decoded_mimg_ops(self):
        """Issue #938: the scan report names gather, compare-sample and ray-intersection ops as LLVM does."""
        shader = bytes.fromhex(
            "080f20f104102303"  # image_gather4_c
            "0807a0f004102303"  # image_sample_c
            "019f98f100040100"  # image_bvh_intersect_ray
            "000081bf"          # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "ps_mimg.bin").write_bytes(shader)
            result = {r["shader_name"]: r for r in scan_directory_or_files([tmp_path])}["ps_mimg.bin"]
        self.assertTrue(result["opcodes_recognized"])
        self.assertEqual(result["unsupported_opcodes"], {})
        self.assertEqual(
            [d["raw"].split()[0] for d in result["unsupported_details"]],
            ["image_gather4_c", "image_sample_c", "image_bvh_intersect_ray"],
        )

    def test_sop2_lshl_add_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import SOP2_OPS
        target_ops = {0x2e, 0x2f, 0x30, 0x31}
        for op in target_ops:
            expected_name, _ = SOP2_OPS[op]
            w0 = 0x80000000 | (op << 23) | (2 << 16) | (1 << 8) | 0
            raw = struct.pack("<I", w0)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding SOP2 op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 4)

    # Encodings and spellings from llvm-mc gfx1013 (identical text on LLVM 15 and 19).
    SOP2_CARRY_LOGIC_PACK_FORMS = (
            ("01020082", "s_addc_u32 s0, s1, s2"),
            ("01028082", "s_subb_u32 s0, s1, s2"),
            ("0102008b", "s_orn2_b32 s0, s1, s2"),
            ("0204808b", "s_orn2_b64 s[0:1], s[2:3], s[4:5]"),
            ("0102008c", "s_nand_b32 s0, s1, s2"),
            ("0204808c", "s_nand_b64 s[0:1], s[2:3], s[4:5]"),
            ("0102008d", "s_nor_b32 s0, s1, s2"),
            ("0204808d", "s_nor_b64 s[0:1], s[2:3], s[4:5]"),
            ("0102008e", "s_xnor_b32 s0, s1, s2"),
            ("0204808e", "s_xnor_b64 s[0:1], s[2:3], s[4:5]"),
            ("01020099", "s_pack_ll_b32_b16 s0, s1, s2"),
            ("01028099", "s_pack_lh_b32_b16 s0, s1, s2"),
            ("0102009a", "s_pack_hh_b32_b16 s0, s1, s2"),
    )

    def test_sop2_carry_logic_pack_forms(self):
        """Issue #938: SOP2 carry, inverted-logic and 16-bit pack ops decode to LLVM's text and re-assemble to the same bytes."""
        raw = b"".join(bytes.fromhex(h) for h, _ in self.SOP2_CARRY_LOGIC_PACK_FORMS)
        expected = [text for _, text in self.SOP2_CARRY_LOGIC_PACK_FORMS]
        self.assertEqual([rec["assembly"] for rec in disasm_rdna2(raw)], expected)
        if not os.environ.get("LLVM_LIB"):
            return
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])
        if not assembler or not objcopy:
            self.fail(f"SOP2 round trip requires llvm-mc and llvm-objcopy (assembler={assembler}, objcopy={objcopy})")
        self.assertEqual(assemble_with_llvm_mc(expected, assembler, objcopy), raw)
        self.assertEqual(llvm_disassemble(raw, resolved_lib), expected)

    def test_vop2_integer_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import VOP2_OPS
        target_ops = {0x16, 0x18, 0x1a, 0x1b, 0x1c, 0x1d, 0x25, 0x26, 0x27}
        for op in target_ops:
            expected_name, _ = VOP2_OPS[op]
            w0 = (op << 25) | (2 << 17) | (1 << 9) | 0
            raw = struct.pack("<I", w0)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding VOP2 op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 4)

        truncated_raw = b"\x16\x00"
        insns = disasm_rdna2(truncated_raw)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "data")
        self.assertEqual(insns[0]["length"], 2)

    def test_vop3_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import VOP3_NATIVE_OPS
        for op, (expected_name, _, arity, is_vop3b, dst_width, src_widths) in VOP3_NATIVE_OPS.items():
            if is_vop3b:
                w0 = 0xd4000000 | (op << 16) | (0 << 8) | 1  # sdst=s0, vdst=v1
            else:
                w0 = 0xd4000000 | (op << 16) | 1  # vdst=v1
            w1 = 258 | (259 << 9)  # src0=v2, src1=v3
            if arity >= 3:
                w1 |= (260 << 18)  # src2=v4
            raw = struct.pack("<2I", w0, w1)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding VOP3 op 0x{op:03x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 8)
            if expected_name == "v_readlane_b32":
                expected_dst = "s1"
            else:
                expected_dst = "v[1:2]" if dst_width == 2 else "v1"
            expected_srcs = []
            raw_regs = [2, 3, 4]
            for i in range(arity):
                w = src_widths[i]
                r = raw_regs[i]
                expected_srcs.append(f"v[{r}:{r+1}]" if w == 2 else f"v{r}")
            srcs_str = ", ".join(expected_srcs)
            if is_vop3b:
                self.assertEqual(insns[0]["assembly"], f"{expected_name} {expected_dst}, s0, {srcs_str}")
            else:
                self.assertEqual(insns[0]["assembly"], f"{expected_name} {expected_dst}, {srcs_str}")

        # Scanner recognizes all native VOP3 opcodes
        self.assertTrue(is_opcode_recognized("v_fma_f32"))
        self.assertTrue(is_opcode_recognized("v_add_co_u32"))
        self.assertTrue(is_opcode_recognized("v_cndmask_b32_e64"))
        self.assertEqual(classify_isa_category("v_fma_f32"), "VOP / VALU (Vector ALU - VOP1/VOP2/VOP3)")

    def test_vop3_distinguishes_non_default_modifiers(self):
        # Base instruction: v_add_f32_e64 v3, v1, v2
        w0_base = 0xd4000000 | (0x103 << 16) | 3
        w1_base = 257 | (258 << 9)

        base_asm = disasm_rdna2(struct.pack("<2I", w0_base, w1_base))[0]["assembly"]
        self.assertEqual(base_asm, "v_add_f32_e64 v3, v1, v2")

        # abs on src0 (w0[8] = 1)
        abs_asm = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 8), w1_base))[0]["assembly"]
        self.assertEqual(abs_asm, "v_add_f32_e64 v3, |v1|, v2")

        # neg on src0 (w1[29] = 1)
        neg_asm = disasm_rdna2(struct.pack("<2I", w0_base, w1_base | (1 << 29)))[0]["assembly"]
        self.assertEqual(neg_asm, "v_add_f32_e64 v3, -v1, v2")

        # abs and neg on src0
        abs_neg_asm = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 8), w1_base | (1 << 29)))[0]["assembly"]
        self.assertEqual(abs_neg_asm, "v_add_f32_e64 v3, -|v1|, v2")

        # clamp (w0[15] = 1)
        clamp_asm = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 15), w1_base))[0]["assembly"]
        self.assertEqual(clamp_asm, "v_add_f32_e64 v3, v1, v2 clamp")

        # omod mul:2 (w1[27:28] = 1)
        omod_asm = disasm_rdna2(struct.pack("<2I", w0_base, w1_base | (1 << 27)))[0]["assembly"]
        self.assertEqual(omod_asm, "v_add_f32_e64 v3, v1, v2 mul:2")

        # op_sel (w0[11] = 1)
        opsel_asm = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 11), w1_base))[0]["assembly"]
        self.assertEqual(opsel_asm, "v_add_f32_e64 v3, v1, v2 op_sel:[1,0,0]")

        variants = [base_asm, abs_asm, neg_asm, abs_neg_asm, clamp_asm, omod_asm, opsel_asm]
        self.assertEqual(len(set(variants)), len(variants), "VOP3 modifiers produced colliding assemblies")

    def test_vop3_operand_widths_and_stream_alignment(self):
        # Verify opcode-specific operand widths from LLVM GFX10 MC test cases,
        # mixed-width forms, conversions, and byte-stream alignment with trailing s_endpgm.
        stream = bytes.fromhex(
            "fe0064d501050202"  # v_add_f64 v[254:255], v[1:2], v[2:3]
            "0200ffd683080200"  # v_lshlrev_b64 v[2:3], 3, v[4:5]
            "020068d5040d0000"  # v_ldexp_f64 v[2:3], v[4:5], s6
            "000076d502061204"  # v_mad_u64_u32 v[0:1], s0, v2, v3, v[4:5]
            "00026ed502091a04"  # v_div_scale_f64 v[0:1], s2, v[2:3], v[4:5], v[6:7]
            "000090d502010000"  # v_cvt_f64_f32_e64 v[0:1], v2
            "00008fd502010000"  # v_cvt_f32_f64_e64 v0, v[2:3]
            "000021d402090200"  # v_cmp_lt_f64_e64 s0, v[2:3], v[4:5]
            "000080d502091a04"  # v_nop (8-byte VOP3 slot)
            "0000abd502091a04"  # v_rcp_iflag_f32_e64 v0, v2
            "0000afd502091a04"  # v_rcp_f64_e64 v[0:1], v[2:3]
            "000040d502091a04"  # v_mad_legacy_f32 v0, v2, v4, v6
            "000050d502091a04"  # v_mullit_f32 v0, v2, v4, v6
            "000072d502091a04"  # v_vop3_172 (undecoded special form consumes 8 bytes)
            "05004bd501ff09040000803f"  # v_fma_f32 v5, v1, 0x3f800000, v2 (12-byte literal)
            "000081bf"          # s_endpgm
        )
        insns = disasm_rdna2(stream)
        self.assertEqual(
            [(i["opcode"], i["length"], i["assembly"]) for i in insns],
            [
                ("v_add_f64", 8, "v_add_f64 v[254:255], v[1:2], v[2:3]"),
                ("v_lshlrev_b64", 8, "v_lshlrev_b64 v[2:3], 3, v[4:5]"),
                ("v_ldexp_f64", 8, "v_ldexp_f64 v[2:3], v[4:5], s6"),
                ("v_mad_u64_u32", 8, "v_mad_u64_u32 v[0:1], s0, s2, v3, v[4:5]"),
                ("v_div_scale_f64", 8, "v_div_scale_f64 v[0:1], s2, v[2:3], v[4:5], v[6:7]"),
                ("v_cvt_f64_f32_e64", 8, "v_cvt_f64_f32_e64 v[0:1], v2"),
                ("v_cvt_f32_f64_e64", 8, "v_cvt_f32_f64_e64 v0, v[2:3]"),
                ("v_cmp_lt_f64_e64", 8, "v_cmp_lt_f64_e64 s0, v[2:3], v[4:5]"),
                ("v_nop", 8, "v_nop"),
                ("v_rcp_iflag_f32_e64", 8, "v_rcp_iflag_f32_e64 v0, v2"),
                ("v_rcp_f64_e64", 8, "v_rcp_f64_e64 v[0:1], v[2:3]"),
                ("v_mad_legacy_f32", 8, "v_mad_legacy_f32 v0, v2, v4, v6"),
                ("v_mullit_f32", 8, "v_mullit_f32 v0, v2, v4, v6"),
                ("v_vop3_172", 8, "v_vop3_172 v0, v2, v4, v6"),
                ("v_fma_f32", 12, "v_fma_f32 v5, v1, 0x3f800000, v2"),
                ("s_endpgm", 4, "s_endpgm"),
            ],
        )

    def test_vop3p_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import VOP3P_OPS, VOP3P_3SRC
        for op, (expected_name, _) in VOP3P_OPS.items():
            is_3src = op in VOP3P_3SRC
            w0 = 0xcc000000 | (op << 16) | 1  # vdst = 1
            w1 = (256 + 2) | ((256 + 3) << 9)  # v2, v3
            if is_3src:
                w1 |= ((256 + 4) << 18)  # v4
            if not expected_name.startswith("v_fma_mix"):
                w1 |= (1 << 27) | (1 << 28)
                if is_3src:
                    w0 |= (1 << 14)
            raw = struct.pack("<2I", w0, w1)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding VOP3P op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 8)
            if is_3src:
                self.assertEqual(insns[0]["assembly"], f"{expected_name} v1, v2, v3, v4")
            else:
                self.assertEqual(insns[0]["assembly"], f"{expected_name} v1, v2, v3")

        # Truncated VOP3P word
        truncated_raw = struct.pack("<I", 0xcc000000)
        insns = disasm_rdna2(truncated_raw)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "vop3p_truncated")
        self.assertIn("<truncated>", insns[0]["assembly"])

        # Literal source in VOP3P
        w0_lit = 0xcc000000 | (0x11 << 16) | 25  # v_pk_min_f16 v25
        w1_lit = (256 + 24) | (255 << 9) | (1 << 27) | (1 << 28)  # v24, literal
        raw_lit = struct.pack("<3I", w0_lit, w1_lit, 0x40007e00)
        insns = disasm_rdna2(raw_lit)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "v_pk_min_f16")
        self.assertEqual(insns[0]["length"], 12)
        self.assertEqual(insns[0]["assembly"], "v_pk_min_f16 v25, v24, 0x40007e00")

        # Truncated literal
        raw_lit_trunc = struct.pack("<2I", w0_lit, w1_lit)
        insns = disasm_rdna2(raw_lit_trunc)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "vop3p_truncated")
        self.assertIn("<truncated>", insns[0]["assembly"])

    def test_unsupported_vop3p_keeps_literal_stream_aligned(self):
        # Packed add with a source-0 literal and a reserved source-2 selection bit.
        first = 0xcc0a0004 | (1 << 13)
        second = 255 | (129 << 9) | (3 << 27)
        stream = struct.pack("<4I", first, second, 0x3f800000, 0xbf810000)
        decoded = disasm_rdna2(stream)
        self.assertEqual([(row["opcode"], row["length"]) for row in decoded],
                         [("v_pk_add_u16_unsupported", 12), ("s_endpgm", 4)])

    def test_vop3p_distinguishes_non_default_modifiers_and_selections(self):
        """VOP3P instructions must distinguish non-default op_sel, neg, and clamp from default forms."""
        # Base 2-source instruction: v_pk_add_u16 v4, v3, 1 (default op_sel_hi:[1,1])
        w0_base = 0xcc000000 | (0x0a << 16) | 4
        w1_base = 259 | (129 << 9) | (1 << 27) | (1 << 28)

        default_insn = disasm_rdna2(struct.pack("<2I", w0_base, w1_base))[0]["assembly"]
        self.assertEqual(default_insn, "v_pk_add_u16 v4, v3, 1")

        # Modified op_sel on src0 (w0[11] = 1) -> op_sel:[1,0]
        op_sel_insn = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 11), w1_base))[0]["assembly"]
        self.assertEqual(op_sel_insn, "v_pk_add_u16 v4, v3, 1 op_sel:[1,0]")

        # Modified op_sel_hi on src0 (w1[27] = 0) -> op_sel_hi:[0,1]
        op_sel_hi_insn = disasm_rdna2(struct.pack("<2I", w0_base, w1_base & ~(1 << 27)))[0]["assembly"]
        self.assertEqual(op_sel_hi_insn, "v_pk_add_u16 v4, v3, 1 op_sel_hi:[0,1]")

        # Modified neg_lo on src0 (w1[29] = 1) -> neg_lo:[1,0]
        neg_lo_insn = disasm_rdna2(struct.pack("<2I", w0_base, w1_base | (1 << 29)))[0]["assembly"]
        self.assertEqual(neg_lo_insn, "v_pk_add_u16 v4, v3, 1 neg_lo:[1,0]")

        # Modified neg_hi on src0 (w0[8] = 1) -> neg_hi:[1,0]
        neg_hi_insn = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 8), w1_base))[0]["assembly"]
        self.assertEqual(neg_hi_insn, "v_pk_add_u16 v4, v3, 1 neg_hi:[1,0]")

        # Clamp (w0[15] = 1)
        clamp_insn = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 15), w1_base))[0]["assembly"]
        self.assertEqual(clamp_insn, "v_pk_add_u16 v4, v3, 1 clamp")

        # All variants have identical register operands but distinct assemblies
        disasms = [default_insn, op_sel_insn, op_sel_hi_insn, neg_lo_insn, neg_hi_insn, clamp_insn]
        self.assertEqual(len(set(disasms)), len(disasms), "VOP3P modifier variants collided")

        # 3-source instruction: v_pk_fma_f16 v1, v2, v3, v4
        w0_3src = 0xcc000000 | (0x0e << 16) | (1 << 14) | 1  # op_sel_hi[2] = 1
        w1_3src = (256 + 2) | ((256 + 3) << 9) | ((256 + 4) << 18) | (1 << 27) | (1 << 28)
        default_3src = disasm_rdna2(struct.pack("<2I", w0_3src, w1_3src))[0]["assembly"]
        self.assertEqual(default_3src, "v_pk_fma_f16 v1, v2, v3, v4")

        # Modified op_sel_hi on src2 (w0[14] = 0) -> op_sel_hi:[1,1,0]
        op_sel_hi_3src = disasm_rdna2(struct.pack("<2I", w0_3src & ~(1 << 14), w1_3src))[0]["assembly"]
        self.assertEqual(op_sel_hi_3src, "v_pk_fma_f16 v1, v2, v3, v4 op_sel_hi:[1,1,0]")

        # Modified neg_hi on src2 (w0[10] = 1) -> neg_hi:[0,0,1]
        neg_hi_3src = disasm_rdna2(struct.pack("<2I", w0_3src | (1 << 10), w1_3src))[0]["assembly"]
        self.assertEqual(neg_hi_3src, "v_pk_fma_f16 v1, v2, v3, v4 neg_hi:[0,0,1]")

        # Modified neg_lo on src2 (w1[31] = 1) -> neg_lo:[0,0,1]
        neg_lo_3src = disasm_rdna2(struct.pack("<2I", w0_3src, w1_3src | (1 << 31)))[0]["assembly"]
        self.assertEqual(neg_lo_3src, "v_pk_fma_f16 v1, v2, v3, v4 neg_lo:[0,0,1]")

        # Both 0 and 1 in bit 14 (LLVM MC default op_sel_hi = 0b111) decode to default 2-source form
        self.assertEqual(disasm_rdna2(struct.pack("<2I", w0_base | (1 << 14), w1_base))[0]["assembly"], "v_pk_add_u16 v4, v3, 1")

        # 2-source encoding with reserved bit set (e.g. w1[31], w0[10], w0[13], or src2 != 0) is flagged as unsupported
        for reserved_w0, reserved_w1 in [
            (w0_base, w1_base | (1 << 31)),          # neg_lo[2]
            (w0_base | (1 << 10), w1_base),          # neg_hi[2]
            (w0_base | (1 << 13), w1_base),          # op_sel[2]
            (w0_base, w1_base | (256 << 18)),        # src2 = v0
        ]:
            reserved_raw = struct.pack("<2I", reserved_w0, reserved_w1)
            res_rec = disasm_rdna2(reserved_raw)[0]
            self.assertEqual(res_rec["opcode"], "v_pk_add_u16_unsupported")
            self.assertIn("<unsupported_reserved_bits>", res_rec["assembly"])

    def test_vop3p_fma_mix_inline_abs_neg_modifiers(self):
        """v_fma_mix* must decode w0[10:8] as abs and w1[31:29] as neg inline, not neg_lo/neg_hi."""
        # Issue reproduction encoding: v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]
        repro = disasm_rdna2(bytes.fromhex("0c1920cc2c26524c"))
        self.assertEqual(len(repro), 1)
        self.assertEqual(
            repro[0]["assembly"],
            "v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]",
        )
        operands, op_sel, op_sel_hi, clamp = parse_fma_mix(repro[0]["assembly"].split(" ", 2)[2])
        self.assertEqual((operands, op_sel, op_sel_hi, clamp), (["|s44|", "-v19", "v20"], [1, 1, 0], [1, 0, 0], False))
        scalar = {"s44": "params[0].x"}
        vector = {"v19": "alu_19", "v20": "alu_20"}
        lowered = [
            fma_mix_operand(tok, bool(f16), bool(hi), scalar, vector, set(), set())
            for tok, f16, hi in zip(operands, op_sel, op_sel_hi)
        ]
        self.assertEqual(lowered, ["abs(widen16(asuint(params[0].x) >> 16))", "(-widen16(asuint(alu_19)))", "alu_20"])

        def encode_fma_mix(op, vdst, src0, src1, src2, abs_bits=(0, 0, 0), neg_bits=(0, 0, 0),
                           sel=(0, 0, 0), sel_hi=(0, 0, 0), use_clamp=False, literal=None):
            w0 = (
                0xcc000000
                | (op << 16)
                | (int(use_clamp) << 15)
                | (sel_hi[2] << 14)
                | (sel[2] << 13)
                | (sel[1] << 12)
                | (sel[0] << 11)
                | (abs_bits[2] << 10)
                | (abs_bits[1] << 9)
                | (abs_bits[0] << 8)
                | vdst
            )
            w1 = (
                (neg_bits[2] << 31)
                | (neg_bits[1] << 30)
                | (neg_bits[0] << 29)
                | (sel_hi[1] << 28)
                | (sel_hi[0] << 27)
                | (src2 << 18)
                | (src1 << 9)
                | src0
            )
            return struct.pack("<3I", w0, w1, literal) if literal is not None else struct.pack("<2I", w0, w1)

        opcodes = (
            (0x20, "v_fma_mix_f32"),
            (0x21, "v_fma_mixlo_f16"),
            (0x22, "v_fma_mixhi_f16"),
        )
        cases = (
            # (abs_bits, neg_bits, sel, sel_hi, use_clamp, expected_ops, expected_suffix)
            ((1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), False, "|v2|, v3, v4", ""),
            ((0, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), False, "v2, |v3|, v4", ""),
            ((0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0), False, "v2, v3, |v4|", ""),
            ((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0), False, "-v2, v3, v4", ""),
            ((0, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0), False, "v2, -v3, v4", ""),
            ((0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0), False, "v2, v3, -v4", ""),
            ((1, 1, 1), (1, 0, 1), (1, 1, 1), (1, 0, 1), True,
             "-|v2|, |v3|, -|v4|", " op_sel:[1,1,1] op_sel_hi:[1,0,1] clamp"),
        )
        vec = {"v2": "alu_2", "v3": "alu_3", "v4": "alu_4"}
        for op, name in opcodes:
            for abs_b, neg_b, sel, sel_hi, use_clamp, expected_ops, suffix in cases:
                with self.subTest(op=name, abs_b=abs_b, neg_b=neg_b, clamp=use_clamp):
                    raw = encode_fma_mix(op, 1, 258, 259, 260, abs_b, neg_b, sel, sel_hi, use_clamp)
                    insns = disasm_rdna2(raw)
                    self.assertEqual(len(insns), 1)
                    asm = insns[0]["assembly"]
                    self.assertEqual(asm, f"{name} v1, {expected_ops}{suffix}")
                    self.assertNotIn("neg_lo", asm)
                    self.assertNotIn("neg_hi", asm)
                    self.assertTrue(is_instruction_supported(asm))
                    parsed_ops, parsed_sel, parsed_hi, parsed_clamp = parse_fma_mix(asm.split(" ", 2)[2])
                    self.assertEqual((parsed_sel, parsed_hi, parsed_clamp), (list(sel), list(sel_hi), use_clamp))
                    for tok, f16, hi in zip(parsed_ops, parsed_sel, parsed_hi):
                        fma_mix_operand(tok, bool(f16), bool(hi), {}, vec, set(), set())

        # Literal operand with combined -|literal| modifiers must consume trailing DWORD and round-trip.
        lit_raw = encode_fma_mix(0x20, 1, 255, 259, 260, abs_bits=(1, 0, 0), neg_bits=(1, 0, 0), literal=0x3f800000)
        lit_insns = disasm_rdna2(lit_raw)
        self.assertEqual(len(lit_insns), 1)
        self.assertEqual(lit_insns[0]["length"], 12)
        self.assertEqual(lit_insns[0]["assembly"], "v_fma_mix_f32 v1, -|0x3f800000|, v3, v4")
        lit_ops, lit_sel, lit_hi, lit_clamp = parse_fma_mix(lit_insns[0]["assembly"].split(" ", 2)[2])
        self.assertEqual((lit_ops, lit_sel, lit_hi, lit_clamp), (["-|0x3f800000|", "v3", "v4"], [0, 0, 0], [0, 0, 0], False))

        # Standard VOP3P (e.g. v_dot2_f32_f16) retains neg_lo/neg_hi suffix modifiers.
        dot_raw = encode_fma_mix(0x13, 12, 274, 275, 278, abs_bits=(1, 0, 0), neg_bits=(1, 0, 0), sel_hi=(1, 1, 1))
        self.assertEqual(
            disasm_rdna2(dot_raw)[0]["assembly"],
            "v_dot2_f32_f16 v12, v18, v19, v22 neg_lo:[1,0,0] neg_hi:[1,0,0]",
        )

        # Negative controls: parse_fma_mix rejects neg_lo/neg_hi and f32 high-half selection; truncated literal fails safely.
        for bad_tail in (
            "s44, v19, v20 neg_lo:[0,1,0]",
            "s44, v19, v20 neg_hi:[1,0,0]",
            "v2, v3, v4 op_sel:[0,0,0] op_sel_hi:[1,0,0]",
        ):
            with self.assertRaises(ValueError):
                parse_fma_mix(bad_tail)

        trunc_raw = encode_fma_mix(0x20, 1, 255, 259, 260, abs_bits=(1, 0, 0), neg_bits=(1, 0, 0))
        trunc_insns = disasm_rdna2(trunc_raw)
        self.assertEqual(len(trunc_insns), 1)
        self.assertEqual(trunc_insns[0]["opcode"], "vop3p_truncated")
        self.assertEqual(trunc_insns[0]["length"], 8)
        self.assertIn("<truncated>", trunc_insns[0]["assembly"])

        # End-to-end disasm_rdna2 -> pixel_aot.lower check on a minimal pixel shader with asymmetric
        # v_fma_mix* modifiers and packed v_pk_* forms: swapping abs/neg across sources must change HLSL.
        def encode_vop3p_2src(op, vdst, src0, src1, sel=(0, 0), sel_hi=(1, 1), neg_lo=(0, 0), neg_hi=(0, 0)):
            w0 = (
                0xcc000000
                | (op << 16)
                | (1 << 14)
                | (sel[1] << 12)
                | (sel[0] << 11)
                | (neg_hi[1] << 9)
                | (neg_hi[0] << 8)
                | vdst
            )
            w1 = (
                (neg_lo[1] << 30)
                | (neg_lo[0] << 29)
                | (sel_hi[1] << 28)
                | (sel_hi[0] << 27)
                | (src1 << 9)
                | src0
            )
            return struct.pack("<2I", w0, w1)

        def build_fma_mix_shader(swap_modifiers=False):
            abs_b = (0, 1, 0) if swap_modifiers else (1, 0, 0)
            neg_b = (1, 0, 0) if swap_modifiers else (0, 1, 0)
            return b"".join([
                struct.pack("<I", 0xbe800300 | (21 << 16) | 0x80),                    # s_mov_b32 s21, 0
                struct.pack("<2I", 0xf4000000 | (0x02 << 18) | (28 << 6) | 10, 125 << 25),  # s_load_dwordx4 s[28:31], s[20:21], 0x0
                bytes.fromhex("7fc08cbf"),                                            # s_waitcnt lgkmcnt(0)
                struct.pack("<2I", 0xf4000000 | (0x0a << 18) | (28 << 6) | 14, 125 << 25),  # s_buffer_load_dwordx4 s[28:31], s[28:31], 0x0
                bytes.fromhex("7fc08cbf"),                                            # s_waitcnt lgkmcnt(0)
                struct.pack("<I", 0x7e000000 | (8 << 17) | (0x01 << 9) | 29),         # v_mov_b32_e32 v8, s29
                struct.pack("<I", 0x7e000000 | (9 << 17) | (0x01 << 9) | 30),         # v_mov_b32_e32 v9, s30
                struct.pack("<I", 0x7e000000 | (10 << 17) | (0x01 << 9) | 31),        # v_mov_b32_e32 v10, s31
                encode_fma_mix(0x20, 12, 28, 256 + 8, 256 + 9, abs_bits=abs_b, neg_bits=neg_b,
                               sel=(1, 1, 0), sel_hi=(1, 0, 0)),
                encode_fma_mix(0x22, 10, 256 + 8, 256 + 12, 256 + 9, abs_bits=(1, 0, 0), neg_bits=(1, 0, 0),
                               sel=(1, 0, 0), sel_hi=(1, 0, 0), use_clamp=True),
                struct.pack("<I", 0x7e000000 | (13 << 17) | (0x0b << 9) | (256 + 10)),  # v_cvt_f32_f16_e32 v13, v10
                encode_vop3p_2src(0x0f, 11, 28, 256 + 8, sel=(1, 0), sel_hi=(0, 1), neg_lo=(0, 1)),
                encode_vop3p_2src(0x10, 11, 256 + 11, 256 + 9, neg_hi=(1, 0)),
                encode_fma_mix(0x20, 14, 256 + 11, 0xf2, 0x80, sel=(1, 0, 0), sel_hi=(1, 0, 0)),
                struct.pack("<I", 0x7e000000 | (15 << 17) | (0x01 << 9) | 0xf2),      # v_mov_b32_e32 v15, 1.0
                struct.pack("<I", (0x2f << 25) | (16 << 17) | (13 << 9) | (256 + 12)),# v_cvt_pkrtz_f16_f32_e32 v16, v12, v13
                struct.pack("<I", (0x2f << 25) | (17 << 17) | (15 << 9) | (256 + 14)),# v_cvt_pkrtz_f16_f32_e32 v17, v14, v15
                bytes.fromhex("0f1c00f810110000"),                                    # exp mrt0 v16, v16, v17, v17 done compr vm
                bytes.fromhex("000081bf"),                                            # s_endpgm
            ])

        shader_asm = [rec["assembly"] for rec in disasm_rdna2(build_fma_mix_shader(False))]
        swapped_asm = [rec["assembly"] for rec in disasm_rdna2(build_fma_mix_shader(True))]
        self.assertIn("v_fma_mix_f32 v12, |s28|, -v8, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0]", shader_asm)
        self.assertIn("v_fma_mix_f32 v12, -s28, |v8|, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0]", swapped_asm)
        self.assertIn("v_fma_mixhi_f16 v10, -|v8|, v12, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp", shader_asm)
        self.assertIn("v_pk_add_f16 v11, s28, v8 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]", shader_asm)
        self.assertIn("v_pk_mul_f16 v11, v11, v9 neg_hi:[1,0]", shader_asm)

        hlsl = lower(shader_asm, 20, 24, None, constant_descriptor=0)
        swapped_hlsl = lower(swapped_asm, 20, 24, None, constant_descriptor=0)
        self.assertNotEqual(hlsl, swapped_hlsl)
        self.assertIn("mad(abs(widen16(asuint(params[0].x) >> 16)), (-widen16(asuint(alu5))), alu6)", hlsl)
        self.assertIn("mad((-widen16(asuint(params[0].x) >> 16)), abs(widen16(asuint(alu5))), alu6)", swapped_hlsl)
        self.assertIn("saturate(mad((-abs(widen16(asuint(alu5) >> 16))), alu8, alu6))", hlsl)

        bad_shader_asm = [
            "v_fma_mix_f32 v12, s28, v8, v9 op_sel:[1,1,0] neg_lo:[0,1,0]"
            if line.startswith("v_fma_mix_f32 v12,") else line
            for line in shader_asm
        ]
        with self.assertRaises(ValueError):
            lower(bad_shader_asm, 20, 24, None, constant_descriptor=0)

    def test_mubuf_opcode_table_synthetic_forms(self):
        """Every GFX1013 MUBUF opcode must decode with exact width, scanner recognition, and 8-byte alignment."""
        from disasm_rdna2 import MUBUF_OPS

        self.assertEqual(len(MUBUF_OPS), 72)
        for op, (expected_name, width, _) in MUBUF_OPS.items():
            with self.subTest(op=op, name=expected_name):
                w0 = 0xe0000000 | (op << 18)
                w1 = 0 if width == 0 else ((3 << 24) | (2 << 16) | (8 << 8))
                raw = struct.pack("<2I", w0, w1)
                insns = disasm_rdna2(raw)
                self.assertEqual(len(insns), 1)
                self.assertEqual(insns[0]["opcode"], expected_name)
                self.assertEqual(insns[0]["length"], 8)
                if width == 0:
                    self.assertEqual(insns[0]["assembly"], expected_name)
                else:
                    vdata = "v8" if width == 1 else f"v[8:{8 + width - 1}]"
                    self.assertEqual(insns[0]["assembly"], f"{expected_name} {vdata}, off, s[8:11], s3")
                self.assertTrue(is_opcode_recognized(expected_name))
                self.assertEqual(classify_isa_category(expected_name), "VMEM (Vector Memory / Buffer)")

        # Truncated MUBUF header reports mubuf_truncated instead of crashing or misdecoding.
        trunc = disasm_rdna2(struct.pack("<I", 0xe0300000))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "mubuf_truncated")
        self.assertIn("<truncated>", trunc[0]["assembly"])

        # Unknown MUBUF opcodes keep 8-byte alignment so subsequent instructions decode cleanly.
        stream = struct.pack("<3I", 0xe0000000 | (0x10 << 18), (3 << 24) | (2 << 16) | (8 << 8), 0xbf810000)
        decoded = disasm_rdna2(stream)
        self.assertEqual([(r["opcode"], r["length"]) for r in decoded], [("buffer_mubuf_10", 8), ("s_endpgm", 4)])

    def test_mtbuf_opcode_table_synthetic_forms(self):
        """Every GFX1013 MTBUF opcode and format identifier must decode accurately and stay aligned."""
        from disasm_rdna2 import MTBUF_FORMATS, MTBUF_OPS

        self.assertEqual(len(MTBUF_OPS), 16)
        for op, (expected_name, width, _) in MTBUF_OPS.items():
            with self.subTest(op=op, name=expected_name):
                op_lo = op & 0x7
                op_hi = (op >> 3) & 1
                w0 = 0xe8000000 | (22 << 19) | (op_lo << 16)
                w1 = (3 << 24) | (op_hi << 21) | (2 << 16) | (8 << 8)
                insns = disasm_rdna2(struct.pack("<2I", w0, w1))
                self.assertEqual(len(insns), 1)
                self.assertEqual(insns[0]["opcode"], expected_name)
                self.assertEqual(insns[0]["length"], 8)
                vdata = "v8" if width == 1 else f"v[8:{8 + width - 1}]"
                self.assertEqual(
                    insns[0]["assembly"],
                    f"{expected_name} {vdata}, off, s[8:11], s3 format:[BUF_FMT_32_FLOAT]",
                )
                self.assertTrue(is_opcode_recognized(expected_name))
                self.assertEqual(classify_isa_category(expected_name), "VMEM (Vector Memory / Buffer)")

        # Format 1 is LLVM's omitted default; named formats print in brackets; >=78 print numerically.
        for fmt, expected_suffix in [
            (0, " format:[BUF_FMT_INVALID]"),
            (1, ""),
            *[(code, f" format:[{name}]") for code, name in MTBUF_FORMATS.items() if code != 0],
            (78, " format:78"),
            (127, " format:127"),
        ]:
            w0 = 0xe8000000 | (fmt << 19)
            w1 = (3 << 24) | (2 << 16) | (5 << 8)
            asm = disasm_rdna2(struct.pack("<2I", w0, w1))[0]["assembly"]
            self.assertEqual(asm, f"tbuffer_load_format_x v5, off, s[8:11], s3{expected_suffix}")

        # Truncated MTBUF reports mtbuf_truncated and full instructions keep stream alignment.
        trunc = disasm_rdna2(struct.pack("<I", 0xe8b00000))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "mtbuf_truncated")
        self.assertIn("<truncated>", trunc[0]["assembly"])

        aligned = disasm_rdna2(struct.pack("<3I", 0xe8b00000, (3 << 24) | (2 << 16) | (5 << 8), 0xbf810000))
        self.assertEqual([(r["opcode"], r["length"]) for r in aligned], [("tbuffer_load_format_x", 8), ("s_endpgm", 4)])

    def test_mubuf_mtbuf_distinguishes_modifiers(self):
        """Buffer addressing, cache, LDS, TFE, and format modifiers must produce distinct disassembly."""
        base_w0 = 0xe0000000 | (0x0c << 18)  # buffer_load_dword
        base_w1 = (3 << 24) | (2 << 16) | (5 << 8) | 1
        variants = {
            "base": disasm_rdna2(struct.pack("<2I", base_w0, base_w1))[0]["assembly"],
            "offen": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 12), base_w1))[0]["assembly"],
            "idxen": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 13), base_w1))[0]["assembly"],
            "idxen+offen": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 12) | (1 << 13), base_w1))[0]["assembly"],
            "offset": disasm_rdna2(struct.pack("<2I", base_w0 | 64, base_w1))[0]["assembly"],
            "glc": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 14), base_w1))[0]["assembly"],
            "slc": disasm_rdna2(struct.pack("<2I", base_w0, base_w1 | (1 << 22)))[0]["assembly"],
            "dlc": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 15), base_w1))[0]["assembly"],
            "lds": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 16), base_w1))[0]["assembly"],
            "tfe": disasm_rdna2(struct.pack("<2I", base_w0, base_w1 | (1 << 23)))[0]["assembly"],
            "soffset_m0": disasm_rdna2(struct.pack("<2I", base_w0, (124 << 24) | (2 << 16) | (5 << 8) | 1))[0]["assembly"],
        }
        self.assertEqual(len(set(variants.values())), len(variants))
        self.assertEqual(variants["idxen+offen"], "buffer_load_dword v5, v[1:2], s[8:11], s3 idxen offen")
        self.assertEqual(variants["lds"], "buffer_load_dword off, s[8:11], s3 lds")
        self.assertEqual(variants["tfe"], "buffer_load_dword v5, off, s[8:11], s3 tfe")
        self.assertEqual(variants["soffset_m0"], "buffer_load_dword v5, off, s[8:11], m0")

        # Atomics suppress tfe, matching LLVM AMDGPU disassembly.
        atomic_asm = disasm_rdna2(struct.pack("<2I", 0xe0000000 | (0x32 << 18), base_w1 | (1 << 23)))[0]["assembly"]
        self.assertEqual(atomic_asm, "buffer_atomic_add v5, off, s[8:11], s3")

        # Trap-handler temporary buffer descriptors (srsrc idx 27..30 -> ttmp[0:3]..ttmp[12:15]).
        for srsrc_idx, t in ((27, 0), (28, 4), (29, 8), (30, 12)):
            ttmp_w1 = (3 << 24) | (srsrc_idx << 16) | (5 << 8)
            ttmp_asm = disasm_rdna2(struct.pack("<2I", base_w0, ttmp_w1))[0]["assembly"]
            self.assertEqual(ttmp_asm, f"buffer_load_dword v5, off, ttmp[{t}:{t + 3}], s3")

    def test_flat_global_scratch_opcode_tables_synthetic_forms(self):
        """All 130 GFX1013 FLAT, GLOBAL, and SCRATCH forms must decode cleanly and keep stream alignment."""
        from disasm_rdna2 import FLAT_OPS, GLOBAL_OPS, SCRATCH_OPS

        self.assertEqual(len(FLAT_OPS), 54)
        self.assertEqual(len(GLOBAL_OPS), 54)
        self.assertEqual(len(SCRATCH_OPS), 22)

        def fmt_v(reg: int, width: int) -> str:
            return f"v{reg}" if width == 1 else f"v[{reg}:{reg + width - 1}]"

        for seg, ops_table in ((0, FLAT_OPS), (2, GLOBAL_OPS), (1, SCRATCH_OPS)):
            for op, (expected_name, kind, vdst_w, vdata_w, _) in ops_table.items():
                with self.subTest(seg=seg, op=op, name=expected_name):
                    w0 = 0xdc000000 | (op << 18) | (seg << 14)
                    w1 = (4 << 24) | (0x7d << 16) | (8 << 8) | 2
                    insns = disasm_rdna2(struct.pack("<2I", w0, w1))
                    self.assertEqual(len(insns), 1)
                    self.assertEqual(insns[0]["opcode"], expected_name)
                    self.assertEqual(insns[0]["length"], 8)

                    vaddr = "v[2:3]" if seg in (0, 2) else "v2"
                    if kind == "load":
                        expected_ops = [fmt_v(4, vdst_w), vaddr]
                    elif kind == "store":
                        expected_ops = [vaddr, fmt_v(8, vdata_w)]
                    else:
                        expected_ops = [vaddr, fmt_v(8, vdata_w)]
                    if seg in (1, 2):
                        expected_ops.append("off")
                    self.assertEqual(insns[0]["assembly"], f"{expected_name} {', '.join(expected_ops)}")
                    self.assertTrue(is_opcode_recognized(expected_name))
                    self.assertEqual(classify_isa_category(expected_name), "VMEM (Vector Memory / Buffer)")

        # Truncated FLAT header reports flat_truncated instead of crashing.
        trunc = disasm_rdna2(struct.pack("<I", 0xdc300000))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "flat_truncated")
        self.assertIn("<truncated>", trunc[0]["assembly"])

        # Reserved segment=3 and unsupported scratch atomic keep 8-byte alignment before s_endpgm.
        for w0 in (0xdc30c000, 0xdcc84000):
            stream = struct.pack("<3I", w0, (0x7d << 16) | 2, 0xbf810000)
            decoded = disasm_rdna2(stream)
            self.assertEqual([r["length"] for r in decoded], [8, 4])
            self.assertEqual(decoded[1]["opcode"], "s_endpgm")

    def test_flat_global_scratch_distinguishes_modifiers_and_addressing(self):
        """Atomic glc return-value form, signed vs unsigned offsets, saddr, and LDS must stay distinct."""
        # Atomic glc=0 vs glc=1 controls whether vdst is emitted.
        for seg, base_w0, expected_no_rtn, expected_rtn in [
            (0, 0xdc000000 | (0x32 << 18), "flat_atomic_add v[2:3], v8", "flat_atomic_add v4, v[2:3], v8 glc"),
            (2, 0xdc000000 | (0x32 << 18) | (2 << 14), "global_atomic_add v[2:3], v8, off", "global_atomic_add v4, v[2:3], v8, off glc"),
        ]:
            w1 = (4 << 24) | (0x7d << 16) | (8 << 8) | 2
            no_rtn = disasm_rdna2(struct.pack("<2I", base_w0, w1))[0]["assembly"]
            rtn = disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 16), w1))[0]["assembly"]
            self.assertEqual(no_rtn, expected_no_rtn)
            self.assertEqual(rtn, expected_rtn)

        # FLAT offset is 12-bit unsigned while GLOBAL/SCRATCH offsets are 12-bit signed.
        w1_off = (4 << 24) | (0x7d << 16) | 2
        self.assertEqual(disasm_rdna2(struct.pack("<2I", 0xdc300800, w1_off))[0]["assembly"], "flat_load_dword v4, v[2:3] offset:2048")
        self.assertEqual(disasm_rdna2(struct.pack("<2I", 0xdc308800, w1_off))[0]["assembly"], "global_load_dword v4, v[2:3], off offset:-2048")
        self.assertEqual(disasm_rdna2(struct.pack("<2I", 0xdc304fff, w1_off))[0]["assembly"], "scratch_load_dword v4, v2, off offset:-1")

        # Global saddr forms: off vs s[2:3] vs vcc vs ttmp[0:1], plus lds/slc/dlc/glc.
        global_variants = {
            "off": disasm_rdna2(struct.pack("<2I", 0xdc308000, (4 << 24) | (0x7d << 16) | 2))[0]["assembly"],
            "s2": disasm_rdna2(struct.pack("<2I", 0xdc308000, (4 << 24) | (2 << 16) | 1))[0]["assembly"],
            "vcc": disasm_rdna2(struct.pack("<2I", 0xdc308000, (4 << 24) | (106 << 16) | 1))[0]["assembly"],
            "ttmp0": disasm_rdna2(struct.pack("<2I", 0xdc308000, (4 << 24) | (108 << 16) | 1))[0]["assembly"],
            "lds": disasm_rdna2(struct.pack("<2I", 0xdc30a000, (0 << 24) | (2 << 16) | 1))[0]["assembly"],
            "slc_dlc": disasm_rdna2(struct.pack("<2I", 0xdc329010, (4 << 24) | (2 << 16) | 1))[0]["assembly"],
        }
        self.assertEqual(len(set(global_variants.values())), len(global_variants))
        self.assertEqual(global_variants["vcc"], "global_load_dword v4, v1, vcc")
        self.assertEqual(global_variants["ttmp0"], "global_load_dword v4, v1, ttmp[0:1]")
        self.assertEqual(global_variants["lds"], "global_load_dword v1, s[2:3] lds")
        self.assertEqual(global_variants["slc_dlc"], "global_load_dword v4, v1, s[2:3] offset:16 slc dlc")

        # Scratch saddr vs vaddr addressing.
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", 0xdc304000, (4 << 24) | (2 << 16) | 1))[0]["assembly"],
            "scratch_load_dword v4, off, s2",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", 0xdc306010, (0 << 24) | (2 << 16) | 0))[0]["assembly"],
            "scratch_load_dword off, s2 offset:16 lds",
        )

    def test_ds_opcode_table_synthetic_forms(self):
        """All 154 GFX1013 DS opcodes must decode cleanly and invalid variants keep 8-byte alignment."""
        from disasm_rdna2 import DS_OPS

        self.assertEqual(len(DS_OPS), 154)

        def fmt_v(reg: int, width: int) -> str:
            return f"v{reg}" if width == 1 else f"v[{reg}:{reg + width - 1}]"

        for op, (expected_name, vdst_w, has_addr, data0_w, data1_w, offset_mode, gds_mode) in DS_OPS.items():
            with self.subTest(op=op, name=expected_name):
                gds = 1 if gds_mode == "gds_only" else 0
                w0 = 0xd8000000 | (op << 18) | (gds << 17)
                w1 = (24 << 24) | (16 << 16) | (8 << 8) | 2
                insns = disasm_rdna2(struct.pack("<2I", w0, w1))
                self.assertEqual(len(insns), 1)
                self.assertEqual(insns[0]["opcode"], expected_name)
                self.assertEqual(insns[0]["length"], 8)

                ops = []
                if vdst_w:
                    ops.append(fmt_v(24, vdst_w))
                if has_addr:
                    ops.append("v2")
                if data0_w:
                    ops.append(fmt_v(8, data0_w))
                if data1_w:
                    ops.append(fmt_v(16, data1_w))
                expected_asm = expected_name
                if ops:
                    expected_asm += f" {', '.join(ops)}"
                if gds:
                    expected_asm += " gds"
                self.assertEqual(insns[0]["assembly"], expected_asm)
                self.assertTrue(is_opcode_recognized(expected_name))
                self.assertEqual(classify_isa_category(expected_name), "LDS (Local Data Share)")

        # Truncated DS header reports ds_truncated instead of crashing.
        trunc = disasm_rdna2(struct.pack("<I", 0xd8000000 | (0x36 << 18)))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "ds_truncated")
        self.assertIn("<truncated>", trunc[0]["assembly"])

        # Unknown DS opcode, gds_only with gds=0, no_gds with gds=1, and ds_nop with non-zero offset
        # must all consume 8 bytes so following instructions stay aligned.
        invalid_w0s = [
            0xd8000000 | (0x16 << 18),
            0xd8000000 | (0x3f << 18),
            0xd8000000 | (0xb2 << 18) | (1 << 17),
            0xd8000000 | (0x14 << 18) | 4,
        ]
        for bad_w0 in invalid_w0s:
            stream = struct.pack("<3I", bad_w0, 0, 0xbf810000)
            decoded = disasm_rdna2(stream)
            self.assertEqual([r["length"] for r in decoded], [8, 4])
            self.assertEqual(decoded[1]["opcode"], "s_endpgm")

    def test_ds_distinguishes_offsets_swizzle_and_gds(self):
        """DS single offsets, dual offset0/offset1 pairs, swizzle patterns, and gds must stay distinct."""
        w1_read = (24 << 24) | 2
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", 0xd8000000 | (0x36 << 18) | 8721, w1_read))[0]["assembly"],
            "ds_read_b32 v24, v2 offset:8721",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", 0xd8000000 | (0x36 << 18) | (1 << 17) | 64, w1_read))[0]["assembly"],
            "ds_read_b32 v24, v2 offset:64 gds",
        )

        # Dual offsets emit non-zero offset0 / offset1 independently.
        dual_base = 0xd8000000 | (0x37 << 18)
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", dual_base, w1_read))[0]["assembly"],
            "ds_read2_b32 v[24:25], v2",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", dual_base | 17, w1_read))[0]["assembly"],
            "ds_read2_b32 v[24:25], v2 offset0:17",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", dual_base | (34 << 8), w1_read))[0]["assembly"],
            "ds_read2_b32 v[24:25], v2 offset1:34",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", dual_base | (34 << 8) | 17, w1_read))[0]["assembly"],
            "ds_read2_b32 v[24:25], v2 offset0:17 offset1:34",
        )

        # Swizzle modes: omitted zero, QUAD_PERM, SWAP, REVERSE, BROADCAST, BITMASK_PERM, decimal fallback.
        swz_base = 0xd8000000 | (0x35 << 18)
        for imm, suffix in (
            (0x0000, ""),
            (0x80e4, " offset:swizzle(QUAD_PERM,0,1,2,3)"),
            (0x041f, " offset:swizzle(SWAP,1)"),
            (0x1c1f, " offset:swizzle(REVERSE,8)"),
            (0x0058, " offset:swizzle(BROADCAST,8,2)"),
            (0x1234, ' offset:swizzle(BITMASK_PERM,"10i01")'),
            (0xc0e4, " offset:49380"),
        ):
            self.assertEqual(
                disasm_rdna2(struct.pack("<2I", swz_base | imm, w1_read))[0]["assembly"],
                f"ds_swizzle_b32 v24, v2{suffix}",
            )

    def test_cli_hex_entry_point(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "disasm_rdna2.py"
        res = subprocess.run(
            [sys.executable, "-B", str(script), "--hex", "000080bf 000081bf"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("s_nop 0", res.stdout)
        self.assertIn("s_endpgm", res.stdout)

    def test_llvm_comparison(self):
        llvm_lib_env = os.environ.get("LLVM_LIB")
        if not llvm_lib_env:
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")

        if llvm_disassemble is None:
            self.fail("pixel_aot.disassemble helper could not be imported")

        resolved_lib = resolve_llvm_lib(llvm_lib_env)

        fixture_instructions = [
            "s_nop 0",
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt_expcnt null, 0",
            "v_mov_b32_e32 v0, v1",
            "v_mov_b32_e32 v0, s1",
            "v_mov_b32_e32 v2, 1.0",
            "v_mov_b32_e32 v3, 0.5",
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v4, src_shared_base",
            "v_mov_b32_e32 v4, src_pops_exiting_wave_id",
            "v_mov_b32_e32 v4, src_scc",
            "v_mov_b32_e32 v4, src_lds_direct",
            "v_add_f32_e32 v3, 0.15915494, v2",
            "v_add_f64 v[0:1], 0.15915494309189532, v[2:3]",
            # Issue #746: VOP1 destination is an SGPR; the LLVM 15 gfx1013 CI job
            # assembles and disassembles this line in test_llvm_comparison.
            "v_readfirstlane_b32 s2, v0",
            "v_cndmask_b32_e32 v3, v1, v2, vcc_lo",
            "v_add_f32_e32 v3, v1, v2",
            "v_sub_f32_e32 v3, v1, v2",
            "v_subrev_f32_e32 v3, v1, v2",
            "v_mul_legacy_f32_e32 v3, v1, v2",
            "v_mul_f32_e32 v3, v1, v2",
            "v_min_f32_e32 v3, v1, v2",
            "v_max_f32_e32 v3, v1, v2",
            "v_min_i32_e32 v3, v1, v2",
            "v_max_i32_e32 v3, v1, v2",
            "v_min_u32_e32 v3, v1, v2",
            "v_max_u32_e32 v3, v1, v2",
            "v_lshrrev_b32_e32 v3, v1, v2",
            "v_ashrrev_i32_e32 v3, v1, v2",
            "v_lshlrev_b32_e32 v3, v1, v2",
            "v_and_b32_e32 v3, v1, v2",
            "v_or_b32_e32 v3, v1, v2",
            "v_xor_b32_e32 v3, v1, v2",
            "v_add_nc_u32_e32 v3, v1, v2",
            "v_sub_nc_u32_e32 v3, v1, v2",
            "v_subrev_nc_u32_e32 v3, v1, v2",
            "v_fmac_f32_e32 v3, v1, v2",
            "v_mac_f32_e32 v3, v1, v2",
            "v_cvt_pkrtz_f16_f32_e32 v3, v1, v2",
            "v_cvt_f32_ubyte0_e32 v5, v1",
            "v_cvt_f32_ubyte1_e32 v5, v1",
            "v_cvt_f32_ubyte2_e32 v5, v1",
            "v_cvt_f32_ubyte3_e32 v5, v1",
            "v_fract_f32_e32 v5, v1",
            "v_trunc_f32_e32 v5, v1",
            "v_ceil_f32_e32 v5, v1",
            "v_rndne_f32_e32 v5, v1",
            "v_floor_f32_e32 v5, v1",
            "v_mov_b32_e32 v4, 0x12345678",
            "v_fmamk_f32 v5, v1, 0x11213141, v3",
            "v_fmaak_f32 v5, v1, v2, 0x11213141",
            "v_interp_p1_f32_e32 v1, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr1.y",
            "s_mov_b32 s2, 0x12345678",
            "s_and_saveexec_b32 s2, s0",
            "s_or_saveexec_b32 exec_lo, s1",
            "s_andn2_wrexec_b32 exec_lo, s1",
            "s_and_saveexec_b64 exec, s[2:3]",
            "s_andn2_wrexec_b64 exec, s[2:3]",
            "s_brev_b32 s2, s0",
            "s_brev_b64 s[2:3], s[0:1]",
            "s_bcnt0_i32_b32 s2, s0",
            "s_bcnt0_i32_b64 s2, s[0:1]",
            "s_bcnt1_i32_b32 s2, s0",
            "s_bcnt1_i32_b64 s2, s[0:1]",
            "s_ff0_i32_b32 s2, s0",
            "s_ff0_i32_b64 s2, s[0:1]",
            "s_ff1_i32_b32 s2, s0",
            "s_ff1_i32_b64 s2, s[0:1]",
            "s_flbit_i32_b32 s2, s0",
            "s_flbit_i32_b64 s2, s[0:1]",
            "s_getpc_b64 s[0:1]",
            "s_setpc_b64 s[0:1]",
            "s_rfe_b64 s[0:1]",
            "s_bfe_u32 s0, s1, s2",
            "s_bfe_i32 s0, s1, s2",
            "s_bfe_u64 s[0:1], s[2:3], s4",
            "s_bfe_i64 s[0:1], s[2:3], s4",
            "s_mul_hi_u32 s0, s1, s2",
            "s_mul_hi_i32 s0, s1, s2",
            "v_mul_i32_i24_e32 v5, v1, v2",
            "v_mul_hi_i32_i24_e32 v5, v1, v2",
            "v_mul_u32_u24_e32 v5, v1, v2",
            "v_mul_hi_u32_u24_e32 v5, v1, v2",
            "v_xnor_b32_e32 v5, v1, v2",
            "v_readlane_b32 s5, v1, s2",
            "v_writelane_b32 v5, 0, s2",
            "s_add_u32 s3, s2, 0x100",
            "s_load_dwordx8 s[20:27], s[2:3], 0x0",
            "s_buffer_load_dword s5, s[4:7], s0",
            "s_buffer_load_dwordx4 s[20:23], s[4:7], 0x20",
            "image_load v0, v[2:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "image_sample_lz v[64:66], [v32, v16], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY",
            "image_sample_l v[64:66], [v32, v16, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY",
            "image_sample_b v[64:66], [v32, v16, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY",
            "exp mrt0 v4, off, off, off done",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_cmp_eq_i32 s0, s1",
            "s_cmp_eq_u32 s0, s1",
            "s_cmp_gt_i32 s2, s3",
            "s_cmp_eq_u64 s[0:1], s[2:3]",
            "s_bitcmp0_b32 s0, s1",
            "s_bitcmp0_b64 s[4:5], s6",
            "v_pk_add_u16 v4, v3, 1",
            "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]",
            "v_pk_mul_f16 v23, v22, v20 neg_hi:[1,0]",
            "v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]",
            "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp",
            "v_cmp_lt_f32_e32 vcc_lo, v1, v2",
            "v_cmp_eq_u32_e32 vcc_lo, s1, v2",
            "v_cmpx_gt_i32_e32 v1, v2",
            "buffer_load_dword v5, off, s[8:11], s3",
            "buffer_load_dword v5, v[0:1], s[8:11], 0 idxen offen offset:4095 glc slc dlc",
            "buffer_load_dword off, s[8:11], s3 lds",
            "buffer_load_dword v5, off, s[8:11], null tfe",
            "buffer_load_dword v5, off, ttmp[0:3], s3",
            "buffer_load_format_d16_xyzw v[5:6], off, s[8:11], 0",
            "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen",
            "buffer_atomic_add v5, v0, s[8:11], s3 offen glc",
            "buffer_atomic_cmpswap_x2 v[5:8], v0, s[8:11], 0 offen glc",
            "buffer_gl0_inv",
            "buffer_gl1_inv",
            "tbuffer_load_format_x v5, off, s[8:11], s3 format:[BUF_FMT_32_FLOAT]",
            "tbuffer_load_format_xyzw v[5:8], v0, s[8:11], s3 format:[BUF_FMT_32_32_32_32_FLOAT] offen offset:64 glc slc dlc",
            "tbuffer_store_format_d16_xy v5, v[0:1], s[8:11], 0 format:[BUF_FMT_16_16_FLOAT] idxen offen",
            "tbuffer_load_format_x v5, off, s[8:11], s3",
            "tbuffer_load_format_x v5, off, s[8:11], s3 format:127",
            "flat_load_dword v4, v[2:3]",
            "flat_load_dwordx4 v[4:7], v[2:3] offset:2047 glc slc dlc",
            "flat_store_dword v[2:3], v8 offset:16",
            "flat_atomic_add v[2:3], v8",
            "flat_atomic_cmpswap_x2 v[4:5], v[2:3], v[8:11] glc",
            "global_load_dword v4, v[2:3], off",
            "global_load_dword v4, v1, s[2:3] offset:-16 glc slc dlc",
            "global_load_dword v1, s[2:3] lds",
            "global_store_byte_d16_hi v1, v8, s[2:3] offset:-2048",
            "global_atomic_add v1, v8, s[2:3]",
            "global_atomic_add v4, v1, v8, s[2:3] glc",
            "scratch_load_dword v4, v1, off offset:16",
            "scratch_load_dword v4, off, s2 offset:-16 glc slc dlc",
            "scratch_load_dword off, s2 offset:16 lds",
            "scratch_store_dwordx2 off, v[8:9], s2 offset:-32",
            "ds_nop",
            "ds_read_b32 v24, v2",
            "ds_read_b32 v24, v2 offset:64",
            "ds_read_b64 v[24:25], v2 offset:16",
            "ds_read_b128 v[24:27], v2",
            "ds_read2_b32 v[24:25], v2 offset0:4 offset1:8",
            "ds_read2st64_b64 v[24:27], v2 offset0:1 offset1:2",
            "ds_write_b32 v2, v8 offset:128",
            "ds_write2_b32 v2, v8, v16 offset0:4 offset1:8",
            "ds_add_rtn_u32 v24, v2, v8 offset:8 gds",
            "ds_mskor_b32 v2, v8, v16 offset:16",
            "ds_wrxchg2_rtn_b32 v[24:25], v2, v8, v16 offset0:1 offset1:2",
            "ds_swizzle_b32 v24, v2",
            "ds_swizzle_b32 v24, v2 offset:swizzle(QUAD_PERM,3,2,1,0)",
            "ds_swizzle_b32 v24, v2 offset:swizzle(SWAP,16)",
            "ds_swizzle_b32 v24, v2 offset:swizzle(REVERSE,8)",
            "ds_swizzle_b32 v24, v2 offset:swizzle(BROADCAST,4,2)",
            'ds_swizzle_b32 v24, v2 offset:swizzle(BITMASK_PERM,"1100p")',
            "ds_append v24 offset:12 gds",
            "ds_consume v24",
            "ds_ordered_count v24, v2 offset:4 gds",
            "ds_gws_sema_release_all gds",
            "ds_gws_barrier v2 offset:8 gds",
            "ds_permute_b32 v24, v2, v8 offset:4",
            "ds_bpermute_b32 v24, v2, v8",
            "ds_write_addtid_b32 v8 offset:16",
            "ds_read_addtid_b32 v24 offset:16",
            "ds_add_src2_f32 v2 offset:32",
            "ds_condxchg32_rtn_b64 v[24:25], v2, v[8:9]",
            "v_mad_f32 v0, v1, v2, v3",
            "v_fma_f32 v0, v1, v2, v3",
            "v_add_f32_e64 v3, v1, v2",
            "v_cndmask_b32_e64 v14, 0, 1.0, vcc_lo",
            "v_add_co_u32 v1, vcc_lo, v2, v3",
            "s_endpgm",
        ]

        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])

        if not assembler or not objcopy:
            self.fail(
                f"Designated LLVM comparison job requires llvm-mc and llvm-objcopy "
                f"(assembler={assembler}, objcopy={objcopy})"
            )

        try:
            emitted_bytes = assemble_with_llvm_mc(fixture_instructions, assembler, objcopy)
        except Exception as e:
            self.fail(f"Failed to assemble comparison fixture with llvm-mc: {e}")

        self.assertGreater(len(emitted_bytes), 0, "llvm-mc emitted empty machine code")
        self.assertEqual(len(emitted_bytes) % 4, 0, "Emitted bytes length must be multiple of 4")

        try:
            llvm_lines = llvm_disassemble(emitted_bytes, resolved_lib)
        except Exception as e:
            self.fail(
                f"Designated LLVM comparison failed to load or run with '{llvm_lib_env}' "
                f"(resolved: '{resolved_lib}'): {e}"
            )

        rdna2_insns = disasm_rdna2(emitted_bytes)
        self.assertEqual(
            len(llvm_lines), len(fixture_instructions),
            f"LLVM disassembled instruction count mismatch: {llvm_lines}"
        )
        self.assertEqual(
            len(rdna2_insns), len(fixture_instructions),
            f"RDNA 2 disassembled instruction count mismatch: {rdna2_insns}"
        )

        for i, (orig_inst, llvm_line, rdna2_rec) in enumerate(zip(fixture_instructions, llvm_lines, rdna2_insns)):
            orig_op = orig_inst.split()[0].lower()
            llvm_op = llvm_line.split()[0].lower()
            rdna2_op = rdna2_rec["opcode"].lower()
            self.assertEqual(
                llvm_op, orig_op,
                f"Instruction {i}: LLVM opcode '{llvm_op}' does not match original '{orig_op}'"
            )
            self.assertEqual(
                rdna2_op, orig_op,
                f"Instruction {i}: disasm_rdna2 opcode '{rdna2_op}' does not match original '{orig_op}'"
            )
            self.assertEqual(
                rdna2_op, llvm_op,
                f"Instruction {i}: disasm_rdna2 opcode '{rdna2_op}' does not match LLVM opcode '{llvm_op}'"
            )
            if orig_op.startswith(("v_", "exp", "image_", "buffer_", "tbuffer_", "flat_", "global_", "scratch_", "ds_")):
                self.assertEqual(
                    " ".join(rdna2_rec["assembly"].lower().split()),
                    " ".join(llvm_line.lower().split()),
                    f"Instruction {i}: disasm_rdna2 '{rdna2_rec['assembly']}' does not match LLVM '{llvm_line}'"
                )

    def test_interp_f16_llvm_round_trip(self):
        """Issue #938 step 2: the binary16 interpolation forms round-trip through LLVM 15 with every field intact.

        Each line is assembled by llvm-mc, decoded by disasm_rdna2, compared with LLVM's text, and the decoded text
        is assembled again and must give the same bytes. Neighbouring lines differ in one field (attribute,
        channel, high, a source, a modifier, clamp or the output modifier), so a dropped or swapped field changes
        the bytes. Inline-constant sources are left out: LLVM 15 prints them as `/*invalid immediate*/`, which it
        cannot assemble again (test_interp_f16_uses_llvm_spelling pins that text).
        """
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])
        if not assembler or not objcopy:
            self.fail(f"interp round trip requires llvm-mc and llvm-objcopy (assembler={assembler}, objcopy={objcopy})")
        source = [
            "v_interp_p1ll_f16 v1, v2, attr0.x",
            "v_interp_p1ll_f16 v1, v2, attr1.x",          # attribute
            "v_interp_p1ll_f16 v1, v2, attr1.y",          # channel
            "v_interp_p1ll_f16 v1, v2, attr1.y high",     # high half
            "v_interp_p1ll_f16 v1, s2, attr1.y high",     # SGPR source
            "v_interp_p1ll_f16 v1, -|v2|, attr63.w clamp div:2",
            "v_interp_p1lv_f16 v1, v2, attr1.y, v3",
            "v_interp_p1lv_f16 v1, v3, attr1.y, v2",      # sources swapped
            "v_interp_p1lv_f16 v1, v2, attr1.y, -v3 mul:4",
            "v_interp_p1lv_f16 v255, |m0|, attr32.z, exec_lo high mul:2",
            "v_interp_p2_f16 v1, v2, attr2.z, v3",
            "v_interp_p2_f16 v1, v2, attr2.z, v3 high",
            "v_interp_p2_f16 v1, -v2, attr2.z, |v3| clamp",
            "v_interp_p2_f16 v200, vcc_lo, attr17.x, s105",
            "s_endpgm",
        ]
        emitted = assemble_with_llvm_mc(source, assembler, objcopy)
        llvm_lines = llvm_disassemble(emitted, resolved_lib)
        decoded = [rec["assembly"] for rec in disasm_rdna2(emitted)]
        self.assertEqual(llvm_lines, source)
        self.assertEqual(decoded, llvm_lines)
        self.assertEqual(assemble_with_llvm_mc(decoded, assembler, objcopy), emitted)
        records = [emitted[i:i + 8] for i in range(0, 8 * (len(source) - 1), 8)]
        self.assertEqual(len(set(records)), len(records), "each line must encode differently")

    def test_hwreg_llvm_round_trip(self):
        """Issue #938: hwreg operands round-trip through LLVM 15 with register, offset and width intact.

        Each line is assembled by llvm-mc, decoded by disasm_rdna2, compared with LLVM's text, and the decoded
        text is assembled again and must give the same bytes. Neighbouring lines differ in one field only (the
        register, the offset or the width), and the offsets and widths are unequal where it matters, so a decoder
        that swaps or drops a field produces different bytes.
        """
        if not os.environ.get("LLVM_LIB"):
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")
        resolved_lib = resolve_llvm_lib(os.environ["LLVM_LIB"])
        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])
        if not assembler or not objcopy:
            self.fail(f"hwreg round trip requires llvm-mc and llvm-objcopy (assembler={assembler}, objcopy={objcopy})")

        source = [
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
            "s_setreg_imm32_b32 hwreg(HW_REG_STATUS, 2, 2), 3",   # register only
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 3, 2), 3",     # offset only
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 3), 3",     # width only
            "s_setreg_imm32_b32 hwreg(HW_REG_TRAPSTS, 1, 5), 0x41",
            "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0",
            "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_HI), 0x400",
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), -1",
            "s_setreg_imm32_b32 hwreg(4), 0x41",                  # numeric fallback, whole register
            "s_getreg_b32 s4, hwreg(HW_REG_MODE)",
            "s_getreg_b32 s5, hwreg(HW_REG_HW_ID1, 8, 4)",
            "s_getreg_b32 s7, hwreg(HW_REG_GPR_ALLOC, 6, 9)",
            "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s4",
            "s_setreg_b32 hwreg(63, 31, 1), m0",                  # numeric fallback, partial field
            "s_endpgm",
        ]
        emitted = assemble_with_llvm_mc(source, assembler, objcopy)
        llvm_lines = llvm_disassemble(emitted, resolved_lib)
        decoded = [rec["assembly"] for rec in disasm_rdna2(emitted)]
        self.assertEqual(llvm_lines, source)
        self.assertEqual(decoded, llvm_lines)
        self.assertEqual(assemble_with_llvm_mc(decoded, assembler, objcopy), emitted)
        # The first four lines are 8 bytes each; the three one-field variants must encode differently.
        self.assertEqual(len({emitted[i:i + 8] for i in range(0, 32, 8)}), 4)


class TestRdna2Extractor(unittest.TestCase):
    """Test shader extraction from RPF archives against known source-owned fixtures."""

    def test_extract_bytes_match_fixture(self):
        known_code = bytes.fromhex(
            "000080bf"  # s_nop 0
            "7f008cbf"  # s_waitcnt
            "000081bf"  # s_endpgm
        )
        record = envelope(known_code, b"PS_SourceOwned_Fixture")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "synthetic_shaders.rpf"
            archive.write_bytes(b"RPF_MAGIC_HEADER_PAD\0" + record)

            output_dir = tmp_path / "extracted"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})

            extracted_file = out_path / "pixel" / "PS_SourceOwned_Fixture.bin"
            self.assertTrue(extracted_file.is_file(), "Extracted binary file not found")
            extracted_bytes = extracted_file.read_bytes()

            # Byte-for-byte exact check against source-owned input
            self.assertEqual(extracted_bytes, known_code)

            # Check manifest.json metadata
            self.assertTrue(manifest_file.is_file(), "Manifest not generated")
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 1)
            self.assertEqual(manifest["shaders"][0]["name"], "PS_SourceOwned_Fixture")
            self.assertEqual(manifest["shaders"][0]["native_size"], len(known_code))
            self.assertEqual(manifest["shaders"][0]["stage"], "pixel")

    def test_duplicate_record_names(self):
        """Verify that duplicate record names produce distinct files without overwriting."""
        code_1 = bytes.fromhex("000080bf 000081bf")  # s_nop 0, s_endpgm
        code_2 = bytes.fromhex("000382be 000081bf")  # s_mov_b32, s_endpgm
        record_1 = envelope(code_1, b"PS_DuplicateName")
        record_2 = envelope(code_2, b"PS_DuplicateName")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "dups.rpf"
            archive.write_bytes(b"RPF\0" + record_1 + record_2)

            output_dir = tmp_path / "extracted_dups"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})

            file_1 = out_path / "pixel" / "PS_DuplicateName.bin"
            file_2 = out_path / "pixel" / "PS_DuplicateName_1.bin"

            self.assertTrue(file_1.is_file(), "First duplicate shader file not found")
            self.assertTrue(file_2.is_file(), "Second duplicate disambiguated file not found")

            self.assertEqual(file_1.read_bytes(), code_1)
            self.assertEqual(file_2.read_bytes(), code_2)

            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 2)
            self.assertEqual(manifest["shaders"][0]["native_file"], "pixel/PS_DuplicateName.bin")
            self.assertEqual(manifest["shaders"][1]["native_file"], "pixel/PS_DuplicateName_1.bin")
            self.assertNotEqual(manifest["shaders"][0]["sha256"], manifest["shaders"][1]["sha256"])

    def test_cli_extraction(self):
        """Invoke extractor via its actual CLI entry point on synthetic fixture."""
        known_code = bytes.fromhex("000080bf 000081bf")
        record = envelope(known_code, b"PS_CliTest")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "cli_shaders.rpf"
            archive.write_bytes(b"RPF_MAGIC\0" + record)
            output_dir = tmp_path / "cli_out"
            script = Path(__file__).resolve().parents[1] / "tools" / "extract_shaders.py"
            res = subprocess.run(
                [sys.executable, "-B", str(script), str(archive), "--output-dir", str(output_dir), "--stage", "pixel"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(res.returncode, 0)
            self.assertTrue((output_dir / "pixel" / "PS_CliTest.bin").is_file())
            self.assertEqual((output_dir / "pixel" / "PS_CliTest.bin").read_bytes(), known_code)
            self.assertTrue((output_dir / "manifest.json").is_file())

    def test_cli_limit_must_be_positive(self):
        """A limit below one is rejected instead of extracting a record anyway."""
        records = [envelope(bytes.fromhex("000080bf 000081bf"), b"PS_First"),
                   envelope(bytes.fromhex("000382be 000081bf"), b"PS_Second")]
        script = Path(__file__).resolve().parents[1] / "tools" / "extract_shaders.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "limit.rpf"
            archive.write_bytes(b"RPF\0" + b"".join(records))
            for limit in ("0", "-1"):
                output_dir = tmp_path / f"limit_{limit}"
                res = subprocess.run(
                    [sys.executable, "-B", str(script), str(archive), "--output-dir", str(output_dir),
                     "--stage", "pixel", "--limit", limit],
                    capture_output=True, text=True,
                )
                self.assertEqual(res.returncode, 2, res.stdout + res.stderr)
                self.assertIn("--limit", res.stderr)
                self.assertFalse(output_dir.exists(), "no output may be written for a rejected limit")
            output_dir = tmp_path / "limit_1"
            subprocess.run(
                [sys.executable, "-B", str(script), str(archive), "--output-dir", str(output_dir),
                 "--stage", "pixel", "--limit", "1"],
                capture_output=True, text=True, check=True,
            )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 1)

    def test_extracted_names_round_trip_length_prefixed_records(self) -> None:
        code = bytes.fromhex("000080bf 000081bf")
        names = (b"PS_A", b"PS_" + b"S" * 28, b"!" + b"A" * 32,
                 b"PS_" + b"X" * 122, b"~" + b"B" * 125, b"~" + b"C" * 126)
        legacy_name = b"PS_Legacy"
        legacy = envelope(code, legacy_name)[1:]
        truncated = envelope(code, b"PS_Truncated_Length_Prefix_Fixture")[:-1]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            archive = work / "source_built.rpf"
            archive.write_bytes(b"RPF\0" + b"".join(envelope(code, name) for name in names) + legacy + truncated)
            output, manifest_path = extract_shaders(archive, work / "shaders", target_stages={"pixel"},
                                                    length_prefixed_names=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["total_extracted"], len(names) + 1)
            for source_name, record in zip((*names, legacy_name), manifest["shaders"], strict=True):
                name = source_name.decode("ascii")
                with self.subTest(length=len(source_name)):
                    self.assertEqual(record["name"], name)
                    self.assertEqual(record["native_file"], f"pixel/{name}.bin")
                    self.assertEqual((output / record["native_file"]).read_bytes(), code)
                    if source_name != legacy_name:
                        self.assertEqual(native_program(archive, record["name"])[1], code)

    def test_ambiguous_length_prefix_requires_explicit_mode(self) -> None:
        code = bytes.fromhex("000080bf 000081bf")
        legacy_name = b"PS_" + b"L" * 77
        prefixed_name = legacy_name[1:]
        data = envelope(code, legacy_name)[1:]
        self.assertEqual(data, envelope(code, prefixed_name))
        script = Path(__file__).resolve().parents[1] / "tools" / "extract_shaders.py"

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            archive = work / "ambiguous.rpf"
            archive.write_bytes(b"RPF\0" + data)
            for expected_name, options in ((legacy_name, []), (prefixed_name, ["--length-prefixed-names"])):
                name = expected_name.decode("ascii")
                with self.subTest(options=options):
                    output = work / name
                    subprocess.run([sys.executable, "-B", str(script), str(archive), "--output-dir", str(output),
                                    *options], capture_output=True, check=True)
                    manifest_path = output / "manifest.json"
                    record, = json.loads(manifest_path.read_text(encoding="utf-8"))["shaders"]
                    self.assertEqual(record["name"], name)
                    self.assertEqual(record["native_file"], f"pixel/{name}.bin")
                    self.assertEqual((output / record["native_file"]).read_bytes(), code)

    def test_stage_filtering(self):
        known_code = bytes.fromhex("000080bf 000081bf")
        record = envelope(known_code, b"PS_PixelOnly")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "test.rpf"
            archive.write_bytes(b"HEADER\0" + record)

            output_dir = tmp_path / "extracted_compute"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"compute"})
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 0)

    def test_false_positive_header_does_not_skip_following_record(self):
        """A candidate header whose deflate payload fails must not skip a valid record inside its span."""
        known_code = bytes.fromhex("000080bf 000081bf")
        valid_record = envelope(known_code, b"PS_AfterFalsePositive")
        bogus_stored = len(valid_record) + 16
        false_header = b"FAKE_HDR\0\0" + struct.pack("<4I", bogus_stored, 0, bogus_stored, 256)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "false_positive.rpf"
            archive.write_bytes(false_header + b"\x00\x01\x02\x03" + valid_record + bytes(16))

            output_dir = tmp_path / "extracted_fp"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})
            extracted_file = out_path / "pixel" / "PS_AfterFalsePositive.bin"
            self.assertTrue(extracted_file.is_file(), "Valid record after false-positive header was skipped")
            self.assertEqual(extracted_file.read_bytes(), known_code)
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 1)

    def test_oversized_claimed_span_with_unused_data_does_not_skip_following_record(self):
        """Cursor advancement must stop at consumed deflate bytes when unused_data holds another record."""
        code_1 = bytes.fromhex("000080bf 000081bf")
        code_2 = bytes.fromhex("000082be 000081bf")
        record_1 = envelope(code_1, b"PS_OversizedSpan")
        record_2 = envelope(code_2, b"PS_InsideUnusedTail")
        name_end = record_1.index(b"PS_OversizedSpan") + len(b"PS_OversizedSpan")
        stored_1, _, _, decoded_1 = struct.unpack_from("<4I", record_1, name_end + 2)
        payload_1 = record_1[name_end + 18:name_end + 18 + stored_1]
        oversized_stored = stored_1 + len(record_2)
        oversized_record_1 = (record_1[:name_end + 2]
                              + struct.pack("<4I", oversized_stored, 0, oversized_stored, decoded_1)
                              + payload_1)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "oversized_span.rpf"
            archive.write_bytes(b"RPF\0" + oversized_record_1 + record_2)

            output_dir = tmp_path / "extracted_oversized"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})
            file_1 = out_path / "pixel" / "PS_OversizedSpan.bin"
            file_2 = out_path / "pixel" / "PS_InsideUnusedTail.bin"
            self.assertTrue(file_1.is_file(), "First shader in oversized span not extracted")
            self.assertTrue(file_2.is_file(), "Second shader inside unused_data tail was skipped")
            self.assertEqual(file_1.read_bytes(), code_1)
            self.assertEqual(file_2.read_bytes(), code_2)
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 2)

    def test_disambiguated_stem_does_not_collide_with_existing_or_case_variant(self):
        """Disambiguated stems must not overwrite explicit _1 names or case variants."""
        codes = [
            bytes.fromhex("000080bf 000081bf"),
            bytes.fromhex("000082be 000081bf"),
            bytes.fromhex("000182be 000081bf"),
            bytes.fromhex("0000027e 000081bf"),
        ]
        records = [
            envelope(codes[0], b"PS_Dup"),
            envelope(codes[1], b"PS_Dup"),
            envelope(codes[2], b"PS_Dup_1"),
            envelope(codes[3], b"ps_dup"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "collisions.rpf"
            archive.write_bytes(b"RPF\0" + b"".join(records))

            output_dir = tmp_path / "extracted_collisions"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 4)
            paths = [entry["native_file"] for entry in manifest["shaders"]]
            self.assertEqual(len({p.casefold() for p in paths}), 4, f"Colliding output paths: {paths}")
            for entry, expected in zip(manifest["shaders"], codes):
                self.assertEqual((out_path / entry["native_file"]).read_bytes(), expected)

    def test_overlong_record_names_get_bounded_distinct_files(self):
        """A record name near the one-byte length limit must not make the output name unwritable."""
        codes = [bytes.fromhex("000080bf 000081bf"), bytes.fromhex("000382be 000081bf")]
        names = [b"PS_" + b"a" * 250 + b"1", b"PS_" + b"a" * 250 + b"2"]
        records = [envelope(code, name) for code, name in zip(codes, names)]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "long.rpf"
            archive.write_bytes(b"RPF\0" + b"".join(records))

            out_path, manifest_file = extract_shaders(archive, tmp_path / "extracted_long", target_stages={"pixel"})
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 2)
            self.assertEqual([entry["name"] for entry in manifest["shaders"]],
                             [name.decode("ascii") for name in names])
            paths = [entry["native_file"] for entry in manifest["shaders"]]
            self.assertEqual(len({p.casefold() for p in paths}), 2, f"Colliding output paths: {paths}")
            for path in paths:
                self.assertLessEqual(len(Path(path).name.encode("utf-8")), 160, path)
            for path, expected in zip(paths, codes):
                self.assertEqual((out_path / path).read_bytes(), expected)
        # Truncation counts bytes and never splits a multi-byte character.
        for name in ("é" * 200, "a€" * 100):
            stem = sanitize_filename(name)
            self.assertTrue(name.startswith(stem) and 0 < len(stem.encode("utf-8")) <= 128, stem)

    def test_empty_archive_produces_empty_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "empty.rpf"
            archive.write_bytes(b"")

            output_dir = tmp_path / "extracted_empty"
            _, manifest_file = extract_shaders(archive, output_dir, target_stages={"all"})
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 0)
            self.assertEqual(manifest["shaders"], [])

    def test_invalid_xbch_offset_or_size_rejected(self):
        known_code = bytes.fromhex("000080bf 000081bf")
        program_start = 256
        image = bytearray(program_start) + known_code
        image[:4] = b"FSSM"
        struct.pack_into("<I", image, 192, program_start - 40)
        compressed = zlib.compress(image)

        wrong_size_native = b"XBCH" + bytes(12) + struct.pack("<II", len(image) + 4, 0) + compressed
        wrong_size_program = (bytes(24) + b"XDAT" + struct.pack("<II", 1, 0)
                              + struct.pack("<III", 13, 48, len(wrong_size_native)) + wrong_size_native)
        self.assertIsNone(extract_native_rdna2_code(wrong_size_program))

        valid_native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + compressed
        overlapping_program = (b"XBCH" + bytes(20) + b"XDAT" + struct.pack("<II", 1, 0)
                               + struct.pack("<III", 13, 0, len(valid_native)) + valid_native)
        self.assertIsNone(extract_native_rdna2_code(overlapping_program))


class TestRdna2Scanner(unittest.TestCase):
    """Test scanner instruction recognition vs full shader translator acceptance."""

    def test_accepted_full_shader(self):
        """Canonical straight-line pixel shader accepted by pixel_aot translator."""
        canonical_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("canonical_ps", canonical_shader)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

    def test_accepted_full_shader_wave32(self):
        """Wave32 pixel shader accepted by pixel_aot translator and recognized by scanner."""
        wave32_shader = [
            "s_mov_b64 s14, exec_lo",
            "s_wqm_b64 exec_lo, exec_lo",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec_lo, exec_lo, s14",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("canonical_ps_wave32", wave32_shader)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

    def test_accepted_full_shader_wave32_direct_restore(self):
        """Wave32 pixel shader with direct exec_lo restore via s_mov_b64."""
        wave32_shader = [
            "s_mov_b64 s14, exec_lo",
            "s_wqm_b64 exec_lo, exec_lo",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_mov_b64 exec_lo, s14",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("canonical_ps_wave32_direct", wave32_shader)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

    def test_accepted_full_shader_wave32_b32_lifecycle(self):
        """Native Wave32 pixel shader with 32-bit save, WQM and restore operations."""
        wave32_shader = [
            "s_mov_b32 s14, exec_lo",
            "s_wqm_b32 exec_lo, exec_lo",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b32 exec_lo, exec_lo, s14",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("canonical_ps_wave32_b32", wave32_shader, wave_size=32)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

        # Low-only lifecycle rejected without validated wave32 profile
        res_default = analyze_shader_instructions("canonical_ps_wave32_b32", wave32_shader)
        self.assertFalse(res_default["translator_accepted"])
        self.assertIn("requires a validated wave32 guest profile", res_default["lower_error"])

        res_w64 = analyze_shader_instructions("canonical_ps_wave32_b32", wave32_shader, wave_size=64)
        self.assertFalse(res_w64["translator_accepted"])
        self.assertIn("requires a validated wave32 guest profile", res_w64["lower_error"])

        res_invalid = analyze_shader_instructions("canonical_ps_wave32_b32", wave32_shader, wave_size=16)
        self.assertFalse(res_invalid["translator_accepted"])
        self.assertIn("Invalid guest wave size: 16", res_invalid["lower_error"])

    def test_exec_mask_scanner_translator_agreement(self):
        """Scanner instruction support matches pixel_aot translator acceptance for exec mask ops."""
        from pixel_aot import lower
        from pixel_aot_check import assembly
        from pixel_cb_read import scalar_writes

        base = list(assembly(False))
        save_idx = base.index("s_mov_b64 s[50:51], exec")
        wqm_idx = base.index("s_wqm_b64 exec, exec")
        restore_idx = base.index("s_and_b64 exec, exec, s[50:51]")

        def translator_accepts_save(line):
            shader = list(base)
            shader[save_idx] = line
            wave_size = None
            if line.startswith("s_mov_b32"):
                shader[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
                shader[restore_idx] = "s_and_b32 exec_lo, exec_lo, s50"
                wave_size = 32
            elif "s50" in line:
                shader[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
                shader[restore_idx] = "s_and_b64 exec_lo, exec_lo, s50"
            try:
                lower(shader, 20, 24, [1, 2, 3, 4], wave_size=wave_size)
                return True
            except Exception:
                return False

        def translator_accepts_restore(line):
            shader = list(base)
            wave_size = None
            if line.startswith(("s_mov_b32", "s_and_b32")):
                shader[save_idx] = "s_mov_b32 s50, exec_lo"
                shader[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
                wave_size = 32
            elif "50" in line or "exec_lo" in line or "s14" in line:
                shader[save_idx] = "s_mov_b64 s50, exec_lo" if "50" in line else "s_mov_b64 s14, exec_lo"
                shader[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
            shader[restore_idx] = line
            try:
                lower(shader, 20, 24, [1, 2, 3, 4], wave_size=wave_size)
                return True
            except Exception:
                return False

        save_cases = [
            ("s_mov_b64 s[50:51], exec", True),
            ("s_mov_b64 s[50:51], exec_lo", True),
            ("s_mov_b64 s50, exec_lo", True),
            ("s_mov_b64 s50, exec", True),
            ("s_mov_b32 s50, exec_lo", True),
            ("s_mov_b32 s50, exec", True),
            ("s_mov_b32 s[50:51], exec", False),
            ("s_mov_b32 s[50:51], exec_lo", False),
            ("s_mov_b32 s50, exec_hi", False),
        ]

        restore_cases = [
            ("s_mov_b64 exec, s[50:51]", True),
            ("s_mov_b64 exec_lo, s[50:51]", True),
            ("s_mov_b64 exec, s50", True),
            ("s_mov_b64 exec_lo, s50", True),
            ("s_mov_b32 exec, s50", True),
            ("s_mov_b32 exec_lo, s50", True),
            ("s_mov_b32 exec, s[50:51]", False),
            ("s_mov_b32 exec_lo, s[50:51]", False),
            ("s_and_b64 exec, exec, s[50:51]", True),
            ("s_and_b64 exec_lo, exec_lo, s[50:51]", True),
            ("s_and_b64 exec, exec, s50", True),
            ("s_and_b64 exec_lo, exec_lo, s50", True),
            ("s_and_b32 exec, exec, s50", True),
            ("s_and_b32 exec_lo, exec_lo, s50", True),
            ("s_and_b32 exec, exec, s[50:51]", False),
            ("s_and_b32 exec_lo, exec_lo, s[50:51]", False),
            ("s_and_b64 exec, exec, s14", True),
        ]

        for line, expected in save_cases:
            self.assertEqual(is_instruction_supported(line), expected, f"scanner mismatch on {line}")
            self.assertEqual(translator_accepts_save(line), expected, f"translator mismatch on save {line}")

        for line, expected in restore_cases:
            self.assertEqual(is_instruction_supported(line), expected, f"scanner mismatch on {line}")
            self.assertEqual(translator_accepts_restore(line), expected, f"translator mismatch on restore {line}")

        self.assertEqual(scalar_writes("s_mov_b64 s50, exec_lo"), range(50, 52))
        self.assertEqual(scalar_writes("s_mov_b64 s[50:51], exec"), range(50, 52))
        self.assertEqual(scalar_writes("s_mov_b32 s50, exec_lo"), range(50, 51))
        self.assertEqual(scalar_writes("s_mov_b64 exec, s[50:51]"), ())
        self.assertEqual(scalar_writes("s_mov_b32 exec_lo, s50"), ())

        # Mixed-width stream: 64-bit WQM expansion followed by 32-bit low-only restore must be rejected
        mixed_shader = list(base)
        mixed_shader[save_idx] = "s_mov_b64 s50, exec_lo"
        mixed_shader[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
        mixed_shader[restore_idx] = "s_and_b32 exec_lo, exec_lo, s50"
        with self.assertRaises(ValueError):
            lower(mixed_shader, 20, 24, [1, 2, 3, 4])

        # Clobbered high SGPR: full-width restore must verify both halves intact
        clobbered_shader = list(base)
        clobbered_shader[save_idx] = "s_mov_b64 s50, exec_lo"
        clobbered_shader[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
        clobbered_shader.insert(restore_idx, "s_mov_b32 s51, 0")
        clobbered_shader[restore_idx + 1] = "s_mov_b64 exec, s50"
        with self.assertRaises(ValueError):
            lower(clobbered_shader, 20, 24, [1, 2, 3, 4])

        # Wave32 32-bit execution mask stream rejected without validated wave32 profile
        b32_shader = list(base)
        b32_shader[save_idx] = "s_mov_b32 s50, exec_lo"
        b32_shader[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
        b32_shader[restore_idx] = "s_and_b32 exec_lo, exec_lo, s50"
        with self.assertRaises(ValueError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_size=64)
        with self.assertRaises(ValueError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_size=16)
        with self.assertRaises(ValueError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_size="wave32")
        with self.assertRaises(TypeError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_profile="wave32")
        self.assertIsNotNone(lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_size=32))

    def test_rounding_instructions_accepted(self):
        """v_ceil_f32 and v_rndne_f32 are known RDNA 2 opcodes that pixel_aot lowers."""
        canonical = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_ceil_f32_e32 v12, v12",
            "v_rndne_f32_e64 v13, v13",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("rounding_ps", canonical)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])
    def test_trig_instructions_accepted(self):
        """v_sin_f32 and v_cos_f32 are RDNA 2 opcodes that pixel_aot lowers in both encodings."""
        trig_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_sin_f32_e32 v12, v12",
            "v_cos_f32_e64 v13, -|v13| clamp",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("trig_ps", trig_shader)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

    def test_opcodes_recognized_but_translator_rejected(self):
        """Shader with valid RDNA 2 opcodes whose structure violates pixel_aot constraints."""
        valid_rdna2_but_rejected = [
            "s_mov_b64 s[50:51], exec",
            "s_branch 0",  # s_branch is a valid RDNA 2 opcode recognized by scanner, but rejected by translator
            "s_mov_b32 m0, s24",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("branched_ps", valid_rdna2_but_rejected)
        # All opcodes are valid RDNA 2 instructions
        self.assertTrue(res["opcodes_recognized"])
        # But translator rejects the non-straight-line structure
        self.assertFalse(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "opcode_recognized_translator_rejected")
        self.assertFalse(res["is_fully_supported"])
        self.assertIsNotNone(res["lower_error"])

    def test_unsupported_opcode_shader(self):
        """Shader containing opcodes unknown or unsupported by the ISA scanner."""
        unknown_opcode_shader = [
            "s_mov_b64 s[50:51], exec",
            "v_imaginary_opcode_xyz v0, v1, v2",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("unknown_ps", unknown_opcode_shader)
        self.assertFalse(res["opcodes_recognized"])
        self.assertFalse(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "unsupported_opcodes")
        self.assertIn("v_imaginary_opcode_xyz", res["unsupported_opcodes"])

    def test_vopc_classification(self):
        """Vector compare opcodes with type suffixes must classify as VOPC, not VALU."""
        for opcode in ("v_cmp_lt_f32", "v_cmp_eq_u32", "v_cmp_gt_i32", "v_cmp_le_f16"):
            self.assertEqual(classify_isa_category(opcode), "VOPC (Vector Compare)", opcode)
        for pred in ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"):
            self.assertEqual(classify_isa_category(f"v_cmp_{pred}_f32"), "VOPC (Vector Compare)")
        self.assertEqual(classify_isa_category("v_add_f32"), "VOP / VALU (Vector ALU - VOP1/VOP2/VOP3)")

    def test_vopc_e32_destination_must_be_vcc(self):
        """An explicit _e32 compare may only write vcc; suffix-less and _e64 may name an SGPR."""
        families = [f"v_cmp_{op}_{kind}" for kind in ("f16", "i32", "u32", "i16", "u16")
                    for op in ("lt", "eq")] + ["v_cmp_class_f32", "v_cmp_class_f16"]
        f32_predicates = [f"v_cmp_{op}_f32" for op in ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")]
        for name in set(families) | set(f32_predicates):
            for dst in ("s52", "s[52:53]"):
                self.assertFalse(is_instruction_supported(f"{name}_e32 {dst}, s44, v18"), (name, dst))
                self.assertTrue(is_instruction_supported(f"{name} {dst}, s44, v18"), (name, dst))
                self.assertTrue(is_instruction_supported(f"{name}_e64 {dst}, s44, v18"), (name, dst))
            for dst in ("vcc", "vcc_lo"):
                self.assertTrue(is_instruction_supported(f"{name}_e32 {dst}, s44, v18"), (name, dst))

    def test_f32_vector_compares_recognition(self):
        """All 16 binary32 vector compare predicates must be recognized in base opcodes and instruction forms."""
        for op in ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"):
            base = f"v_cmp_{op}_f32"
            self.assertTrue(is_opcode_recognized(base), f"Base opcode not recognized: {base}")
            self.assertTrue(is_opcode_recognized(f"{base}_e32"), f"e32 not recognized: {base}_e32")
            self.assertTrue(is_opcode_recognized(f"{base}_e64"), f"e64 not recognized: {base}_e64")
            self.assertTrue(is_instruction_supported(f"{base}_e32 vcc_lo, s44, v18"))
            self.assertTrue(is_instruction_supported(f"{base}_e64 s52, -s44, |v18|"))
            self.assertTrue(is_instruction_supported(f"{base}_sdwa vcc_lo, s44, v18 src0_sel:DWORD src1_sel:DWORD"))

    def test_binary_scan_recognizes_vopc_sdwa(self):
        """Issue #938: binary shaders with VOPC/VOPCX SDWA reach the scanner as decoded opcodes, not vopc_sdwa."""
        supported = bytes.fromhex(
            "f904027c01861606"  # v_cmp_lt_f32_sdwa s6, -v1, v2 src0_sel:DWORD src1_sel:DWORD
            "000081bf"          # s_endpgm
        )
        unlowered = bytes.fromhex(
            "f904887d01000506"  # v_cmp_gt_u32_sdwa vcc_lo, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
            "f904247c01001606"  # v_cmpx_eq_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
            "000081bf"          # s_endpgm
        )
        f64_fallback = bytes.fromhex(
            "f904427c01000606"  # v_cmp_lt_f64_e32 with src0=249 (unsupported SDWA control)
            "000081bf"          # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "ps_vopc_sdwa_supported.bin").write_bytes(supported)
            (tmp_path / "ps_vopc_sdwa_unlowered.bin").write_bytes(unlowered)
            (tmp_path / "ps_vopc_sdwa_f64_fallback.bin").write_bytes(f64_fallback)
            results = {r["shader_name"]: r for r in scan_directory_or_files([tmp_path])}

        res_sup = results["ps_vopc_sdwa_supported.bin"]
        self.assertEqual((res_sup["total_instructions"], res_sup["supported_instructions"], res_sup["unsupported_count"]),
                         (2, 2, 0), res_sup.get("unsupported_details"))
        self.assertTrue(res_sup["opcodes_recognized"])
        self.assertEqual(res_sup["unsupported_opcodes"], {})

        res_unl = results["ps_vopc_sdwa_unlowered.bin"]
        self.assertTrue(res_unl["opcodes_recognized"])
        self.assertEqual(res_unl["unsupported_opcodes"], {})
        self.assertFalse(res_unl["translator_accepted"])
        self.assertEqual(res_unl["compatibility_category"], "opcode_recognized_translator_rejected")
        self.assertEqual(res_unl["unsupported_count"], 2)
        self.assertEqual(
            [d["raw"] for d in res_unl["unsupported_details"]],
            [
                "v_cmp_gt_u32_sdwa vcc_lo, v1, v2 src0_sel:WORD_1 src1_sel:DWORD",
                "v_cmpx_eq_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD",
            ],
        )

        res_bad = results["ps_vopc_sdwa_f64_fallback.bin"]
        self.assertFalse(res_bad["translator_accepted"])
        self.assertEqual(res_bad["unsupported_count"], 1)
        self.assertEqual(
            [d["raw"] for d in res_bad["unsupported_details"]],
            ["v_cmp_lt_f64_e32 <unsupported SDWA>"],
        )

    def test_vop3p_classification(self):
        """Packed and mixed-precision opcodes must classify as VOP3P."""
        for opcode in (
            "v_pk_add_f16", "v_pk_add_u16", "v_pk_sub_i16", "v_pk_fma_f16",
            "v_dot2_f32_f16", "v_dot4_i32_i8", "v_dot8_u32_u4",
            "v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16",
        ):
            self.assertEqual(
                classify_isa_category(opcode),
                "VOP3P (Vector Packed / Mixed-Precision)",
                opcode,
            )

    def test_sopk_scalar_is_arithmetic_supported(self):
        """The scanner accepts s_addk_i32 and s_mulk_i32 with a scalar destination and a hex or negative immediate, and rejects a vector destination."""
        for line in ("s_addk_i32 s50, 0xfffe", "s_mulk_i32 s50, 0x3",
                     "s_addk_i32 s50, -2"):
            self.assertTrue(is_instruction_supported(line), line)
        self.assertFalse(is_instruction_supported("s_mulk_i32 v50, 0x3"))

    def test_literal_mad_forms_take_four_operands(self):
        """v_madak_f32 and v_madmk_f32 are supported with four operands; e64 and a missing operand are not."""
        for line in ("v_madak_f32 v5, v1, v2, 0x11213141", "v_madmk_f32 v5, v1, 0x11213141, v3"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_madak_f32_e64 v5, v1, v2, 0x11213141", "v_madmk_f32 v5, v1, 0x11213141",
                     "v_madak_f32_dpp v5, v1, v2, 0x11213141 quad_perm:[0,1,2,3]",
                     "v_madak_f32 v5, v1, v2, v3", "v_madmk_f32 v5, v1, v2, v3"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_non_packed_int16_vop3_forms(self):
        """The 12 non-packed 16-bit integer VOP3 forms take two integer sources; modifiers and wide operands are not supported."""
        names = ("add_nc_u16", "add_nc_i16", "sub_nc_u16", "sub_nc_i16", "min_u16", "max_u16", "min_i16",
                 "max_i16", "mul_lo_u16", "lshlrev_b16", "lshrrev_b16", "ashrrev_i16")
        self.assertEqual(set(names), set(VOP3_INT16))
        for name in names:
            line = f"v_{name} v5, v1, v2"
            self.assertTrue(is_instruction_supported(line), line)
            self.assertTrue(is_opcode_recognized(f"v_{name}"), name)
        for line in ("v_add_nc_u16_e64 v5, v1, v2", "v_sub_nc_i16 v5, v1, 0x1234",
                     "v_add_nc_i16 v5, v1, 4", "v_add_nc_u16 v5, v1, -16", "v_add_nc_u16 v5, v1, 64",
                     "v_add_nc_u16 v5, v1, 0x00001234", "v_add_nc_u16 v255, s105, v2"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_add_nc_u16 v5, v1, v2 op_sel:[0,0,1]", "v_add_nc_u16 v5, v1, v2 clamp",
                     "v_add_nc_u16 v5, v1, 1.0", "v_add_nc_u16 v5, v1, 0x10000", "v_add_nc_u16 v5, v1, 65",
                     "v_add_nc_u16 v5, v1, -17", "v_add_nc_u16_dpp v5, v1, v2 quad_perm:[0,1,2,3]",
                     "v_add_nc_u16_sdwa v5, v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0",
                     "v_add_nc_u16 v5, -v1, v2", "v_add_nc_u16 v256, v1, v2", "v_add_nc_u16 v5, v256, v2",
                     "v_add_nc_u16 v5, s106, v2"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_non_packed_int16_ternary_forms(self):
        """The 8 three-operand 16-bit integer VOP3 forms take three integer sources; clamp, op_sel and wide operands are not supported."""
        names = ("mad_u16", "mad_i16", "min3_i16", "min3_u16", "max3_i16", "max3_u16", "med3_i16", "med3_u16")
        self.assertEqual(set(names), set(VOP3_INT16_TERNARY))
        for name in names:
            line = f"v_{name} v5, v1, v2, v3"
            self.assertTrue(is_instruction_supported(line), line)
            self.assertTrue(is_opcode_recognized(f"v_{name}"), name)
            for bad in (f"v_{name} v5, v1, v2", f"v_{name} v5, v1, v2, v3 clamp",
                        f"v_{name} v5, v1, v2, v3 op_sel:[0,0,0,1]", f"v_{name} v5, -v1, v2, v3",
                        f"v_{name} v5, v1, 1.0, v3", f"v_{name} v5, v1, v2, 0x10000",
                        f"v_{name}_dpp v5, v1, v2, v3 quad_perm:[0,1,2,3]", f"v_{name} v256, v1, v2, v3",
                        f"v_{name} v5, v1, v2, s106"):
                self.assertFalse(is_instruction_supported(bad), bad)
        for line in ("v_mad_u16_e64 v5, v1, v2, v3", "v_max3_i16 v5, v1, v2, 4", "v_med3_u16 v5, v1, -16, 64",
                     "v_mad_i16 v5, v1, 0x0012, s105"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_mad_u32_u16 v5, v1, v2, v3", "v_med3_f16 v5, v1, v2, v3"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_float_to_word_conversions_e32_and_e64(self):
        """v_cvt_u32_f32 and v_cvt_i32_f32 accept plain e32 and e64 with optional source abs/neg;
        v_cvt_flr_i32_f32 and v_cvt_rpi_i32_f32 stay out of generic scanner support until arbitrary runtime
        exceptional inputs are backed by a native RDNA reference."""
        for line in ("v_cvt_u32_f32_e32 v12, v13", "v_cvt_i32_f32_e32 v12, s44",
                     "v_cvt_u32_f32_e32 v12, 0x40400000", "v_cvt_u32_f32_e64 v12, v13",
                     "v_cvt_i32_f32_e64 v12, s44", "v_cvt_u32_f32_e64 v12, |s44|",
                     "v_cvt_i32_f32_e64 v12, -s44", "v_cvt_i32_f32_e64 v12, -|v13|",
                     "v_cvt_u32_f32_e64 v12, abs(v13)", "v_cvt_i32_f32_e64 v12, neg(0x40400000)",
                     "v_cvt_u32_f32_e64 v12, |0xc0300000|"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_cvt_u32_f32_e64 v12, s44 clamp", "v_cvt_i32_f32_e64 v12, s44 omod:2",
                     "v_cvt_i32_f32_e32 v12, 1", "v_cvt_u32_f32_e32 v12, 0.5",
                     "v_cvt_u32_f32_e64 v12, 0.5", "v_cvt_i32_f32_e64 v12, -1",
                     "v_cvt_i32_f32_e64 v12, -0x40400000", "v_cvt_u32_f32_e64 v12, |-s44|",
                     "v_cvt_u32_f32_e32  v12, v13", "v_cvt_u32_f32_e64  v12, v13",
                     "v_cvt_flr_i32_f32_e32 v12, v13", "v_cvt_rpi_i32_f32_e64 v12, -v13"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_disasm_cvt_flr_and_rpi_e32_and_e64(self):
        """tools/disasm_rdna2.py decodes v_cvt_rpi_i32_f32 and v_cvt_flr_i32_f32 in VOP1 and VOP3."""
        cases = (
            (bytes([0x00, 0x19, 0x02, 0x7e]), "v_cvt_rpi_i32_f32_e32", "v_cvt_rpi_i32_f32_e32 v1, v0"),
            (bytes([0x00, 0x1b, 0x02, 0x7e]), "v_cvt_flr_i32_f32_e32", "v_cvt_flr_i32_f32_e32 v1, v0"),
            (bytes([0x01, 0x00, 0x8d, 0xd5, 0x00, 0x01, 0x01, 0x02]), "v_cvt_flr_i32_f32_e64", "v_cvt_flr_i32_f32_e64 v1, v0"),
            (bytes([0x01, 0x01, 0x8c, 0xd5, 0x00, 0x01, 0x01, 0x22]), "v_cvt_rpi_i32_f32_e64", "v_cvt_rpi_i32_f32_e64 v1, -|v0|"),
        )
        for raw, expected_op, expected_asm in cases:
            length, op_name, asm, _ = decode_instruction(raw, 0)
            self.assertEqual(length, len(raw))
            self.assertEqual(op_name, expected_op)
            self.assertEqual(asm, expected_asm)

    def test_inline_slash_comments_and_blank_lines(self):
        """Inline // comments must be stripped and blank/comment lines must not inflate counts."""
        self.assertTrue(is_instruction_supported("s_endpgm // shader exit"))
        res = analyze_shader_instructions(
            "commented_ps",
            [
                "",
                "; header comment",
                "// llvm-style comment",
                "s_endpgm // shader exit",
                "v_imaginary_opcode_xyz v0, v1 // unsupported",
            ],
        )
        self.assertEqual(res["total_instructions"], 2)
        self.assertEqual(res["supported_instructions"], 1)
        self.assertEqual(res["unsupported_count"], 1)
        self.assertEqual(res["support_percentage"], 50.0)
        self.assertEqual(res["unsupported_details"][0]["raw"], "v_imaginary_opcode_xyz v0, v1")

    def test_cli_scanner(self):
        """Invoke scanner via its actual CLI entry point on synthetic assembly file."""
        canonical_shader = "\n".join([
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shader_file = tmp_path / "canonical.s"
            shader_file.write_text(".text\n.globl pixel\n.type pixel,@function\npixel:\n"
                                   + canonical_shader + "\n.size pixel, .-pixel\n", encoding="utf-8")
            out_dir = tmp_path / "scan_reports"
            script = Path(__file__).resolve().parents[1] / "tools" / "scan_unsupported_shaders.py"

            # The scanner prints block characters when its stdout encoding allows them, so both
            # sides use UTF-8 whatever the caller's PYTHONIOENCODING and code page (#844).
            def scan():
                return subprocess.run(
                    [sys.executable, "-B", str(script), str(shader_file), "--output-dir", str(out_dir)],
                    capture_output=True,
                    encoding="utf-8",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    check=True,
                )

            res = scan()
            self.assertEqual(res.returncode, 0)
            self.assertIn("\N{FULL BLOCK}", res.stdout)  # the progress bar's UTF-8 form, as in #844
            self.assertIn("SCAN RESULTS SUMMARY", res.stdout)
            self.assertIn("Full Shaders Accepted by Translator:       1", res.stdout)
            self.assertTrue((out_dir / "shader_scan_report.json").is_file())
            self.assertTrue((out_dir / "shader_scan_report.md").is_file())
            with shader_file.open("a", encoding="utf-8") as stream:
                stream.write(".long 0\n")
            result = scan()
            self.assertIn("Full Shaders Accepted by Translator:       0", result.stdout)

    def test_binary_scan_recognizes_vop2_carry_and_image_atomics(self):
        """Issue #938 step 3: binary scan accepts decoded VOP2 carry ops and keeps image_atomic_* recognized-only."""
        carry_bin = bytes.fromhex(
            "01050a50"  # v_add_co_ci_u32_e32 v5, vcc_lo, v1, v2, vcc_lo
            "01050a52"  # v_sub_co_ci_u32_e32 v5, vcc_lo, v1, v2, vcc_lo
            "01050a54"  # v_subrev_co_ci_u32_e32 v5, vcc_lo, v1, v2, vcc_lo
            "000081bf"  # s_endpgm
        )
        atomic_bin = bytes.fromhex(
            "00215cf002010000"  # image_atomic_umax v1, v2, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D glc
            "000081bf"          # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "ps_carry.bin").write_bytes(carry_bin)
            (tmp_path / "ps_atomic.bin").write_bytes(atomic_bin)
            results = {r["shader_name"]: r for r in scan_directory_or_files([tmp_path])}
        carry = results["ps_carry.bin"]
        self.assertEqual(
            (carry["total_instructions"], carry["supported_instructions"], carry["unsupported_count"]),
            (4, 4, 0),
            carry.get("unsupported_details"),
        )
        self.assertTrue(carry["opcodes_recognized"])
        self.assertEqual(carry["unsupported_opcodes"], {})

        atomic = results["ps_atomic.bin"]
        self.assertTrue(atomic["opcodes_recognized"])
        self.assertEqual(atomic["unsupported_opcodes"], {})
        self.assertFalse(atomic["translator_accepted"])
        self.assertEqual(atomic["compatibility_category"], "opcode_recognized_translator_rejected")
        self.assertEqual(atomic["unsupported_count"], 1)
        self.assertEqual(
            [d["raw"] for d in atomic["unsupported_details"]],
            ["image_atomic_umax v1, v2, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D glc"],
        )

        # Negative control: the old unnamed VOP2 spellings failed both recognition and support.
        for old_op in ("v_vop2_28", "v_vop2_29", "v_vop2_2a"):
            self.assertFalse(is_opcode_recognized(old_op), old_op)
            self.assertFalse(is_instruction_supported(f"{old_op} v5, v1, v2"), old_op)

    def test_report_generation(self):
        canonical_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        rejected_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_branch 0",
            "s_endpgm",
        ]
        results = [
            analyze_shader_instructions("accepted_ps", canonical_shader),
            analyze_shader_instructions("rejected_ps", rejected_shader),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_file, md_file = generate_reports(results, tmp_path)
            self.assertTrue(json_file.is_file())
            self.assertTrue(md_file.is_file())

            data = json.loads(json_file.read_text(encoding="utf-8"))
            summary = data["summary"]
            self.assertEqual(summary["total_shaders_scanned"], 2)
            self.assertEqual(summary["translator_accepted"], 1)
            self.assertEqual(summary["opcodes_recognized_translator_rejected"], 1)

    def test_detect_shader_stage(self):
        """Verify pipeline stage detection from file names and relative paths."""
        self.assertEqual(detect_shader_stage("shaders/pixel/ps_draw.bin"), "pixel")
        self.assertEqual(detect_shader_stage("cs_dispatch.bin"), "compute")
        self.assertEqual(detect_shader_stage("shaders/vertex/vs_main.s"), "vertex")
        self.assertEqual(detect_shader_stage("unknown_kernel.bin"), "other")

    def test_binary_scan_recognizes_decoded_interp_f16(self):
        """Issue #938 step 2: a scanned binary's binary16 interpolation forms reach the scanner by name.

        The bytes are llvm-mc 15.0.7 (gfx1013) output for the commented lines. With the old `v_vop3_342`
        spelling the scan reported unrecognized opcodes; now the opcodes are recognized and attributed to VINTRP,
        and the program is a translator rejection, since pixel_aot does not lower these forms.
        """
        code = bytes.fromhex(
            "010042d7c5050200"  # v_interp_p1ll_f16 v1, v2, attr5.w high
            "010043d741040e94"  # v_interp_p1lv_f16 v1, v2, attr1.y, -v3 mul:4
            "01825ad7ff05fe07"  # v_interp_p2_f16 v1, |v2|, attr63.w, v255 high clamp
            "000081bf"          # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ps_interp_f16.bin"
            path.write_bytes(code)
            result, = scan_directory_or_files([path])
        self.assertEqual(result["total_instructions"], 4)
        self.assertTrue(result["opcodes_recognized"], result.get("unsupported_details"))
        self.assertEqual(result["compatibility_category"], "opcode_recognized_translator_rejected")
        self.assertFalse(any(name.startswith("v_vop3_") for name in result["opcode_distribution"]))
        self.assertEqual({name for name in result["opcode_distribution"] if name.startswith("v_interp")},
                         {"v_interp_p1ll_f16", "v_interp_p1lv_f16", "v_interp_p2_f16"})
        self.assertEqual({classify_isa_category(name) for name in ("v_interp_p1ll_f16", "v_interp_p2_f16")},
                         {"VINTRP (Parameter Interpolation)"})

    def test_binary_scan_reports_decoded_hwreg_writes_as_supported(self):
        """Issue #938: a scanned binary's MODE rounding and FLAT_SCRATCH writes reach the scanner in LLVM's spelling.

        The bytes are llvm-mc 15.0.7 (gfx1013) output for the commented source lines. With the raw spelling
        (`s_setreg_imm32_b32 0x881, 0x3`) the first program reported four unsupported instructions.
        """
        accepted = bytes.fromhex(
            "810880ba03000000"  # s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3
            "14f880ba00000000"  # s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0
            "15f880ba00040000"  # s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_HI), 0x400
            "810880ba00000000"  # s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0
            "000081bf"          # s_endpgm
        )
        # Control: round toward +infinity (1) is a directed mode the translator refuses.
        directed = bytes.fromhex(
            "810880ba01000000"  # s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1
            "000081bf"          # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "ps_hwreg.bin").write_bytes(accepted)
            (tmp_path / "ps_hwreg_directed.bin").write_bytes(directed)
            results = {r["shader_name"]: r for r in scan_directory_or_files([tmp_path])}
        res = results["ps_hwreg.bin"]
        self.assertEqual((res["total_instructions"], res["supported_instructions"], res["unsupported_count"]),
                         (5, 5, 0), res.get("unsupported_details"))
        self.assertTrue(res["opcodes_recognized"])
        self.assertEqual(res["unsupported_opcodes"], {})
        control = results["ps_hwreg_directed.bin"]
        self.assertTrue(control["opcodes_recognized"])
        self.assertEqual(control["unsupported_opcodes"], {})
        self.assertFalse(control["translator_accepted"])
        self.assertEqual(control["compatibility_category"], "opcode_recognized_translator_rejected")
        self.assertEqual(control["unsupported_count"], 1)
        self.assertEqual([d["raw"] for d in control["unsupported_details"]],
                         ["s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1"])

    def test_binary_scan_hwreg_recognition_stays_separate_from_translation_support(self):
        """Issue #938 follow-up: hwreg SOPK opcodes are recognized ISA opcodes while s_getreg/s_setreg stay unlowered."""
        for op in ("s_getreg_b32", "s_setreg_b32", "s_setreg_imm32_b32"):
            self.assertTrue(is_opcode_recognized(op), op)

        # Accepted end-to-end pixel shader with supported s_setreg_imm32_b32 forms: now classified as accepted
        # rather than unsupported_opcodes.
        base_pixel_tail = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        accepted_hwreg = analyze_shader_instructions(
            "ps_hwreg_accepted",
            [
                "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0",
                "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_HI), 0x400",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0",
                *base_pixel_tail,
            ],
        )
        self.assertTrue(accepted_hwreg["opcodes_recognized"])
        self.assertEqual(accepted_hwreg["unsupported_opcodes"], {})
        self.assertTrue(accepted_hwreg["translator_accepted"], accepted_hwreg.get("lower_error"))
        self.assertEqual(accepted_hwreg["compatibility_category"], "accepted")
        self.assertTrue(accepted_hwreg["is_fully_supported"])

        # Recognition remains separate from translation support for s_getreg_b32 and s_setreg_b32:
        # both in a binary scan and when embedded in an otherwise accepted pixel shader.
        for label, line in (
            ("getreg", "s_getreg_b32 s4, hwreg(HW_REG_MODE)"),
            ("setreg", "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s4"),
        ):
            with self.subTest(form=label):
                self.assertFalse(is_instruction_supported(line), line)
                rejected = analyze_shader_instructions(f"ps_{label}", [line, *base_pixel_tail])
                self.assertTrue(rejected["opcodes_recognized"])
                self.assertEqual(rejected["unsupported_opcodes"], {})
                self.assertFalse(rejected["translator_accepted"])
                self.assertFalse(rejected["is_fully_supported"])
                self.assertEqual(rejected["compatibility_category"], "opcode_recognized_translator_rejected")
                self.assertEqual(rejected["unsupported_count"], 1)
                self.assertEqual([d["raw"] for d in rejected["unsupported_details"]], [line])

        hwreg_sgpr = bytes.fromhex(
            "01f804b9"  # s_getreg_b32 s4, hwreg(HW_REG_MODE)
            "810884b9"  # s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s4
            "000081bf"  # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "ps_hwreg_sgpr.bin").write_bytes(hwreg_sgpr)
            results = {r["shader_name"]: r for r in scan_directory_or_files([tmp_path])}
        res = results["ps_hwreg_sgpr.bin"]
        self.assertTrue(res["opcodes_recognized"])
        self.assertEqual(res["unsupported_opcodes"], {})
        self.assertEqual(res["compatibility_category"], "opcode_recognized_translator_rejected")
        self.assertFalse(res["translator_accepted"])
        self.assertEqual(res["unsupported_count"], 2)
        self.assertEqual(
            [d["raw"] for d in res["unsupported_details"]],
            [
                "s_getreg_b32 s4, hwreg(HW_REG_MODE)",
                "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s4",
            ],
        )

        # Negative control: an unknown opcode is still flagged as unrecognized.
        unknown = analyze_shader_instructions("ps_unknown", ["s_not_a_real_hwreg_op s0", "s_endpgm"])
        self.assertFalse(unknown["opcodes_recognized"])
        self.assertEqual(unknown["unsupported_opcodes"], {"s_not_a_real_hwreg_op": 1})
        self.assertEqual(unknown["compatibility_category"], "unsupported_opcodes")

    def test_binary_shader_scanning(self):
        """Verify scanning raw .bin files directly with disasm_rdna2 without LLVM."""
        raw_code = bytes.fromhex(
            "000080bf"  # s_nop 0
            "7f008cbf"  # s_waitcnt
            "000081bf"  # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_file = tmp_path / "ps_test.bin"
            bin_file.write_bytes(raw_code)
            results = scan_directory_or_files([bin_file])
            self.assertEqual(len(results), 1)
            res = results[0]
            self.assertEqual(res["shader_name"], "ps_test.bin")
            self.assertEqual(res["stage"], "pixel")
            self.assertEqual(res["total_instructions"], 3)
            self.assertTrue(res["opcodes_recognized"])

    def test_disassemble_binary_shader_error(self):
        """Verify disassemble_binary_shader raises on invalid or empty input."""
        with self.assertRaises(ValueError):
            disassemble_binary_shader(b"")

    def test_mixed_binary_shader_scanning(self):
        """Verify that failed/invalid binary inputs are retained in results and denominator."""
        valid_code = bytes.fromhex(
            "000080bf"  # s_nop 0
            "000081bf"  # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            good_file = tmp_path / "ps_good.bin"
            good_file.write_bytes(valid_code)
            bad_file = tmp_path / "ps_bad.bin"
            bad_file.write_bytes(b"")  # Empty binary triggers decoder failure

            results = scan_directory_or_files([tmp_path])
            # Denominator must account for BOTH inputs
            self.assertEqual(len(results), 2)

            by_name = {r["shader_name"]: r for r in results}
            good_res = by_name.get("ps_good.bin")
            bad_res = by_name.get("ps_bad.bin")

            self.assertIsNotNone(good_res)
            self.assertEqual(good_res["total_instructions"], 2)
            self.assertEqual(good_res["supported_instructions"], 2)
            self.assertTrue(good_res["opcodes_recognized"])

            self.assertIsNotNone(bad_res)
            self.assertEqual(bad_res["total_instructions"], 0)
            self.assertFalse(bad_res["is_fully_supported"])
            self.assertFalse(bad_res["translator_accepted"])
            self.assertEqual(bad_res["lower_status"], "decode_failed")
            self.assertIn("empty", bad_res["lower_error"].lower())

            # Verify that dashboard accounts for both inputs and clearly labels instruction vs file metrics
            out_dir = tmp_path / "out"
            json_file, md_file = generate_progress_dashboard(results, out_dir)
            data = json.loads(json_file.read_text(encoding="utf-8"))

            # File-level denominator tested separately: two inputs, one decode failure
            self.assertEqual(data["overall"]["total_shaders"], 2)
            self.assertEqual(data["overall"]["completely_unsupported"], 1)
            self.assertEqual(data["overall"]["fully_accepted"], 0)

            # Instruction metric: every decoded instruction in valid_code is supported,
            # so instruction coverage legitimately reaches 100% while file denominator retains failure.
            self.assertEqual(data["overall"]["total_instructions"], 2)
            self.assertEqual(data["overall"]["supported_instructions"], 2)
            self.assertEqual(data["overall"]["instruction_coverage_percentage"], 100.0)
            self.assertEqual(data["overall"]["coverage_percentage"], 100.0)

            md_text = md_file.read_text(encoding="utf-8")
            self.assertIn("| **Total Shaders Analyzed** | 2 | 100.0% |", md_text)
            self.assertIn("| **Unsupported / Blocked** | 1 | 50.0% |", md_text)
            self.assertIn("| **Total Decoded Instructions** | 2 | 100.0% |", md_text)
            self.assertIn("Instruction Coverage:", md_text)

    def test_dashboard_report_generation(self):
        """Verify generation of PROGRESS.md and progress.json."""
        canonical_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        results = [analyze_shader_instructions("ps_canonical", canonical_shader)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_file, md_file = generate_progress_dashboard(results, tmp_path)
            self.assertTrue(json_file.is_file())
            self.assertTrue(md_file.is_file())
            md_content = md_file.read_text(encoding="utf-8")
            self.assertIn("Instruction Coverage:", md_content)
            self.assertIn("Pipeline Stage Breakdown", md_content)
            json_content = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertIn("overall", json_content)
            self.assertIn("stages", json_content)
            self.assertIn("missing_opcodes_ranked", json_content)

    def test_supported_shader_instructions(self):
        """Verify scanner patterns recognize subrev, mov_b32, waitcnt_expcnt as supported."""
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt null, 0x0"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt null, 0x3"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt null, 0"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt null, 7"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt s0, 0x0"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt 0x3"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt 0"))
        self.assertTrue(is_instruction_supported("s_waitcnt expcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt expcnt(7)"))
        self.assertTrue(is_instruction_supported("s_waitcnt lgkmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(63)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(0) lgkmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt lgkmcnt(0) vmcnt(0)"))
        self.assertTrue(is_instruction_supported("v_nop"))
        self.assertFalse(is_instruction_supported("v_nop 0"))
        self.assertFalse(is_instruction_supported("v_nop v0"))
        self.assertTrue(is_instruction_supported("v_nop_e64"))
        self.assertFalse(is_instruction_supported("v_nop_e64 0"))
        # The VOP3 interpolation spelling: only the operand shape of the e32 form.
        for phase, source in ((1, "v0"), (2, "v1")):
            line = f"v_interp_p{phase}_f32_e64 v8, {source}, attr0.x"
            self.assertTrue(is_instruction_supported(line), line)
            self.assertTrue(is_instruction_supported(line.replace("_e64", "_e32")), line)
            for wrong in (line.replace(source, f"-{source}"), line.replace(source, f"|{source}|"), line + " clamp",
                          line + " mul:2", line.replace("attr0", "attr1"), line.replace(source, "s0")):
                self.assertFalse(is_instruction_supported(wrong), wrong)
        self.assertTrue(is_instruction_supported("v_subrev_f32_e32 v1, v2, v3"))
        self.assertTrue(is_instruction_supported("v_subrev_f32_e64 v1, v2, v3 clamp"))
        # v_mac_legacy_f32 has an e32 and an e64 form (the latter with clamp) and no DPP form in LLVM 15. The
        # accumulator is the destination, so only two sources are written. v_fmac_legacy_f32 does not exist on gfx1013.
        for line in ("v_mac_legacy_f32_e32 v1, v2, v3", "v_mac_legacy_f32_e64 v1, v2, v3",
                     "v_mac_legacy_f32_e64 v1, -s2, |v3| clamp"):
            self.assertTrue(is_instruction_supported(line), line)
        self.assertTrue(is_opcode_recognized("v_mac_legacy_f32_e64"))
        for line in ("v_mac_legacy_f32_dpp v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                     "v_mac_legacy_f32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
                     "v_fmac_legacy_f32 v1, v2, v3", "v_fmac_legacy_f32_e32 v1, v2, v3"):
            self.assertFalse(is_instruction_supported(line), line)
        # The 32-bit no-carry family. The INTEGER_SOURCE class is coarser than the translator, so this
        # pins the spellings and the refused modifiers, not the operand ranges.
        for line in (
            "v_add_nc_u32_e32 v1, v2, v3", "v_sub_nc_u32 v1, s2, 64", "v_subrev_nc_u32_e32 v1, v2, v3",
            "v_add_nc_u32_e64 v1, s2, v3", "v_sub_nc_u32_e64 v1, s2, 64", "v_subrev_nc_u32_e64 v1, v2, s3",
            "v_add_nc_i32 v1, s2, v3", "v_sub_nc_i32 v1, 64, v3",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_add_nc_u32_e64 v1, s2, v3 clamp", "v_subrev_nc_u32_e64 v1, s2, v3 clamp",
            "v_add_nc_i32 v1, s2, v3 clamp", "v_sub_nc_i32 v1, s2, v3 clamp",
            "v_add_nc_i32_e64 v1, s2, v3", "v_sub_nc_i32_e64 v1, s2, v3",
            "v_subrev_nc_i32 v1, s2, v3",
            "v_subrev_nc_u32_dpp v1, v2, v3 row_shl:1 row_mask:0xf bank_mask:0xf",
            "v_subrev_nc_u32_sdwa v1, v2, v3 src0_sel:DWORD src1_sel:DWORD",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        # SDWA forms of the 32-bit integer operations: registers, sext(), all fields, no UNUSED_SEXT.
        sdwa = "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"
        for line in (
            f"v_and_b32_sdwa v1, v2, v3 {sdwa}", f"v_xnor_b32_sdwa v1, s2, sext(v3) {sdwa}",
            "v_subrev_nc_u32_sdwa v1, sext(s2), v3 dst_sel:BYTE_3 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:BYTE_0",
            "v_not_b32_sdwa v1, sext(v2) dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_2",
            f"v_ashrrev_i32_sdwa v1, v2, v3 {sdwa}", f"v_max_i32_sdwa v1, v2, v3 {sdwa}", f"v_min_u32_sdwa v1, v2, v3 {sdwa}",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            f"v_and_b32_sdwa v1, v2, v3 {sdwa}".replace("UNUSED_PAD", "UNUSED_SEXT"),
            f"v_and_b32_sdwa v1, v2, v3 clamp {sdwa}", f"v_and_b32_sdwa v1, 64, v3 {sdwa}",
            f"v_and_b32_sdwa v1, -v2, v3 {sdwa}", f"v_and_b32_sdwa v1, v2 {sdwa}",
            "v_and_b32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD",
            f"v_not_b32_sdwa v1, v2 {sdwa}", f"v_not_b32_sdwa v1, v2, v3 {sdwa}",
            f"v_bfi_b32_sdwa v1, v2, v3 {sdwa}", f"v_add_nc_i32_sdwa v1, v2, v3 {sdwa}",
            f"v_mul_u32_u24_sdwa v1, v2, v3 {sdwa}",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_add_nc_i32", "v_sub_nc_i32", "v_subrev_nc_u32_e64"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        self.assertFalse(is_opcode_recognized("v_subrev_nc_i32"))
        # SDWA binary32 arithmetic: whole-dword sources only; a byte or word source, UNUSED_SEXT and
        # operations outside pixel_aot.SDWA_ALU stay unsupported.
        for op in ("exp", "log", "rcp", "rcp_iflag", "rsq", "sqrt", "sin", "cos", "rndne", "frexp_mant", "floor"):
            line = f"v_{op}_f32_sdwa v1, -|s2| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD"
            self.assertTrue(is_instruction_supported(line), line)
            self.assertTrue(is_instruction_supported(line.replace("dst_sel:DWORD", "clamp dst_sel:BYTE_2")), line)
            self.assertFalse(is_instruction_supported(line.replace("src0_sel:DWORD", "src0_sel:WORD_1")), line)
            self.assertFalse(is_instruction_supported(line.replace("src0_sel:DWORD", "src0_sel:BYTE_0")), line)
            self.assertFalse(is_instruction_supported(line.replace("UNUSED_PAD", "UNUSED_SEXT")), line)
        self.assertTrue(is_instruction_supported(
            "v_add_f32_sdwa v1, s2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"))
        self.assertFalse(is_instruction_supported(
            "v_add_f32_sdwa v1, s2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0"))
        self.assertFalse(is_instruction_supported(
            "v_mad_f32_sdwa v1, s2, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"))
        self.assertTrue(is_opcode_recognized("v_mov_b32_sdwa"))
        self.assertTrue(is_opcode_recognized("v_cvt_f32_f16_sdwa"))
        self.assertTrue(is_opcode_recognized("v_pk_add_f16"))
        self.assertTrue(is_opcode_recognized("v_dot4_i32_i8"))
        self.assertTrue(is_instruction_supported("v_cvt_f32_u32_e32 v1, v2"))
        self.assertTrue(is_instruction_supported("v_cvt_f32_u32_e64 v1, v2 clamp"))
        self.assertTrue(is_instruction_supported("v_cvt_f32_i32_e32 v1, v2"))
        self.assertTrue(is_instruction_supported("v_cvt_f32_i32_e64 v1, v2 clamp"))
        self.assertTrue(is_instruction_supported("s_nand_b32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_nor_b32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_xnor_b32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_lshl1_add_u32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_lshl2_add_u32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_lshl3_add_u32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_lshl4_add_u32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_bfm_b32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_bfm_b32 s1, 4, 8"))
        for line in (
            "s_bfe_u64 s[0:1], s[2:3], s4",
            "s_bfe_u64 s[0:1], s[2:3], 0x400000",
            "s_bfe_u64 s[0:1], 0x80000000, 32",
            "s_bfe_i64 s[0:1], s[2:3], s4",
            "s_bfe_i64 s[0:1], s[2:3], 0x20001f",
            "s_bfm_b64 s[0:1], s2, s3",
            "s_bfm_b64 s[0:1], 32, 16",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "s_bfe_u64 s0, s[2:3], s4",
            "s_bfe_i64 s0, s[2:3], s4",
            "s_bfe_u64 s[0:1], s2, s4",
            "s_bfe_i64 s[0:1], s2, s4",
            "s_bfe_u64 s[0:1], s[2:3]",
            "s_bfe_i64 s[0:1], s[2:3]",
            "s_bfe_u64 s[0:1], 0x8123456789abcdef, s4",
            "s_bfe_i64 s[0:1], 0x8123456789abcdef, s4",
            "s_bfm_b64 s0, s2, s3",
            "s_bfm_b64 s[0:1], s2",
            "s_bfm_b64 s[0:1], s[2:3], s4",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("s_bfe_u64", "s_bfe_i64", "s_bfm_b64"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        # 64-bit integer comparisons: register pairs or an inline integer; VOPC writes VCC and takes a VGPR pair
        # as src1. Literals, floats, modifiers, SGPR src1 or an SGPR destination on e32, and exec writers are refused.
        for op in ("f", "lt", "eq", "le", "gt", "ne", "ge", "t"):
            for kind in ("i64", "u64"):
                for line in (f"v_cmp_{op}_{kind}_e32 vcc_lo, v[18:19], v[20:21]",
                             f"v_cmp_{op}_{kind} vcc_lo, s[44:45], v[20:21]",
                             f"v_cmp_{op}_{kind}_e32 vcc_lo, -1, v[20:21]",
                             f"v_cmp_{op}_{kind}_e64 s52, v[18:19], v[19:20]",
                             f"v_cmp_{op}_{kind}_e64 vcc_lo, s[44:45], s[46:47]",
                             f"v_cmp_{op}_{kind}_e64 s[52:53], 64, v[20:21]",
                             # Without a suffix LLVM 15 picks e64 when the operands need it.
                             f"v_cmp_{op}_{kind} s[52:53], v[18:19], v[20:21]",
                             f"v_cmp_{op}_{kind} vcc_lo, v[18:19], s[44:45]"):
                    self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_cmp_lt_i64_e32 vcc_lo, v[18:19], s[44:45]",
            "v_cmp_lt_i64_e32 vcc_lo, v[18:19], 64",
            "v_cmp_lt_i64_e32 s52, v[18:19], v[20:21]",
            "v_cmp_lt_u64_e64 s52, 0x12345678, v[20:21]",
            "v_cmp_lt_u64_e64 s52, 1.0, v[20:21]",
            "v_cmp_lt_u64_e64 s52, v[18:19], v[20:21] clamp",
            "v_cmp_lt_u64_e64 s52, -v[18:19], v[20:21]",
            "v_cmp_lt_u64_e64 s52, v18, v[20:21]",
            "v_cmp_lt_u64_e64 exec, v[18:19], v[20:21]",
            "v_cmpx_lt_i64_e32 v[18:19], v[20:21]",
            "v_cmpx_lt_u64_e64 v[18:19], v[20:21]",
            "v_cmp_lt_f64_e64 s52, v[18:19], v[20:21]",
            "v_cmp_ngt_u64_e64 s52, v[18:19], v[20:21]",
            "v_cmp_lt_u64_e64 s52, 65, v[20:21]",
            "v_cmp_lt_u64_e64 s52, v[18:19], -17",
            "v_cmp_lt_u64_e64 s52, v[255:256], v[20:21]",
            "v_cmp_lt_u64_e64 s52, s[45:46], v[20:21]",
            "v_cmp_lt_i64_e32 vcc_lo, s[45:46], v[20:21]",
            "v_cmp_lt_u64_e64 s52, v[18:20], v[20:21]",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for line in (
            "v_readfirstlane_b32 s0, v1",
            "v_readfirstlane_b32 s44, v12",
            "v_readfirstlane_b32 s10, 0x1234",
            "v_readfirstlane_b32 s10, 42",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_readfirstlane_b32 v0, v1",
            "v_readfirstlane_b32 s0, v1, v2",
            "v_readfirstlane_b32 s0",
            "v_readfirstlane_b32",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        self.assertTrue(is_opcode_recognized("v_readfirstlane_b32"))
        for line in (
            "v_alignbit_b32 v1, v2, v3, v4",
            "v_alignbit_b32_e64 v1, v2, v3, 4",
            "v_alignbit_b32 v1, s2, v3, 31",
            "v_alignbyte_b32 v1, v2, v3, v4",
            "v_alignbyte_b32_e64 v1, s2, v3, 2",
            "v_alignbyte_b32 v1, v2, s3, 0",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_alignbit_b32_e32 v1, v2, v3, 4",
            "v_alignbyte_b32_e32 v1, v2, v3, 2",
            "v_alignbit_b32 v1, v2, v3",
            "v_alignbyte_b32 v1, v2, v3",
            "v_alignbit_b32 v1, v2, v3, 4, 5",
            "v_alignbyte_b32 v1, v2, v3, 1, 2",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_alignbit_b32", "v_alignbyte_b32"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_add_co_u32 v1, vcc_lo, s2, v3",
            "v_sub_co_u32 v1, s[60:61], 64, v3",
            "v_subrev_co_u32 v1, vcc, v2, 0x12345678",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_add_co_u32_e64 v1, vcc, v2, v3",
            "v_sub_co_u32_e32 v1, vcc, v2, v3",
            "v_subrev_co_u32 v1, exec, v2, v3",
            "v_add_co_u32 v1, v4, v2, v3",
            "v_add_co_u32 v1, vcc, v2, v3 clamp",
            "v_sub_co_u32 v1, vcc, -v2, v3",
            "v_add_co_u32 v1, vcc, v2",
            "v_add_co_ci_u32_e64 v1, vcc, v2, v3, vcc, vcc",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_add_co_ci_u32_e32 v1, vcc_lo, s2, v3, vcc_lo",
            "v_sub_co_ci_u32_e32 v1, vcc, 0x12345678, v3, vcc",
            "v_subrev_co_ci_u32_e32 v1, vcc_lo, -16, v3, vcc_lo",
            "v_add_co_ci_u32_e64 v1, vcc, v2, v3, vcc",
            "v_sub_co_ci_u32_e64 v1, s5, s2, 7, s6",
            "v_subrev_co_ci_u32_e64 v1, s[4:5], v2, 2.0, s[6:7]",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_add_co_ci_u32 v1, vcc_lo, v2, v3, vcc_lo",
            "v_add_co_ci_u32_e32 v1, vcc_lo, v2, s3, vcc_lo",
            "v_sub_co_ci_u32_e32 v1, s5, v2, v3, vcc_lo",
            "v_subrev_co_ci_u32_e32 v1, vcc_lo, v2, v3, s6",
            "v_add_co_ci_u32_e64 v1, s5, v2, v3, s6 clamp",
            "v_sub_co_ci_u32_e64 v1, s5, -v2, v3, s6",
            "v_subrev_co_ci_u32_e64 v1, exec_lo, v2, v3, s6",
            "v_add_co_ci_u32_e64 v1, s5, v2, v3, v6",
            "v_add_co_ci_u32_e64 v1, s5, v2, v3",
            "v_add_co_ci_u32_dpp v1, vcc_lo, v2, v3, vcc_lo quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_add_co_ci_u32_sdwa v1, vcc_lo, v2, v3, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD "
            "src0_sel:DWORD src1_sel:DWORD",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_cvt_pk_u16_u32 v1, v2, v3",
            "v_cvt_pk_u16_u32 v1, 0x12345, v3",
            "v_cvt_pk_i16_i32 v1, s2, v3",
            "v_cvt_pk_i16_i32_e64 v1, -16, v3",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_cvt_pk_u16_u32_e32 v1, v2, v3",
            "v_cvt_pk_i16_i32_e32 v1, v2, v3",
            "v_cvt_pk_u16_u32 v1, v2",
            "v_cvt_pk_i16_i32 v1, v2, v3, v4",
            "v_cvt_pk_i16_i32 v1, v2, v3 clamp",
            "v_cvt_pknorm_i16_f32 v1, v2, v3",
            "v_cvt_pknorm_u16_f32 v1, v2, v3",
            "v_cvt_pk_u8_f32 v1, v2, v3, v4",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_sad_hi_u8 v1, v2, v3, v4",
            "v_sad_u32 v1, s2, v3, 64",
            "v_sad_u32_e64 v1, v2, v3, v4",
            "v_msad_u8 v1, v2, 0x7f00ff01, s4",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_sad_hi_u8_e32 v1, v2, v3, v4",
            "v_sad_u32 v1, v2, v3",
            "v_msad_u8 v1, v2, v3, v4, v5",
            "v_msad_u8 v1, v2, v3, v4 clamp",
            "v_sad_u8 v1, v2, v3, v4",
            "v_sad_u16 v1, v2, v3, v4",
            "v_mqsad_pk_u16_u8 v[1:2], v[3:4], v5, v[6:7]",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_sad_hi_u8", "v_sad_u32", "v_msad_u8"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_add3_u32 v1, v2, v3, v4",
            "v_add3_u32_e64 v1, s2, v3, 4",
            "v_or3_b32 v1, v2, v3, v4",
            "v_xor3_b32 v1, v2, v3, v4",
            "v_and_or_b32 v1, v2, v3, v4",
            "v_xad_u32 v1, v2, v3, v4",
            "v_lshl_or_b32 v1, v2, v3, v4",
            "v_lshl_add_u32 v1, v2, v3, v4",
            "v_add_lshl_u32 v1, v2, v3, v4",
            "v_add_lshl_u32_e64 v1, s2, v3, 31",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_add3_u32_e32 v1, v2, v3, 4",
            "v_or3_b32_e32 v1, v2, v3, 4",
            "v_xor3_b32_e32 v1, v2, v3, 4",
            "v_and_or_b32_e32 v1, v2, v3, 4",
            "v_xad_u32_e32 v1, v2, v3, 4",
            "v_lshl_or_b32_e32 v1, v2, v3, 4",
            "v_lshl_add_u32_e32 v1, v2, v3, 4",
            "v_add_lshl_u32_e32 v1, v2, v3, 4",
            "v_add3_u32 v1, v2, v3",
            "v_add3_u32 v1, v2, v3, 4, 5",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in (
            "v_add3_u32", "v_or3_b32", "v_xor3_b32", "v_and_or_b32",
            "v_xad_u32", "v_lshl_or_b32", "v_lshl_add_u32", "v_add_lshl_u32",
        ):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for unsupported in ("v_cvt_f32_i32 v1, v2",
                            "v_cvt_f32_i32_e32 v1, v2 clamp",
                            "v_cvt_f32_u32_e64 v1, -v2",
                            "v_cvt_f32_u32_e64 v1, v2 omod:2",
                            "v_cvt_f32_i32_e32 v1, 1.5",
                            "s_nand_b32 s1, s2",
                            "s_nor_b32 s1, s2",
                            "s_xnor_b32 s1, s2",
                            "s_nand_b32 v1, s2, s3",
                            "s_bfm_b32 s1, s2",
                            "s_bfm_b32 v1, s2, s3",
                            "s_bfm_b32 s1, v2, s3",
                            "s_bfm_b32 s[2:3], s2, s3"):
            self.assertFalse(is_instruction_supported(unsupported), unsupported)

        for line in ("v_swap_b32 v1, v2", "v_swap_b32_e32 v1, v2", "v_swap_b32 v1, v1", "v_swap_b32 v255, v254"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_swap_b32 v1, s2",
                     "v_swap_b32 s1, v2",
                     "v_swap_b32_e64 v1, v2",
                     "v_swap_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                     "v_swap_b32_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
                     "v_swap_b32 v1, v2 clamp",
                     "v_swap_b32 v1, -v2",
                     "v_swap_b32 v1",
                     "v_swaprel_b32 v1, v2"):
            self.assertFalse(is_instruction_supported(line), line)
        self.assertTrue(is_opcode_recognized("v_swap_b32"))

        # Verify unsupported operands or spellings are rejected by scanner
        self.assertFalse(is_opcode_recognized("v_imaginary_opcode_sdwa"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt null, 0x8"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt null, 8"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt 0x8"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt 8"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt invalid"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt null, invalid"))
        self.assertFalse(is_instruction_supported("s_waitcnt expcnt(8)"))
        self.assertFalse(is_instruction_supported("s_waitcnt vmcnt(64)"))
        self.assertFalse(is_instruction_supported("s_waitcnt lgkmcnt(64)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(7) lgkmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(7) lgkmcnt(63)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt 0"))
        self.assertTrue(is_instruction_supported("s_waitcnt 0x0"))
        self.assertFalse(is_instruction_supported("s_waitcnt foo"))
        self.assertFalse(is_instruction_supported("s_waitcnt vmcnt(0) lgkmcnt(0) vmcnt(1)"))
        self.assertFalse(is_instruction_supported("s_waitcnt lgkmcnt(0)lgkmcnt(1)"))
        self.assertFalse(is_instruction_supported("s_waitcnt expcnt(0) expcnt(1)"))

    def test_sopk_waitcnt_forms(self):
        """null with a count of 0..63 is supported for the three SOPK wait forms; an SGPR or larger count is not."""
        for op in ("s_waitcnt_vmcnt", "s_waitcnt_lgkmcnt", "s_waitcnt_vscnt"):
            for count in ("0x0", "0x3", "0x3f", "0", "1", "9", "10", "63"):
                self.assertTrue(is_instruction_supported(f"{op} null, {count}"), (op, count))
            for unsupported in ("s5, 0x1", "s0, 0x0", "vcc_lo, 0x0", "0x1", "null", "null, 0x40", "null, 64",
                                "null, 0xffff", "null, 0x400", "null, -1", "null, invalid", "null,"):
                self.assertFalse(is_instruction_supported(f"{op} {unsupported}"), (op, unsupported))
        self.assertFalse(is_instruction_supported("s_waitcnt_depctr null, 0x0"))
        decoded = disassemble_binary_shader(bytes.fromhex("00007dbc" "00007dbd" "0000fdbb" "000081bf"))
        self.assertEqual(decoded, ["s_waitcnt_vmcnt null, 0x0", "s_waitcnt_lgkmcnt null, 0x0",
                                   "s_waitcnt_vscnt null, 0x0", "s_endpgm"])
        result = analyze_shader_instructions("sopk_wait", decoded)
        self.assertEqual((result["supported_instructions"], result["unsupported_count"]), (4, 0))
        result = analyze_shader_instructions("sopk_wait_sgpr", ["s_waitcnt_vmcnt s5, 0x1", "s_endpgm"])
        self.assertEqual(result["unsupported_count"], 1)

    def test_shift64_scanner_support(self):
        """Issue #928: the 64-bit VOP3 shifts are supported with at most one SGPR or literal source."""
        for line in ("v_lshlrev_b64 v[12:13], v20, v[18:19]", "v_lshrrev_b64 v[2:3], s46, v[4:5]",
                     "v_ashrrev_i64 v[2:3], 0x41, v[4:5]", "v_lshlrev_b64 v[2:3], v1, s[4:5]",
                     "v_ashrrev_i64 v[2:3], 3, -1", "v_lshlrev_b64_e64 v[2:3], v1, v[4:5]"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_lshlrev_b64 v[2:3], s4, s[4:5]", "v_lshlrev_b64 v[2:3], 0x41, s[4:5]",
                     "v_lshlrev_b64 v[2:3], v1, 0x41", "v_lshlrev_b64 v[2:3], v1, v[4:5] clamp",
                     "v_lshlrev_b64_e32 v[2:3], v1, v[4:5]", "v_lshlrev_b64 v[2:3], -v1, v[4:5]"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_low16_binary16_e32_scanner_support(self):
        """The ten plain binary16 e32 forms pixel_aot lowers (low 16 bits only) are supported; modifiers, e64 and
        sources the translator refuses are not."""
        supported = []
        for op in ("add", "sub", "subrev", "mul"):
            supported += [f"v_{op}_f16_e32 v24, v18, v19", f"v_{op}_f16_e32 v24, 1.0, v19",
                          f"v_{op}_f16_e32 v24, -0.5, v19", f"v_{op}_f16_e32 v24, 0x3c00, v19",
                          f"v_{op}_f16_e32 v24, s45, v19", f"v_{op}_f16_e32 v24, 0, v19"]
        for op in ("floor", "ceil", "trunc", "rndne"):
            supported += [f"v_{op}_f16_e32 v24, v18", f"v_{op}_f16_e32 v24, 2.0", f"v_{op}_f16_e32 v24, 0x4248"]
        for kind in ("i16", "u16"):
            supported += [f"v_cvt_f16_{kind}_e32 v24, v18", f"v_cvt_f16_{kind}_e32 v24, s45",
                          f"v_cvt_f16_{kind}_e32 v24, 5", f"v_cvt_f16_{kind}_e32 v24, 0x1234"]
        for line in supported:
            self.assertTrue(is_instruction_supported(line), line)
        refused = [
            "v_add_f16_e32 v24, v18, 1.0",          # e32 src1 must be a VGPR
            "v_add_f16_e32 v24, -v18, v19",         # source modifiers
            "v_add_f16_e32 v24, |v18|, v19",
            "v_add_f16_e32 v24, 3, v19",            # integer inline constant other than 0
            "v_add_f16_e32 v24, 0x10000, v19",      # literal wider than 16 bits
            "v_add_f16_e32 v24, 3.0, v19",          # not a binary16 inline constant
            "v_add_f16_e32 v24, v256, v19",
            "v_add_f16_e32 v24, v18, v19 clamp",
            "v_mul_f16_e64 v24, v18, v19",
            "v_add_f16 v24, v18, v19",
            "v_add_f16_e32 v24, v18",
            "v_floor_f16_e32 v24, -v18",
            "v_floor_f16_e32 v24, v18 clamp",
            "v_floor_f16_e64 v24, v18",
            "v_fract_f16_e32 v24, v18",
            "v_cvt_f16_i16_e32 v24, 65",
            "v_cvt_f16_u16_e32 v24, 0x10000",
            "v_cvt_f16_i16_e32 v24, 1.0",
            "v_cvt_f16_u16_e32 v24, v18 clamp",
            "v_cvt_f16_i16_e64 v24, v18",
        ]
        for line in refused:
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_add_f16", "v_subrev_f16", "v_rndne_f16", "v_cvt_f16_u16"):
            self.assertIn(opcode, KNOWN_RDNA2_BASE_OPCODES)

    def test_sopk_s_version(self):
        """s_version with a 16-bit immediate is supported and known; larger, negative and register operands are not."""
        for operand in ("0x0", "0x1004", "0xffff", "0x00ff", "0", "1", "4100", "9999", "10000", "65535", "59999",
                        "65000", "65529"):
            self.assertTrue(is_instruction_supported(f"s_version {operand}"), operand)
        for operand in ("0x10000", "0xfffff", "65536", "65540", "99999", "100000", "4294967295", "-1", "-0x1",
                        "s0", "null", "v0", "0, 0", "0x1004 0x1", "007", "0x", "1.0", "abc", ""):
            self.assertFalse(is_instruction_supported(f"s_version {operand}".rstrip()), operand)
        self.assertIn("s_version", KNOWN_RDNA2_BASE_OPCODES)
        decoded = disassemble_binary_shader(bytes.fromhex("041080b0" "000080b0" "ffff80b0" "000081bf"))
        self.assertEqual(decoded, ["s_version 0x1004", "s_version 0x0", "s_version 0xffff", "s_endpgm"])
        result = analyze_shader_instructions("s_version", decoded)
        self.assertEqual((result["supported_instructions"], result["unsupported_count"]), (4, 0))
        result = analyze_shader_instructions("s_version_wide", ["s_version 0x10000", "s_endpgm"])
        self.assertEqual(result["unsupported_count"], 1)

    def test_gfx1013_waitcnt_expcnt_decode_and_lower(self):
        """Assemble/decode GFX1013 s_waitcnt_expcnt binary bytes and lower with pixel_aot."""
        from pixel_aot import lower
        code_bytes = bytes.fromhex(
            "0000fdbc"  # s_waitcnt_expcnt null, 0x0
            "000080bf"  # s_nop 0
            "0300fdbc"  # s_waitcnt_expcnt null, 0x3
            "000081bf"  # s_endpgm
        )
        decoded = disassemble_binary_shader(code_bytes)
        self.assertEqual(decoded[0], "s_waitcnt_expcnt null, 0x0")
        self.assertEqual(decoded[1], "s_nop 0")
        self.assertEqual(decoded[2], "s_waitcnt_expcnt null, 0x3")
        self.assertEqual(decoded[3], "s_endpgm")

        res = analyze_shader_instructions("expcnt_test", decoded)
        self.assertTrue(res["opcodes_recognized"])
        self.assertEqual(res["supported_instructions"], 4)
        self.assertEqual(res["unsupported_count"], 0)
        self.assertEqual(res["support_percentage"], 100.0)

        full_shader = ["s_waitcnt_expcnt null, 0x0"] + [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_waitcnt_expcnt null, 0x3",
            "s_endpgm",
        ]
        res_full = analyze_shader_instructions("canonical_ps", full_shader)
        self.assertTrue(res_full["is_fully_supported"])
        self.assertTrue(res_full["translator_accepted"])
        hlsl = lower(full_shader, 20, 24, [1, 2, 3, 4])
        self.assertIn("return float4(", hlsl)

    def test_gfx1013_v_nop_decode_and_lower(self):
        """Assemble/decode GFX1013 v_nop binary bytes and lower with pixel_aot."""
        from pixel_aot import lower
        code_bytes = bytes.fromhex(
            "0000007e"  # v_nop
            "000080bf"  # s_nop 0
            "000081bf"  # s_endpgm
        )
        decoded = disassemble_binary_shader(code_bytes)
        self.assertEqual(decoded[0], "v_nop")
        self.assertEqual(decoded[1], "s_nop 0")
        self.assertEqual(decoded[2], "s_endpgm")

        res = analyze_shader_instructions("v_nop_test", decoded)
        self.assertTrue(res["opcodes_recognized"])
        self.assertEqual(res["supported_instructions"], 3)
        self.assertEqual(res["unsupported_count"], 0)
        self.assertEqual(res["support_percentage"], 100.0)

        full_shader = ["v_nop"] + [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "v_nop",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res_full = analyze_shader_instructions("canonical_ps_with_v_nop", full_shader)
        self.assertTrue(res_full["is_fully_supported"])
        self.assertTrue(res_full["translator_accepted"])
        hlsl = lower(full_shader, 20, 24, [1, 2, 3, 4])
        self.assertIn("return float4(", hlsl)

        # Negative control: v_nop after color export must be rejected
        with_v_nop_after_exp = full_shader[:-1] + ["v_nop", "s_endpgm"]
        with self.assertRaises(ValueError):
            lower(with_v_nop_after_exp, 20, 24, [1, 2, 3, 4])

    def test_valu_3operand_float_math_scanner(self):
        """Test scanning v_div_fmas_f32 in scan_unsupported_shaders and rejecting v_fma_f16 / v_div_scale_f32."""
        valid_insns = [
            "v_div_fmas_f32 v0, v1, v2, v3",
            "v_div_fmas_f32_e64 v0, v1, v2, v3",
            "v_div_fmas_f32 v0, v1, v2, v3 clamp",
        ]
        for insn in valid_insns:
            self.assertTrue(is_instruction_supported(insn), f"Expected supported: {insn}")

        invalid_insns = [
            "v_fma_f16 v0, v1, v2, v3",
            "v_fma_f16_e64 v0, v1, v2, v3",
            "v_fma_f16 v0, v1, v2, v3 clamp",
            "v_fma_f16 v0, v1, v2, v3 op_sel:[1,0,1]",
            "v_fma_f16 v0, v1, v2, v3 op_sel:[1,0,1] clamp",
            "v_fma_f16 v0, v1, v2, v3 clamp op_sel:[1,0,1]",
            "v_fma_f16 v0, -v1, |v2|, 0.5 op_sel:[0,1,0]",
            "v_div_scale_f32 v0, vcc_lo, v1, v2, v3",
            "v_div_scale_f32_e64 v0, vcc, v1, v2, v3",
            "v_div_scale_f32 v0, s52, v1, v2, v3",
            "v_fma_f16 v0, v1, v2",
            "v_fma_f16 v0, v1, v2, v3 omod:2",
            "v_div_scale_f32 v0, v1, v2, v3",
            "v_div_fmas_f32 v0, v1, v2",
            "v_div_fmas_f32 v0, v1, v2, v3 omod:2",
        ]
        for insn in invalid_insns:
            self.assertFalse(is_instruction_supported(insn), f"Expected unsupported: {insn}")

    def test_valu_3operand_float_math_lower(self):
        """Test lowering v_div_fmas_f32 and rejecting unverified v_fma_f16 and v_div_scale_f32."""
        div_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cmp_lt_f32 vcc_lo, 0, 1.0",
            "v_div_fmas_f32 v14, 1.0, 2.0, 3.0",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        hlsl = lower(div_shader, 20, 24, [1, 2, 3, 4])
        self.assertIn("div_fmas32(", hlsl)
        self.assertIn("4294967296.0", hlsl)

        # Negative control: v_fma_f16 must be rejected
        with self.assertRaises(ValueError):
            lower(["v_fma_f16 v22, v19, v20, v21", *div_shader], 20, 24, [1, 2, 3, 4])

        # Negative control: v_div_scale_f32 must be rejected
        with self.assertRaises(ValueError):
            lower(["v_div_scale_f32 v12, vcc_lo, v19, v19, v20", *div_shader], 20, 24, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
