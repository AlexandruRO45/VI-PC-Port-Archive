"""Fake llvm-mc preambles for the tests of shader_disasm.py's decoding check (SHADER-804).

The answers are what conda-forge llvm-tools printed for the check's input (the four DECODER_CHECKS instructions in one
run) with --disassemble -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding. The scalar load's bytes are the ones in
FidelityFX FSPX program 05 at 0x2120 (abilliontokens, #4843).
"""
CHECK_START = "0x16 0x17 0x30 0xf4"   # the check's input starts with s_buffer_load_dwordx16's first word

SMEM_23 = ("\ts_buffer_load_dwordx16 [s92,s93,s94,s95,s96,s97,s98,s99,s100,s101,s102,s103,s104,s105,vcc_lo,vcc_hi], "
           "s[44:47], 0x1d070 ; encoding: [0x16,0x17,0x30,0xf4,0x70,0xd0,0x01,0xfa]\n")
LLVM_23_1_6 = (SMEM_23 +
               "\tv_pk_mul_f16 v0, v1, 0xfbff3c00         ; encoding: "
               "[0x00,0x40,0x10,0xcc,0x01,0xff,0x01,0x1a,0x00,0x3c,0xff,0xfb]\n"
               "\tv_mul_f32_e64 v0, v1, v2                ; encoding: [0x00,0x00,0x08,0xd5,0x01,0x05,0x02,0x02]\n"
               "\tv_add_nc_u16 v0, 0.5, v2                ; encoding: [0x00,0x00,0x03,0xd7,0xf0,0x04,0x02,0x02]\n")
# 22.1.1 to 22.1.8: the scalar load is two invalid words; its second word and the next decode as an exp, and 0.5 as
# a 12-byte literal.
JUNK_EXP = ("\texp mrt7 off, off, off, off vm          ; encoding: [0x70,0x10,0x00,0xf8,0x00,0x40,0x10,0xcc]\n"
            "\texp mrt0 off, off, off, off done compr vm ; encoding: [0x00,0x1c,0x00,0xf8,0x00,0x00,0x08,0xd5]\n"
            "\tv_cndmask_b32_e32 v1, v1, v2, vcc_lo    ; encoding: [0x01,0x05,0x02,0x02]\n")
LLVM_22_1_2 = (JUNK_EXP +
               "\tv_add_nc_u16 v0, 0x3800, v2             ; encoding: "
               "[0x00,0x00,0x03,0xd7,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]\n")
LLVM_22_1_2_WARNINGS = ("<stdin>:1:1: warning: invalid instruction encoding\n"
                        "<stdin>:1:61: warning: invalid instruction encoding\n")
# 21.1.8 and 22.1.0: the same, and v_add_nc_u16 (unused SRC2 of 0x80) is one more invalid word.
LLVM_21_1_8 = JUNK_EXP + "\tv_cndmask_b32_e32 v1, 0.5, v2, vcc_lo   ; encoding: [0xf0,0x04,0x02,0x02]\n"
LLVM_21_1_8_WARNINGS = LLVM_22_1_2_WARNINGS + "<stdin>:1:141: warning: invalid instruction encoding\n"


def preamble(version: str, stdout: str, stderr: str = "") -> str:
    """Python source to start a fake llvm-mc with: it answers --version and the decoding check as LLVM `version`
    did, and hands any other input to the rest of the fake, which reads stdin as usual."""
    return (f"import io, sys\n"
            f"if '--version' in sys.argv:\n"
            f"    print('LLVM (http://llvm.org/):\\n  LLVM version {version}')\n"
            f"    sys.exit(0)\n"
            f"_input = sys.stdin.read()\n"
            f"if _input.startswith({CHECK_START!r}):\n"
            f"    sys.stdout.write({stdout!r})\n"
            f"    sys.stderr.write({stderr!r})\n"
            f"    sys.exit(0)\n"
            f"sys.stdin = io.StringIO(_input)\n")


GOOD = preamble("23.1.6", LLVM_23_1_6)
