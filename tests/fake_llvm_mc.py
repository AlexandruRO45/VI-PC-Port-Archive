"""Fake llvm-mc preambles for the tests of shader_disasm.py's decoding check (SHADER-804).

The answers are what conda-forge llvm-tools printed for the check's input (the three DECODER_CHECKS instructions in one
run) with --disassemble -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding.
"""
CHECK_START = "0x00 0x40 0x10 0xcc"   # the check's input starts with v_pk_mul_f16's first word

LLVM_23_1_6 = ("\tv_pk_mul_f16 v0, v1, 0xfbff3c00         ; encoding: "
               "[0x00,0x40,0x10,0xcc,0x01,0xff,0x01,0x1a,0x00,0x3c,0xff,0xfb]\n"
               "\tv_mul_f32_e64 v0, v1, v2                ; encoding: [0x00,0x00,0x08,0xd5,0x01,0x05,0x02,0x02]\n"
               "\tv_add_nc_u16 v0, 0.5, v2                ; encoding: [0x00,0x00,0x03,0xd7,0xf0,0x04,0x02,0x02]\n")
# 22.1.1 to 22.1.8: the unused SRC2 printed as 0 (0x18, 0x00 instead of 0x1a, 0x02), and 0.5 as a 12-byte literal.
LLVM_22_1_2 = ("\tv_pk_mul_f16 v0, v1, 0xfbff3c00         ; encoding: "
               "[0x00,0x40,0x10,0xcc,0x01,0xff,0x01,0x18,0x00,0x3c,0xff,0xfb]\n"
               "\tv_mul_f32_e64 v0, v1, v2                ; encoding: [0x00,0x00,0x08,0xd5,0x01,0x05,0x02,0x00]\n"
               "\tv_add_nc_u16 v0, 0x3800, v2             ; encoding: "
               "[0x00,0x00,0x03,0xd7,0xff,0x04,0x02,0x00,0x00,0x38,0x00,0x00]\n")
# 21.1.8 and 22.1.0: invalid words, then the rest read as other instructions (the literal as an exp).
LLVM_21_1_8 = ("\texp mrt0 off, off, off, off done compr vm ; encoding: [0x00,0x1c,0x00,0xf8,0x00,0x00,0x08,0xd5]\n"
               "\tv_cndmask_b32_e32 v1, v1, v2, vcc_lo    ; encoding: [0x01,0x05,0x02,0x02]\n"
               "\tv_cndmask_b32_e32 v1, 0.5, v2, vcc_lo   ; encoding: [0xf0,0x04,0x02,0x02]\n")
LLVM_21_1_8_WARNINGS = ("<stdin>:1:1: warning: invalid instruction encoding\n"
                        "<stdin>:1:21: warning: invalid instruction encoding\n"
                        "<stdin>:1:101: warning: invalid instruction encoding\n")


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
