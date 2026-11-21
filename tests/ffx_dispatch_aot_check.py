"""A source-built packed-coordinate regression for the dispatch translator."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffx_dispatch_aot import disassemble, translate
from ffx_scratch_reference import network


def main(assembler, objcopy, library):
    # A signed i8 dot product, replicated low half, clamp and byte insert
    # have independent integer answers on these source-owned test operands.
    fixture = ["buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen",
               "v_dot4_i32_i8 v0, v0, 0x01010101, 7",
               "v_dot4_i32_i8 v1, v1, -6, 0 op_sel_hi:[1,0,1]",
               "v_cvt_pk_u8_f32 v2, v2, 2, v3",
               "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"]
    operands = [[0x7f0080ff, 0x01020304, 0x43960000, 0x12345678]]
    assert network(list(enumerate(fixture)), operands, b"", 24) == [5, (-40)&0xffffffff, 0x12ff5678, 0x12345678]
    # Literal low half decrements x; its zero high half must leave y unchanged.
    # LLVM 15 prints this literal as -1 when disassembling a packed add.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, obj, binary = (root / ("fixture." + ext) for ext in ("s", "o", "bin"))
        source.write_text("v_pk_add_u16 v4, v0, 1\n"
                          "image_load v[5:6], v4, s[24:31] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm a16 d16\n"
                          "s_endpgm\n", encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj",
                        source, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, binary], check=True)
        assembled = binary.read_bytes()
    # llvm-mc canonicalizes a spelling of 0xffff to the short inline -1
    # form. Select the VOP3P literal source explicitly and append its DWORD.
    operands, = struct.unpack_from("<I", assembled, 4)
    assert (operands >> 9) & 511 == 129
    literal_src1 = (operands & ~(511 << 9)) | (255 << 9)
    code = assembled[:4] + struct.pack("<II", literal_src1, 0xffff) + assembled[8:]
    instructions = disassemble(code, library, offsets=True)
    hlsl = translate("rcas", instructions, code)
    assert "0xffffu >> 16" in hlsl and "0xffffffffu" not in hlsl
    assert "load16(uint2(v4 & 65535u, v4>>16))" in hlsl


if __name__ == "__main__":
    main(*sys.argv[1:])
