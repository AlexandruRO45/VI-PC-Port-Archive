"""Run with Python -B verify.py REPOSITORY LLVM_BIN after compiling readback.cpp."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("repo", type=Path)
parser.add_argument("llvm", nargs="?", type=Path, help="assemble and decode the fixture again with LLVM 15")
parser.add_argument("--host", type=Path, default=Path(__file__).with_name("readback.exe"))
parser.add_argument("--backend", choices=("hardware", "warp", "both"), default="both")
parser.add_argument("--case", choices=("positive", "negative", "all"), default="all")
args = parser.parse_args()
repo, llvm = args.repo, args.llvm
sys.path.insert(0, str(repo / "tools"))
from ffx_dispatch_aot import disassemble, translate

assembly = """v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v2, 3, v0
v_lshlrev_b32_e32 v2, 16, v2
v_or_b32_e32 v3, v1, v2
v_pk_add_u16 v4, v3, 1
image_load v[5:6], v4, s[24:31] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm a16 d16
s_waitcnt vmcnt(0)
s_endpgm
"""
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    # LLVM 15 encoding of the source-owned assembly above, for Windows CI
    # machines without the AMDGPU assembler. Optional LLVM mode checks it.
    base = bytes.fromhex("87000236 8300042c 90040434 01050638 04400acc 03030118 "
                         "081700f0 040506c0 703f8cbf 000081bf")
    pc = 16
    if llvm:
        (d/"input.s").write_text(assembly)
        subprocess.run([llvm/"clang.exe","-cc1as","-triple","amdgcn--amdpal","-target-cpu","gfx1013",
                        "-filetype","obj",d/"input.s","-o",d/"input.o"],check=True)
        subprocess.run([llvm/"llvm-objcopy.exe","-O","binary","--only-section=.text",d/"input.o",d/"input.bin"],check=True)
        assert base == (d/"input.bin").read_bytes()
        lib = str(llvm/"LLVM-C.dll")
        pc, = [i for i, op in disassemble(base,lib,offsets=True) if op.startswith("v_pk_add_u16")]
    src, = struct.unpack_from("<I",base,pc+4)
    assert (src >> 9) & 511 == 129
    # Use a literal DWORD in place of LLVM's canonical short inline constant.
    literal_src = (src & ~(511<<9)) | (255<<9)
    for name,word,dx,dy,negative in [
        ("left",0x0000ffff,-1,0,False),("up",0xffff0000,0,-1,False),
        ("diagonal",0xffffffff,-1,-1,False),("wrong-high-half",0xffffffff,-1,0,True)]:
        if args.case != "all" and negative != (args.case == "negative"):
            continue
        code = base[:pc+4]+struct.pack("<II",literal_src,word)+base[pc+8:]
        lines = assembly.splitlines()
        # LLVM 15 prints both all-ones and the left offset using the same
        # short spelling. Only the encoded literal distinguishes them.
        spelling = "0xffff0000" if word == 0xffff0000 else "-1"
        lines[4] = f"v_pk_add_u16 v4, v3, {spelling}"
        instructions = list(zip((0,4,8,12,16,28,36,40), lines))
        if llvm:
            assert instructions == disassemble(code,lib,offsets=True)
        hlsl = translate("rcas",instructions,code)
        # D3D11 test bindings and observation point only. All generated coordinate
        # and load instructions remain unchanged.
        hlsl = hlsl.replace('#include "ffx_dispatch_io.hlsli"', """
Texture2D<float> tex : register(t0);
RWStructuredBuffer<uint> observed : register(u0);
float4 load16(uint2 p) { return float4(tex.Load(int3(p,0)),0,0,0); }
""").replace('[RootSignature(CONSUMER_ROOT)]','')
        assert hlsl.rstrip().endswith("}")
        (d/"probe.hlsl").write_text(hlsl.rstrip()[:-1]+"observed[lane]=(uint)f16tof32(v5 & 65535u);\n}")
        for backend in (("hardware","warp") if args.backend == "both" else (args.backend,)):
            r=subprocess.run([args.host.resolve(),d/"probe.hlsl",backend,str(dx),str(dy)],capture_output=True,text=True)
            print(name,backend,r.stdout.strip(),r.stderr.strip(),flush=True)
            assert r.returncode == int(negative), "Unexpected comparison result"
            if negative: assert "center=18 mismatches=55/64" in r.stdout
