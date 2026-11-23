"""The two SPIR-V modules of the #5533 reproducer (TEST-1296), kept as text because the repository ignores .spv
and .spvasm files. Both are hand-written and synthetic: no game shader went into them.

    python -I modules.py [out_dir]     writes pk_ashr_hi16.spvasm and pk_ashr_hi16_in_place.spvasm (default: here),
                                       and assembles each to .spv when spirv-as is on PATH

Standard library only. See README.md for what the modules show and how to run them with vk_repro.c.
"""
import os
import shutil
import subprocess
import sys

PK_ASHR_HI16 = """\
; Reduced reproducer for an AMD Windows driver miscompile (Leonida #5533; seen on 32.0.21045.29003, gfx1036).
; Plain Vulkan compute, no extensions: one storage buffer at set 0, binding 0; thread i writes w[i].
;
; v24 = 0xcf1db88d, then its high half is replaced by v34's: v24 = 0x4000b88d.
; Both result halves are hi16(v24) sign-extended, >> (bits 19:16 of v30 = 0), & 0xFFFF; the high one is placed
; with << 16. Expected word: 0x40004000. The AMD driver writes 0xb88d4000: the high result half gets v24's LOW
; half, as if the OpBitFieldSExtract offset 16 were lost.
;
; Each value is stored to and loaded from a Function-scope array, behind an OpSelect on a per-thread flag that is
; always true at run time (the translator's EXEC model); without that the module computes the right word.
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %lid %buf
               OpExecutionMode %main LocalSize 64 1 1
               OpDecorate %lid BuiltIn LocalInvocationId
               OpDecorate %rt ArrayStride 4
               OpDecorate %Block Block
               OpMemberDecorate %Block 0 Offset 0
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
        %u0 = OpConstant %uint 0
        %u1 = OpConstant %uint 1
        %u4 = OpConstant %uint 4
       %u16 = OpConstant %uint 16
       %u24 = OpConstant %uint 24
       %u29 = OpConstant %uint 29
       %u30 = OpConstant %uint 30
       %u31 = OpConstant %uint 31
       %u34 = OpConstant %uint 34
      %u256 = OpConstant %uint 256
     %uffff = OpConstant %uint 0xFFFF
     %uones = OpConstant %uint 0xFFFFFFFF
    %init24 = OpConstant %uint 0xcf1db88d
    %init30 = OpConstant %uint 0xbe204b94
    %init34 = OpConstant %uint 0x4000bd60
       %true = OpConstantTrue %bool
       %regs = OpTypeArray %uint %u256
      %zregs = OpConstantNull %regs
      %pRegs = OpTypePointer Function %regs
      %pUfn  = OpTypePointer Function %uint
      %pBfn  = OpTypePointer Function %bool
         %rt = OpTypeRuntimeArray %uint
      %Block = OpTypeStruct %rt
     %pBlock = OpTypePointer StorageBuffer %Block
      %pUsb  = OpTypePointer StorageBuffer %uint
      %pIn   = OpTypePointer Input %v3uint
        %buf = OpVariable %pBlock StorageBuffer
        %lid = OpVariable %pIn Input
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %r = OpVariable %pRegs Function %zregs
       %live = OpVariable %pBfn Function %true
        %id3 = OpLoad %v3uint %lid
         %id = OpCompositeExtract %uint %id3 0
       %lane = OpBitwiseAnd %uint %id %u31
       %mask = OpShiftRightLogical %uint %uones %lane
        %bit = OpBitwiseAnd %uint %mask %u1
         %on = OpINotEqual %bool %bit %u0
               OpStore %live %on
       %p24a = OpAccessChain %pUfn %r %u24
         %l1 = OpLoad %bool %live
         %s1 = OpSelect %uint %l1 %init24 %u256
               OpStore %p24a %s1
       %p30a = OpAccessChain %pUfn %r %u30
         %l2 = OpLoad %bool %live
         %s2 = OpSelect %uint %l2 %init30 %u256
               OpStore %p30a %s2
       %p34a = OpAccessChain %pUfn %r %u34
         %l3 = OpLoad %bool %live
         %s3 = OpSelect %uint %l3 %init34 %u256
               OpStore %p34a %s3
        %v24 = OpLoad %uint %p24a
       %keep = OpBitwiseAnd %uint %v24 %uffff
        %v34 = OpLoad %uint %p34a
       %h34  = OpShiftRightLogical %uint %v34 %u16
       %h34s = OpShiftLeftLogical %uint %h34 %u16
       %n24  = OpBitwiseOr %uint %keep %h34s
               OpStore %p24a %n24
        %a30 = OpLoad %uint %p30a
        %b24 = OpLoad %uint %p24a
       %amt0 = OpBitFieldUExtract %uint %a30 %u16 %u4
       %val0 = OpBitFieldSExtract %uint %b24 %u16 %u16
        %sh0 = OpShiftRightArithmetic %uint %val0 %amt0
        %lo  = OpBitwiseAnd %uint %sh0 %uffff
       %amt1 = OpBitFieldUExtract %uint %a30 %u16 %u4
       %val1 = OpBitFieldSExtract %uint %b24 %u16 %u16
        %sh1 = OpShiftRightArithmetic %uint %val1 %amt1
       %hi16 = OpBitwiseAnd %uint %sh1 %uffff
        %hi  = OpShiftLeftLogical %uint %hi16 %u16
        %res = OpBitwiseOr %uint %lo %hi
       %p29a = OpAccessChain %pUfn %r %u29
               OpStore %p29a %res
        %out = OpLoad %uint %p29a
       %pout = OpAccessChain %pUsb %buf %u0 %id
               OpStore %pout %out
               OpReturn
               OpFunctionEnd
"""

PK_ASHR_HI16_IN_PLACE = """\
; Control for pk_ashr_hi16.spvasm (Leonida #5533): the same module, except that the high result half is computed
; where it sits instead of being extracted, shifted and moved back up. High half = ((v24 & 0xFFFF0000) as a signed
; word) >> amt, & 0xFFFF0000; it equals (SExtract(v24, 16, 16) >> amt & 0xFFFF) << 16 for amt 0..15.
; Expected word: 0x40004000, which the AMD driver that miscompiles pk_ashr_hi16.spvasm also writes.
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %lid %buf
               OpExecutionMode %main LocalSize 64 1 1
               OpDecorate %lid BuiltIn LocalInvocationId
               OpDecorate %rt ArrayStride 4
               OpDecorate %Block Block
               OpMemberDecorate %Block 0 Offset 0
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
        %u0 = OpConstant %uint 0
        %u1 = OpConstant %uint 1
        %u4 = OpConstant %uint 4
       %u16 = OpConstant %uint 16
       %u24 = OpConstant %uint 24
       %u29 = OpConstant %uint 29
       %u30 = OpConstant %uint 30
       %u31 = OpConstant %uint 31
       %u34 = OpConstant %uint 34
      %u256 = OpConstant %uint 256
     %uffff = OpConstant %uint 0xFFFF
       %uhi = OpConstant %uint 0xFFFF0000
     %uones = OpConstant %uint 0xFFFFFFFF
    %init24 = OpConstant %uint 0xcf1db88d
    %init30 = OpConstant %uint 0xbe204b94
    %init34 = OpConstant %uint 0x4000bd60
       %true = OpConstantTrue %bool
       %regs = OpTypeArray %uint %u256
      %zregs = OpConstantNull %regs
      %pRegs = OpTypePointer Function %regs
      %pUfn  = OpTypePointer Function %uint
      %pBfn  = OpTypePointer Function %bool
         %rt = OpTypeRuntimeArray %uint
      %Block = OpTypeStruct %rt
     %pBlock = OpTypePointer StorageBuffer %Block
      %pUsb  = OpTypePointer StorageBuffer %uint
      %pIn   = OpTypePointer Input %v3uint
        %buf = OpVariable %pBlock StorageBuffer
        %lid = OpVariable %pIn Input
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %r = OpVariable %pRegs Function %zregs
       %live = OpVariable %pBfn Function %true
        %id3 = OpLoad %v3uint %lid
         %id = OpCompositeExtract %uint %id3 0
       %lane = OpBitwiseAnd %uint %id %u31
       %mask = OpShiftRightLogical %uint %uones %lane
        %bit = OpBitwiseAnd %uint %mask %u1
         %on = OpINotEqual %bool %bit %u0
               OpStore %live %on
       %p24a = OpAccessChain %pUfn %r %u24
         %l1 = OpLoad %bool %live
         %s1 = OpSelect %uint %l1 %init24 %u256
               OpStore %p24a %s1
       %p30a = OpAccessChain %pUfn %r %u30
         %l2 = OpLoad %bool %live
         %s2 = OpSelect %uint %l2 %init30 %u256
               OpStore %p30a %s2
       %p34a = OpAccessChain %pUfn %r %u34
         %l3 = OpLoad %bool %live
         %s3 = OpSelect %uint %l3 %init34 %u256
               OpStore %p34a %s3
        %v24 = OpLoad %uint %p24a
       %keep = OpBitwiseAnd %uint %v24 %uffff
        %v34 = OpLoad %uint %p34a
       %h34  = OpShiftRightLogical %uint %v34 %u16
       %h34s = OpShiftLeftLogical %uint %h34 %u16
       %n24  = OpBitwiseOr %uint %keep %h34s
               OpStore %p24a %n24
        %a30 = OpLoad %uint %p30a
        %b24 = OpLoad %uint %p24a
       %amt0 = OpBitFieldUExtract %uint %a30 %u16 %u4
       %val0 = OpBitFieldSExtract %uint %b24 %u16 %u16
        %sh0 = OpShiftRightArithmetic %uint %val0 %amt0
        %lo  = OpBitwiseAnd %uint %sh0 %uffff
       %amt1 = OpBitFieldUExtract %uint %a30 %u16 %u4
       %val1 = OpBitwiseAnd %uint %b24 %uhi
        %sh1 = OpShiftRightArithmetic %uint %val1 %amt1
        %hi  = OpBitwiseAnd %uint %sh1 %uhi
        %res = OpBitwiseOr %uint %lo %hi
       %p29a = OpAccessChain %pUfn %r %u29
               OpStore %p29a %res
        %out = OpLoad %uint %p29a
       %pout = OpAccessChain %pUsb %buf %u0 %id
               OpStore %pout %out
               OpReturn
               OpFunctionEnd
"""


MODULES = {"pk_ashr_hi16": PK_ASHR_HI16, "pk_ashr_hi16_in_place": PK_ASHR_HI16_IN_PLACE}


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    spirv_as = shutil.which("spirv-as")
    for name, text in MODULES.items():
        src = os.path.join(out_dir, name + ".spvasm")
        with open(src, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"wrote {src}")
        if spirv_as:
            spv = os.path.join(out_dir, name + ".spv")
            r = subprocess.run([spirv_as, "--target-env", "vulkan1.2", src, "-o", spv], capture_output=True, text=True)
            if r.returncode:
                print(f"error: spirv-as failed on {src}: {r.stderr.strip()}", file=sys.stderr)
                return 1
            print(f"wrote {spv}")
    if not spirv_as:
        print("spirv-as is not on PATH: assemble with spirv-as --target-env vulkan1.2 <name>.spvasm -o <name>.spv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
