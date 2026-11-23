"""ctest vk_compute_batch: vk_compute_test --batch on one Vulkan device (runtime/test/vk_compute_test.cpp).

Five jobs of SHADER (below, assembled with the SDK's spirv-as) in one process: one that passes mirror64, one whose VKC_SGPRS makes it fail
mirror64 (exit 1) with its own dump size, one with no settings, which must pass again (nothing leaks from the
job before), one with a bad group count (exit 2) and one that copies a VKC_HEAP_FILE into the heap past the output. Each job's exit code, stdout and stderr must equal a single run of the same program with the same
settings, and its dump the single run's dump. Then a job that keeps the GPU busy for seconds runs under
--job-timeout 0.25: it must be reported as timed out (exit 124) and the batch must stop there (exit 3, no record
for the next job). Exit 0 when all hold, 1 when one fails, 77 without a Vulkan device.

Usage: check_vk_compute_batch.py VK_COMPUTE_TEST SPIRV_AS WORK_DIR [--runner CMD]
"""
import argparse
import os
import struct
import subprocess
import sys
import time

SETTINGS = ("VKC_OUT_WORDS", "VKC_DUMP", "VKC_HEAP_INIT", "VKC_HEAP_FILE", "VKC_SUBGROUP_SIZE", "VKC_SGPRS",
            "VKC_BENCH", "LEONIDA_PC_TAIL_UBO")
NO_DEVICE = (b"no Vulkan physical device", b"could not load")
# One compute group of 64 threads writes the mirror64 pattern plus s4, out[g*64 + t] = g*64 + 63 - t + s4, so
# VKC_SGPRS=0 (or none) passes "mirror64" and any other s4 fails it. Before the store each thread runs s5 rounds of a
# linear congruential step on its id (s5 = 0: none); a large s5 keeps the dispatch busy for seconds, for the
# --job-timeout case. The result only changes when the step lands on s6 with s5 != 0, which keeps the loop from being
# optimised away. Push constants as vk_compute_test sets them: s0..s31, then the heap base (words 32, 33).
SHADER = """               OpCapability Shader
               OpCapability Int64
               OpCapability PhysicalStorageBufferAddresses
               OpMemoryModel PhysicalStorageBuffer64 GLSL450
               OpEntryPoint GLCompute %main "main" %pc %wg %li
               OpExecutionMode %main LocalSize 64 1 1
               OpDecorate %arr36 ArrayStride 4
               OpMemberDecorate %PC 0 Offset 0
               OpDecorate %PC Block
               OpDecorate %wg BuiltIn WorkgroupId
               OpDecorate %li BuiltIn LocalInvocationIndex
               OpDecorate %rta ArrayStride 4
               OpMemberDecorate %Heap 0 Offset 0
               OpDecorate %Heap Block
       %void = OpTypeVoid
     %fnvoid = OpTypeFunction %void
       %uint = OpTypeInt 32 0
      %ulong = OpTypeInt 64 0
       %bool = OpTypeBool
     %v3uint = OpTypeVector %uint 3
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_4 = OpConstant %uint 4
     %uint_5 = OpConstant %uint 5
     %uint_6 = OpConstant %uint 6
    %uint_32 = OpConstant %uint 32
    %uint_33 = OpConstant %uint 33
    %uint_36 = OpConstant %uint 36
    %uint_63 = OpConstant %uint 63
    %uint_64 = OpConstant %uint 64
   %ulong_32 = OpConstant %ulong 32
      %lcg_a = OpConstant %uint 1664525
      %lcg_c = OpConstant %uint 1013904223
      %arr36 = OpTypeArray %uint %uint_36
         %PC = OpTypeStruct %arr36
     %PC_ptr = OpTypePointer PushConstant %PC
    %pcu_ptr = OpTypePointer PushConstant %uint
         %pc = OpVariable %PC_ptr PushConstant
      %in_v3 = OpTypePointer Input %v3uint
       %in_u = OpTypePointer Input %uint
         %wg = OpVariable %in_v3 Input
         %li = OpVariable %in_u Input
        %rta = OpTypeRuntimeArray %uint
       %Heap = OpTypeStruct %rta
   %heap_ptr = OpTypePointer PhysicalStorageBuffer %Heap
     %hu_ptr = OpTypePointer PhysicalStorageBuffer %uint
       %main = OpFunction %void None %fnvoid
      %entry = OpLabel
        %wgv = OpLoad %v3uint %wg
          %g = OpCompositeExtract %uint %wgv 0
          %t = OpLoad %uint %li
        %p32 = OpAccessChain %pcu_ptr %pc %uint_0 %uint_32
         %lo = OpLoad %uint %p32
        %p33 = OpAccessChain %pcu_ptr %pc %uint_0 %uint_33
         %hi = OpLoad %uint %p33
       %lo64 = OpUConvert %ulong %lo
       %hi64 = OpUConvert %ulong %hi
      %hi_up = OpShiftLeftLogical %ulong %hi64 %ulong_32
       %addr = OpBitwiseOr %ulong %hi_up %lo64
       %heap = OpConvertUToPtr %heap_ptr %addr
         %p4 = OpAccessChain %pcu_ptr %pc %uint_0 %uint_4
         %s4 = OpLoad %uint %p4
         %p5 = OpAccessChain %pcu_ptr %pc %uint_0 %uint_5
         %s5 = OpLoad %uint %p5
         %p6 = OpAccessChain %pcu_ptr %pc %uint_0 %uint_6
         %s6 = OpLoad %uint %p6
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %uint_0 %entry %inext %body
          %x = OpPhi %uint %t %entry %xnext %body
       %more = OpULessThan %bool %i %s5
               OpLoopMerge %merge %body None
               OpBranchConditional %more %body %merge
       %body = OpLabel
         %xm = OpIMul %uint %x %lcg_a
      %xnext = OpIAdd %uint %xm %lcg_c
      %inext = OpIAdd %uint %i %uint_1
               OpBranch %header
      %merge = OpLabel
        %g64 = OpIMul %uint %g %uint_64
       %base = OpIAdd %uint %g64 %uint_63
     %mirror = OpISub %uint %base %t
      %plain = OpIAdd %uint %mirror %s4
        %hit = OpIEqual %bool %x %s6
       %busy = OpINotEqual %bool %s5 %uint_0
       %both = OpLogicalAnd %bool %hit %busy
       %bump = OpSelect %uint %both %uint_1 %uint_0
        %val = OpIAdd %uint %plain %bump
        %idx = OpIAdd %uint %g64 %t
        %dst = OpAccessChain %hu_ptr %heap %uint_0 %idx
               OpStore %dst %val Aligned 4
               OpReturn
               OpFunctionEnd
"""


def plain(data):
    return data.replace(b"\r\n", b"\n")


def records(stdout):
    """[(index, exit, status, out, err)] and whether the batch reached its end line."""
    found, at, ended = [], 0, False
    while at < len(stdout):
        nl = stdout.find(b"\n", at)
        if nl < 0:
            break          # a process that died mid-line: the records before it count
        words = stdout[at:nl].decode("ascii", "replace").split()
        at = nl + 1
        if words[:1] == ["@vkc-job-end"]:
            n_out, n_err = int(words[4]), int(words[5])
            found.append((int(words[1]), int(words[2]), words[3], stdout[at:at + n_out],
                          stdout[at + n_out:at + n_out + n_err]))
            at += n_out + n_err
        elif words[:1] == ["@vkc-batch-end"]:
            ended = True
    return found, ended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("spirv_as")
    ap.add_argument("work")
    ap.add_argument("--runner", default="")
    a = ap.parse_args()
    exe = ([a.runner] if a.runner else []) + [a.exe]
    os.makedirs(a.work, exist_ok=True)
    spv, source = os.path.join(a.work, "vk_compute_batch.spv"), os.path.join(a.work, "vk_compute_batch.spvasm")
    with open(source, "w", encoding="utf-8", newline="\n") as f:
        f.write(SHADER)
    r = subprocess.run([a.spirv_as, "--target-env", "vulkan1.2", "-o", spv, source], capture_output=True, text=True)
    if r.returncode:
        print(f"spirv-as failed: {r.stdout}{r.stderr}")
        return 1
    base = {k: v for k, v in os.environ.items() if k not in SETTINGS}
    path = lambda name: os.path.join(a.work, name)   # noqa: E731
    heap_bytes = bytes(range(16))
    with open(path("heap.bin"), "wb") as f:
        f.write(heap_bytes)
    jobs = [(["2", "mirror64"], {"VKC_DUMP": "a"}),
            (["1", "mirror64"], {"VKC_SGPRS": "5", "VKC_DUMP": "b", "VKC_OUT_WORDS": "100"}),
            (["4", "mirror64"], {}),
            (["0", "mirror64"], {}),
            (["2", "mirror64"], {"VKC_HEAP_FILE": path("heap.bin") + "@0x400", "VKC_DUMP": "c"})]
    singles = []
    for args, settings in jobs:
        env = dict(base, **{k: path(v + ".single.bin") if k == "VKC_DUMP" else v for k, v in settings.items()})
        s = subprocess.run(exe + [spv] + args, capture_output=True, env=env, timeout=300)
        if s.returncode and any(m in s.stderr for m in NO_DEVICE):
            print("SKIP: no Vulkan device:", s.stderr.decode(errors="replace").strip())
            return 77
        singles.append(s)
    lines = ["\t".join([spv] + args + [f"{k}={path(v + '.batch.bin') if k == 'VKC_DUMP' else v}"
                                       for k, v in settings.items()]) for args, settings in jobs]
    with open(path("jobs.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# five jobs\n" + "\n".join(lines) + "\n")
    b = subprocess.run(exe + ["--batch", path("jobs.txt")], capture_output=True, env=base, timeout=300)
    found, ended = records(b.stdout)
    problems = []
    if b.returncode != 0 or not ended or [f[0] for f in found] != [0, 1, 2, 3, 4]:
        problems.append(f"batch: exit {b.returncode}, records {[f[0] for f in found]}, end line {ended}")
    for (index, code, status, out, err), s in zip(found, singles):
        if (code, plain(out), plain(err)) != (s.returncode, plain(s.stdout), plain(s.stderr)):
            problems.append(f"job {index}: batch gave exit {code} {out!r} {err!r}, a single run exit "
                            f"{s.returncode} {s.stdout!r} {s.stderr!r}")
        if status != "done":
            problems.append(f"job {index}: status {status}")
    if [f[1] for f in found] != [0, 1, 0, 2, 0]:
        problems.append(f"exit codes {[f[1] for f in found]}, expected [0, 1, 0, 2, 0] (job 1 fails mirror64, "
                        "job 3 has no groups)")
    for name in ("a", "b", "c"):
        single, batch = (open(path(f"{name}.{kind}.bin"), "rb").read() if os.path.exists(path(f"{name}.{kind}.bin"))
                         else None for kind in ("single", "batch"))
        if batch is None or batch != single:
            problems.append(f"dump {name}: batch and single run differ")
    if os.path.exists(path("b.batch.bin")):
        words = struct.unpack("<100I", open(path("b.batch.bin"), "rb").read())
        if words[0] != 63 + 5 or set(words[64:]) != {0xEEEEEEEE}:
            problems.append(f"dump b: out[0] = {words[0]}, expected 68, and the 36 words after the group untouched")
    if os.path.exists(path("c.batch.bin")):
        words = open(path("c.batch.bin"), "rb").read()
        if words[0x400:0x410] != heap_bytes or len(words) != 0x410:
            problems.append("dump c: the heap file is not at 0x400 in a buffer that ends with it")
    # LEONIDA_VK_VALIDATION applies to the whole batch (when the layer is installed): a job reads the same as its run.
    venv = dict(base, LEONIDA_VK_VALIDATION="1")
    vsingle = subprocess.run(exe + [spv, "2", "mirror64"], capture_output=True, env=venv, timeout=300)
    if b"is not installed" not in vsingle.stderr:
        with open(path("validation.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{spv}\t2\tmirror64\n")
        vb = subprocess.run(exe + ["--batch", path("validation.txt")], capture_output=True, env=venv, timeout=300)
        vfound, vended = records(vb.stdout)
        if not vended or len(vfound) != 1 or (vfound[0][1], plain(vfound[0][3]), plain(vfound[0][4])) != (
                vsingle.returncode, plain(vsingle.stdout), plain(vsingle.stderr)):
            problems.append(f"validation: batch gave {vfound!r}, a single run exit {vsingle.returncode} "
                            f"{vsingle.stdout!r} {vsingle.stderr!r}")
        if b"(validation on)" not in vsingle.stderr:
            problems.append("validation: a single run does not say \"(validation on)\"")
    # The timeout: s5 = 2^32 - 1 rounds per thread (seconds on any GPU), then the plain job, which must not run.
    with open(path("timeout.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{spv}\t1\tmirror64\tVKC_SGPRS=0,0xffffffff,0x12345\n{spv}\t4\tmirror64\n")
    start = time.monotonic()
    t = subprocess.run(exe + ["--batch", path("timeout.txt"), "--job-timeout", "0.25"], capture_output=True,
                       env=base, timeout=300)
    took = time.monotonic() - start
    found, ended = records(t.stdout)
    if t.returncode != 3 or ended or [(f[0], f[1], f[2]) for f in found] != [(0, 124, "timeout")]:
        problems.append(f"timeout batch: exit {t.returncode}, records {[(f[0], f[1], f[2]) for f in found]}, "
                        f"end line {ended}: {t.stderr!r}")
    print(f"batch of 5: exit codes {[f[1] for f in records(b.stdout)[0]]}; timeout batch: exit {t.returncode} "
          f"after {took:.1f} s (device open included)")
    for p in problems:
        print("FAIL:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
