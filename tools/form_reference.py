"""Regenerate the instruction reference lists in docs/coverage/ (rarely needed).

Usage: out/pe-env/Scripts/python.exe tools/form_reference.py <llvm-docs-dir>

<llvm-docs-dir> holds AMDGPUAsmGFX10.rst, AMDGPUAsmGFX1011.rst, AMDGPUAsmGFX1013.rst and
AMDGPUAsmGFX1030.rst from llvm-project tag llvmorg-15.0.7, llvm/docs/AMDGPU/ (Apache-2.0 WITH
LLVM-exception). These are generated from LLVM's instruction tables. The x64 list comes from the
iced-x86 package already in requirements-dev.txt. Output lines are "group<TAB>form".
"""
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "coverage"
LLVM_SOURCE = "llvm-project llvmorg-15.0.7 llvm/docs/AMDGPU/"

# Xbox Series CPUs are AMD Zen 2. These iced CPUID groups are what Zen 2 runs in user mode; the group
# names are what the dashboard shows. AVX-512, XOP/FMA4/TBM, 3DNow!, AMX and vendor extras are left out.
ZEN2 = {
    "Integer": "INTEL8086 INTEL186 INTEL286 INTEL386 INTEL486 X64 CMOV CX8 CMPXCHG16B MOVBE POPCNT LZCNT "
               "MULTIBYTENOP PAUSE TSC RDTSCP CPUID RDRAND RDSEED RDPID SYSCALL PREFETCHW CLFSH CLFLUSHOPT "
               "CLWB CLZERO MONITORX FSGSBASE RDPRU",
    "BMI/ADX": "BMI1 BMI2 ADX",
    "x87": "FPU FPU287 FPU387",
    "MMX": "MMX",
    "SSE": "SSE FXSR",
    "SSE2": "SSE2",
    "SSE3-4": "SSE3 SSSE3 SSE4_1 SSE4_2 SSE4A",
    "Crypto": "AES PCLMULQDQ SHA",
    "AVX": "AVX F16C",
    "AVX2": "AVX2",
    "FMA": "FMA",
    "XSAVE": "XSAVE XSAVEOPT XSAVEC XSAVES",
}


def llvm_forms(*files):
    forms, section = set(), None
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if i + 1 < len(lines) and re.fullmatch(r"-{2,}", lines[i + 1]) and re.fullmatch(r"[A-Z0-9_]+", line):
                section = line
            elif line.startswith(".. toctree::"):
                section = None  # the operand-page index at the end isn't instructions
            elif section and (m := re.match(r"    ([a-z][a-z0-9_]+)(\s|$)", line)):
                forms.add((section, m.group(1)))
    # Guard against filing forms under the wrong section heading.
    expect = {"DPP8": "_dpp$", "DPP16": "_dpp$", "SDWA": "_sdwa$", "DS": "^ds_", "SMEM": "^s_", "MIMG": "^image_"}
    bad = [f for f in forms if f[0] in expect and not re.search(expect[f[0]], f[1])]
    assert not bad, bad[:5]
    return forms


def x64_forms():
    import iced_x86 as iced

    def names(module):
        return {v: k for k, v in vars(module).items() if not k.startswith("_") and isinstance(v, int)}

    group_of = {feature: group for group, features in ZEN2.items() for feature in features.split()}
    feature_name, mnemonic_name = names(iced.CpuidFeature), names(iced.Mnemonic)
    forms = {}
    for code in sorted(names(iced.Code)):
        info = iced.OpCodeInfo(code)
        if not info.is_instruction or not info.mode64 or info.is_privileged:
            continue
        instruction = iced.Instruction()
        instruction.code = code
        groups = [group_of.get(feature_name[f]) for f in instruction.cpuid_features()]
        if None not in groups:
            forms.setdefault(mnemonic_name[info.mnemonic].lower(), groups[0])
    return {(group, mnemonic) for mnemonic, group in forms.items()}


def write(name, source, forms):
    OUT.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{g}\t{f}\n" for g, f in sorted(forms))
    (OUT / name).write_text(f"# {len(forms)} forms. Source: {source}. Regenerate: tools/form_reference.py\n"
                            + body, encoding="utf-8")
    print(name, len(forms))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    docs = Path(sys.argv[1])
    write("gfx1013.tsv", LLVM_SOURCE + "AMDGPUAsmGFX10/1011/1013.rst",
          llvm_forms(*(docs / f"AMDGPUAsmGFX{v}.rst" for v in ("10", "1011", "1013"))))
    write("rdna2.tsv", LLVM_SOURCE + "AMDGPUAsmGFX1030.rst", llvm_forms(docs / "AMDGPUAsmGFX1030.rst"))
    write("x64_zen2.tsv", "iced-x86 1.21.0 mnemonics, Zen 2 user-mode CPUID groups", x64_forms())
