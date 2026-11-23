"""The #5533 pair and its standalone driver reproducer (TEST-1296, tools/driver_repro/amd_pk_ashr_hi16).

AMD's Windows driver 32.0.21045.29003 (gfx1036) writes 0xb88d4000 for ``v_pk_ashrrev_i16 v29, v30, v24
op_sel:[1,1] op_sel_hi:[1,1]`` after ``v_med3_f16 ... op_sel:[1,1,1,1]`` set v24 = 0x4000b88d, where the answer is
0x40004000. These tests pin the CPU side: the translator's module computes 0x40004000, so the difference is the
driver's; and the two reproducer modules in the folder assemble, validate, and keep the spelling they document.
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

import spirv_interp as si
from conftest import HAVE_SPIRV
from listing import translate_listing

REPRO = Path(__file__).resolve().parent.parent / "tools" / "driver_repro" / "amd_pk_ashr_hi16"
_spec = importlib.util.spec_from_file_location("amd_pk_ashr_hi16_modules", REPRO / "modules.py")
modules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(modules)

# valu_fuzz --family f16 --show 10:55:32, cut to the registers and the two instructions that matter (#5533).
ISSUE_5533 = "\n".join([
    "v_mov_b32 v20, 0xbe279724",
    "v_mov_b32 v24, 0xcf1db88d",
    "v_mov_b32 v27, 0x4000e7b4",
    "v_mov_b32 v30, 0xbe204b94",
    "v_mov_b32 v34, 0x4000bd60",
    "v_mov_b32 v7, 0",
    "v_or_b32 v20, 0x3c00, v7",
    "v_med3_f16_e64 v24, v20, v27, v34 op_sel:[1,1,1,1]",
    "v_pk_ashrrev_i16 v29, v30, v24 op_sel:[1,1] op_sel_hi:[1,1]",
])


def test_issue_5533_pair_translates_to_the_model_value(tmp_path):
    """med3 of the high halves (0x0000, 0x4000, 0x4000) is 0x4000 and lands in v24's high half; both packed-shift
    result halves are hi(v24) >> (hi(v30) & 15) = 0x4000 >> 0."""
    state = si.run(translate_listing(tmp_path, ISSUE_5533))
    assert state.vgpr[24] == 0x4000B88D
    assert state.vgpr[29] == 0x40004000


def test_modules_script_writes_both_modules(tmp_path):
    """`python -I modules.py <dir>` writes each module's text unchanged (and its .spv when spirv-as is present)."""
    r = subprocess.run([sys.executable, "-I", str(REPRO / "modules.py"), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(modules.MODULES) == ["pk_ashr_hi16", "pk_ashr_hi16_in_place"]
    for name, text in modules.MODULES.items():
        assert (tmp_path / f"{name}.spvasm").read_text(encoding="utf-8") == text


@pytest.mark.skipif(not HAVE_SPIRV, reason="needs spirv-as and spirv-val on PATH")
@pytest.mark.parametrize("name", ["pk_ashr_hi16", "pk_ashr_hi16_in_place"])
def test_repro_modules_assemble_and_validate(tmp_path, name):
    src = tmp_path / f"{name}.spvasm"
    out = tmp_path / f"{name}.spv"
    src.write_text(modules.MODULES[name], encoding="utf-8", newline="\n")
    for cmd in (["spirv-as", "--target-env", "vulkan1.2", str(src), "-o", str(out)],
                ["spirv-val", "--target-env", "vulkan1.2", str(out)]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


def _body(name):
    """The module's instructions, comments dropped and whitespace collapsed."""
    return [" ".join(ln.split(";")[0].split()) for ln in modules.MODULES[name].splitlines()]


def test_repro_keeps_the_translator_spelling():
    """The failing module extracts the high half, shifts it by a non-constant amount, masks it and moves it back up,
    as rdna2spirv.py's packed-shift lowering does; the control computes the high half in place."""
    failing = "\n".join(_body("pk_ashr_hi16"))
    assert re.search(r"%val1 = OpBitFieldSExtract %uint %b24 %u16 %u16\b", failing)
    assert re.search(r"%sh1 = OpShiftRightArithmetic %uint %val1 %amt1\b", failing)
    assert re.search(r"%hi = OpShiftLeftLogical %uint %hi16 %u16\b", failing)
    assert re.search(r"%s1 = OpSelect %uint %l1 %init24 %u256\b", failing)
    control = "\n".join(_body("pk_ashr_hi16_in_place"))
    assert re.search(r"%val1 = OpBitwiseAnd %uint %b24 %uhi\b", control)
    assert re.search(r"%hi = OpBitwiseAnd %uint %sh1 %uhi\b", control)
    assert "%hi16" not in control
    # The rest of the two modules is the same, so the control isolates the high-half spelling.
    assert set(failing.splitlines()) - set(control.splitlines()) == {
        "%val1 = OpBitFieldSExtract %uint %b24 %u16 %u16", "%hi16 = OpBitwiseAnd %uint %sh1 %uffff",
        "%hi = OpShiftLeftLogical %uint %hi16 %u16"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
