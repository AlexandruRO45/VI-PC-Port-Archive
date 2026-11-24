"""tools/rdna2_mnemonics.py : seconde source, les définitions gfx10 de VOP3PInstructions.td (LLVM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rdna2_mnemonics  # noqa: E402

TD_EXCERPT = """
multiclass VOP3P_Real_gfx10<bits<8> op> {
defm V_PK_MAD_I16     : VOP3P_Real_gfx10_gfx11_gfx12_gfx13<0x00>;
defm V_PK_ADD_F16     : VOP3P_Real_gfx10_gfx11_gfx12_gfx13<0x0f>;
defm V_DOT2_I32_I16 : VOP3P_Real_gfx10 <0x14>;
defm V_FMA_MIX_F32    : VOP3P_Real_gfx10_gfx11_gfx12_gfx13_Triple<0x20>;
defm V_PK_ADD_BF16    : VOP3P_Real_gfx12<0x23>;
"""


def test_td_lines_give_gfx10_vop3p_opcodes(tmp_path):
    (tmp_path / "VOP3PInstructions.td").write_text(TD_EXCERPT, encoding="utf-8")
    assert rdna2_mnemonics.td_mnemonics(tmp_path) == {
        ("VOP3P", 0x00): "v_pk_mad_i16", ("VOP3P", 0x0F): "v_pk_add_f16",
        ("VOP3P", 0x14): "v_dot2_i32_i16", ("VOP3P", 0x20): "v_fma_mix_f32"}


def test_encodings_win_over_td_and_disagreements_are_reported():
    table = {("VOP3P", 0x0F): "v_pk_add_f16"}
    td = {("VOP3P", 0x0F): "v_pk_add_f16", ("VOP3P", 0x10): "v_pk_mul_f16", ("VOP3P", 0x00): "autre"}
    table[("VOP3P", 0x00)] = "v_pk_mad_i16"
    disagreements = rdna2_mnemonics.merge_td(table, td)
    assert table[("VOP3P", 0x10)] == "v_pk_mul_f16" and table[("VOP3P", 0x00)] == "v_pk_mad_i16"
    assert disagreements == [(("VOP3P", 0x00), "v_pk_mad_i16", "autre")]
