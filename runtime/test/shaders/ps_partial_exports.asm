; Partial color exports (docs/design.md, "Partial pixel color exports"): an `exp` writes only the channels its EN
; field enables, and a channel the export leaves off keeps what the previous export to that target wrote. The first
; export enables R and B, the second G and A, so every pixel ends up with the merge of the two, (1, 2, 3, 4). An
; export that wrote the whole target would leave (0, 2, 0, 4) here.
; Run: vk_harness vs_tri.spv ps_partial_exports.spv out.raw
v_mov_b32_e32 v0, 0x3f800000      ; 1.0
v_mov_b32_e32 v1, 0x40000000      ; 2.0
v_mov_b32_e32 v2, 0x40400000      ; 3.0
v_mov_b32_e32 v3, 0x40800000      ; 4.0
exp mrt0 v0, off, v2, off
exp mrt0 off, v1, off, v3 done
s_endpgm
