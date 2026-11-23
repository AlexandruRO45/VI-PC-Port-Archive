; Compressed color exports (docs/design.md, "Partial pixel color exports"): a `compr` export takes two channels from
; each of its first and third operands as f16 halves, and its EN field enables the RG pair and the BA pair rather
; than single channels. After the full export leaves (1, 2, 3, 4), the first compressed export enables BA only and
; writes the pair (7, 8) there, and the second enables RG only and writes (9, 10). Every pixel ends up with
; (9, 10, 7, 8): RG from the last export, BA still the pair the first compressed one wrote.
; Run: vk_harness vs_tri.spv ps_partial_exports_compr.spv out.raw
v_mov_b32_e32 v0, 0x3f800000      ; 1.0
v_mov_b32_e32 v1, 0x40000000      ; 2.0
v_mov_b32_e32 v2, 0x40400000      ; 3.0
v_mov_b32_e32 v3, 0x40800000      ; 4.0
v_mov_b32_e32 v4, 0x48004700      ; f16 pair: 7.0 low, 8.0 high
v_mov_b32_e32 v5, 0x49004880      ; f16 pair: 9.0 low, 10.0 high
exp mrt0 v0, v1, v2, v3
exp mrt0 off, off, v4, v4 compr
exp mrt0 v5, v5, off, off done compr
s_endpgm
