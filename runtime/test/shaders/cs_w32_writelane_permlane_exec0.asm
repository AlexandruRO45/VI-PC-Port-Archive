; Wave32 v_permlanex16_b32 under EXEC = 0 must not write, when the register's previous write is a v_writelane_b32 and
; an earlier v_permlanex16_b32 to another register ran under the same EXEC. Second shape of the NVIDIA select-after-shuffle
; miscompile (RTX 5080, Linux 580.65.06): with the gate of #4233 (EXEC read back through a ballot after the shuffle, merged with the ballot of the first permlane),
; the shuffle still reached v4 in lanes where EXEC is off. Reduced from crosslane_fuzz.py seed 70102 (cs_xl32_7010410418,
; --no-maskdata --ops 40) to four ops. docs/design.md "Write gate of a subgroup result". Companion of
; cs_w32_permlane_exec0.asm; only that driver miscompiles it, a pass elsewhere proves nothing about the gate.
; v0 = lane, v3 = permlane source (11 * lane + 0xb5bb0a6d), v4 = 52 * lane + 0x7e8b8958, v5 = 8 * lane + 0xf6c106a1.
; v_cmpx_gt_u32 60, v3 is false in every lane (v3 >= 0xb5bb0a6d), so EXEC = 0. v_writelane_b32 writes lane 25
; anyway (it ignores EXEC); both permlanes must leave v4 and v5 alone. EXEC is restored and lanes 0-3 store v4, v5.
; out[2 * L] = 52 * L + 0x7e8b8958, out[2 * L + 1] = 8 * L + 0xf6c106a1 for L = 0..3.
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_writelane_permlane_exec0.spv 1 dump
;   -> out[0..7] = 0x7e8b8958 0xf6c106a1 0x7e8b898c 0xf6c106a9 0x7e8b89c0 0xf6c106b1 0x7e8b89f4 0xf6c106b9
; mattr: -wavefrontsize64,+wavefrontsize32
; group: 32,1,1
s_mov_b32 s30, exec_lo
v_mad_u32_u24 v2, v0, 52, 0x44b0d373
v_mad_u32_u24 v3, v0, 11, 0xb5bb0a6d
v_mad_u32_u24 v4, v0, 52, 0x7e8b8958
v_mad_u32_u24 v5, v0, 8, 0xf6c106a1
s_mov_b32 s8, 0xfa22374e
v_cmpx_gt_u32_e32 60, v3
s_mov_b32 s29, 0x3473dd73
v_permlanex16_b32 v5, v2, 0xd480ad58, s29
s_mov_b32 s28, 25
v_writelane_b32 v4, s8, s28
s_mov_b32 s29, 0xa7171e37
v_permlanex16_b32 v4, v3, 0xbb395e4, s29
s_mov_b32 exec_lo, s30
v_cmpx_gt_u32_e32 4, v0
v_lshlrev_b32_e32 v22, 3, v0
buffer_store_dword v4, v22, s[0:3], 0 offen
buffer_store_dword v5, v22, s[0:3], 0 offen offset:4
s_endpgm
