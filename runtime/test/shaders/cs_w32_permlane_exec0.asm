; Wave32 v_permlane16_b32 / v_permlanex16_b32 under EXEC = 0 must not write (RDNA2 ISA: VALU writes land in active
; lanes only). Regression guard for an NVIDIA driver miscompile (RTX 5080, Linux 580.65.06): select(active, shuffle,
; old) kept the shuffle in every lane when active was computed before the shuffle and old was itself a select (here:
; the v_writelane_b32 write). Reduced from crosslane_fuzz.py seed 91001 (cs_xl32_9100373491). docs/design.md
; "Write gate of a subgroup result". Only that driver miscompiles it: on other drivers (Windows 616.92, 617.14,
; 621.24, AMD) a translator without the gate passes too, so a pass there proves nothing about the gate.
; v1 = lane, v3 = 16 * lane + 1000 (permlane source), v4 = 51 * lane + 7 (destination), v7 = 46 * lane + 2000.
; v_cmp v7 < v1 is false in every lane, so s_and_saveexec leaves EXEC = 0. v_writelane_b32 writes lane 16 anyway
; (it ignores EXEC); both permlanes must leave v4 alone. EXEC = -1 again, lanes 0-7 store v4.
; out[L] = 51 * L + 7 for L = 0..7 (a translator or driver that lets the permlane write gives 16 * lane' + 1000).
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_permlane_exec0.spv 1 dump
;   -> out[0..7] = 7 58 109 160 211 262 313 364
; mattr: -wavefrontsize64,+wavefrontsize32
; group: 32,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mad_u32_u24 v3, v1, 16, 0x3e8
v_mad_u32_u24 v4, v1, 51, 7
v_mad_u32_u24 v7, v1, 46, 0x7d0
s_mov_b32 s11, 0xb4d677db
v_cmp_lt_u32_e64 s18, v7, v1
s_and_saveexec_b32 s22, s18
s_mov_b32 s28, 16
v_writelane_b32 v4, s11, s28
s_mov_b32 s29, 0x2afaa6f3
v_permlane16_b32 v4, v3, 0x9d823309, s29
v_permlanex16_b32 v4, v3, 0x9d823309, s29
s_mov_b32 exec_lo, -1
v_cmpx_gt_u32_e32 8, v1
v_lshlrev_b32_e32 v21, 2, v1
buffer_store_dword v4, v21, s[0:3], 0 offen
s_endpgm
