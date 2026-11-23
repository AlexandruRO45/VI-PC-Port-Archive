; Wave32 v_writelane_b32 of a v_readlane_b32 result writes the named lane only. Fourth shape of the NVIDIA
; select-after-shuffle miscompile (RTX 5080, Linux 580.65.06): select(lane == 0, shuffle(v4, 7), v6) gave the shuffle
; to every lane. Reduced from crosslane_fuzz.py seed 2031 (cs_xl32_203106417, --dpp --ext --no-maskdata).
; docs/design.md "Write gate of a subgroup result". Only that driver miscompiles it: a pass elsewhere proves nothing.
; v1 = lane, v4 = 60 * lane + 0x9b7557d9, v6 = 24 * lane + 0x996d7497. s13 = v4[7] = 0x9b75597d goes to lane 0 of v6;
; s11 = v6 of the first lane = s13. Lanes 0-3 store v6 and s11.
; out[2 * L] = L ? 24 * L + 0x996d7497 : 0x9b75597d, out[2 * L + 1] = 0x9b75597d for L = 0..3 (with the miscompile
; every v6 word is 0x9b75597d).
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_readlane_writelane.spv 1 dump
;   -> out[0..7] = 0x9b75597d 0x9b75597d 0x996d74af 0x9b75597d 0x996d74c7 0x9b75597d 0x996d74df 0x9b75597d
; mattr: -wavefrontsize64,+wavefrontsize32
; group: 32,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mad_u32_u24 v4, v1, 60, 0x9b7557d9
v_mad_u32_u24 v6, v1, 24, 0x996d7497
s_mov_b32 s28, 7
v_readlane_b32 s13, v4, s28
v_writelane_b32 v6, s13, 0
v_readfirstlane_b32 s11, v6
v_cmpx_gt_u32_e32 4, v1
v_lshlrev_b32_e32 v21, 3, v1
v_mov_b32_e32 v7, s11
buffer_store_dword v6, v21, s[0:3], 0 offen
buffer_store_dword v7, v21, s[0:3], 0 offen offset:4
s_endpgm
