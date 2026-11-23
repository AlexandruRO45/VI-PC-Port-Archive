; group: 32,1,1
; X17: a loop that never exits must still end when it is translated with --loop-budget (run_gpu_tests.py translates this
; fixture with --loop-budget 65536, LOOP_BUDGET_FIXTURES; without the flag the driver hangs and resets the device).
; Lane 0 reads the output buffer's untouched word out[3] (0xeeeeeeee) at run time, so no compiler can fold the loop, and
; makes it odd. s9 then steps 0, 2, 4, ... and the loop leaves only when s9 equals that odd number: never. Each trip
; stores s9 to out[0]. The budget ends the loop after 65,536 trips, so out[0] = 2 * 65536; whatever follows the loop is
; skipped, so out[1..7] keep the sentinel.
; Run: vk_compute_test cs_loop_budget.spv 1 dump
; Expect: out[0..7] = 131072 then seven 4008636142
s_buffer_load_dword s11, s[0:3], 0xc
s_waitcnt lgkmcnt(0)
s_or_b32 s11, s11, 1
s_mov_b32 s9, 0
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 0
s_add_u32 s9, s9, 2
v_mov_b32_e32 v1, s9
buffer_store_dword v1, v2, s[0:3], 0 offen
s_cmp_lg_u32 s9, s11
s_cbranch_scc1 -6
v_mov_b32_e32 v1, 7
v_mov_b32_e32 v2, 4
buffer_store_dword v1, v2, s[0:3], 0 offen
s_endpgm
