; #3137: s_flbit_i32_b32 / s_flbit_i32 of a lane mask (vcc_lo, exec_lo) read the whole wave's mask word.
; RDNA2 ISA: a lane mask is one bit per lane, so a scalar op that reads vcc_lo or exec_lo as a number sees bit L for lane L
; and gives the same answer in every lane. The translator holds a mask as this lane's 0 / ~0, so before the fix each lane
; saw only its own bit (lane 0, which stores, saw 0 when it was outside the mask and ~0 when inside).
;   s_flbit_i32_b32: D = number of zeros above the highest set bit, -1 when the value is 0.
;   s_flbit_i32:     D = number of bits above the first bit that differs from the sign bit, -1 when there is none.
; v0 = lane (LocalInvocationId.x, 0..31). Lane 0 stores eight words:
;   out[0] flbit_b32 of vcc_lo = lane 12 only (0x00001000)    = 31 - 12 = 19
;   out[1] flbit_i32 of the same mask                         = 19 (sign bit 0, first 1 is bit 12)
;   out[2] flbit_b32 of vcc_lo = lanes 20-31 (0xFFF00000)     = 0  (bit 31 set)
;   out[3] flbit_i32 of the same mask                         = 12 (bits 31..20 are 1, bit 19 is 0)
;   out[4] flbit_i32 of vcc_lo = lanes 0 and 31 (0x80000001)  = 1  (bit 31 is 1, bit 30 is 0)
;   out[5] flbit_b32 of exec_lo = lane 12 only (0x00001000)   = 19
;   out[6] flbit_i32 of exec_lo = lane 12 only                = 19
;   out[7] flbit_b32 of vcc_lo = lanes 0 and 31 (0x80000001)  = 0
; Before the fix lane 0 saw only its own bit (0 or ~0) and stored 0xffffffff for out[0..3] (outside those masks), out[4]
; (inside 0x80000001: flbit_i32(~0) = -1), out[5] and out[6] (outside, off in exec), and 0 for out[7] (flbit_b32(~0)).
; group: 32,1,1
; Run: VKC_SUBGROUP_SIZE=32 vk_compute_test cs_w32_flbit_mask.spv 1 dump -> out[0..7] = 19 19 0 12 1 19 19 0
; mattr: +wavefrontsize32,-wavefrontsize64
s_mov_b32 exec_lo, -1
v_cmp_eq_u32_e32 vcc_lo, 12, v0
s_flbit_i32_b32 s6, vcc_lo
s_flbit_i32 s7, vcc_lo
v_cmp_lt_u32_e32 vcc_lo, 19, v0
s_flbit_i32_b32 s8, vcc_lo
s_flbit_i32 s9, vcc_lo
v_xor_b32_e32 v1, 31, v0
v_min_u32_e32 v2, v0, v1
v_cmp_eq_u32_e32 vcc_lo, 0, v2
s_flbit_i32 s10, vcc_lo
s_flbit_i32_b32 s13, vcc_lo
v_cmpx_eq_u32_e32 12, v0
s_flbit_i32_b32 s11, exec_lo
s_flbit_i32 s12, exec_lo
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v20, s6
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s7
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s8
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s9
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s10
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s11
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s12
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s13
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
