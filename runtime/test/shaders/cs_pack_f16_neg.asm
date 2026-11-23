; group: 1,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; v_pack_b32_f16 with neg: flips bit 15 of the selected 16-bit half (IEEE mode 0).
; v0 = 0x44003C00 (low=1.0=0x3C00, high=4.0=0x4400)
; v1 = 0xBC007C00 (low=+inf=0x7C00, high=-1.0=0xBC00)
; Run: vk_compute_test cs_pack_f16_neg.spv 1 dump
; out[0..7] = 0x7C00BC00 0xFC003C00 0x7C00C400 0x3C003C00 0x3C00C400 0xFC00BC00 0x7C003C00 0xBC00C400 (hex)
v_mov_b32_e32 v0, 0x44003C00
v_mov_b32_e32 v1, 0xBC007C00
; 0: -v0, v1: src0 low half neg -> 0xBC00, src1 low = 0x7C00
v_pack_b32_f16 v2, -v0, v1
; 1: v0, -v1: src0 low = 0x3C00, src1 low half neg -> 0xFC00
v_pack_b32_f16 v3, v0, -v1
; 2: -v0, v1 op_sel:[1,0,0]: src0 high half neg -> 0xC400, src1 low = 0x7C00
v_pack_b32_f16 v4, -v0, v1 op_sel:[1,0,0]
; 3: v0, -v1 op_sel:[0,1,0]: src0 low = 0x3C00, src1 high half neg -> 0x3C00
v_pack_b32_f16 v5, v0, -v1 op_sel:[0,1,0]
; 4: -v0, -v1 op_sel:[1,1,0]: src0 high half neg -> 0xC400, src1 high half neg -> 0x3C00
v_pack_b32_f16 v6, -v0, -v1 op_sel:[1,1,0]
; 5: -v0, -v1: src0 low half neg -> 0xBC00, src1 low half neg -> 0xFC00
v_pack_b32_f16 v7, -v0, -v1
; 6: v0, v1 (no neg, control) -> 0x7C003C00
v_pack_b32_f16 v8, v0, v1
; 7: -v0, v1 op_sel:[1,1,0]: src0 high half neg -> 0xC400, src1 high half = 0xBC00
v_pack_b32_f16 v9, -v0, v1 op_sel:[1,1,0]
v_mov_b32_e32 v10, 0
buffer_store_dword v2, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 4
buffer_store_dword v3, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 8
buffer_store_dword v4, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 12
buffer_store_dword v5, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 16
buffer_store_dword v6, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 20
buffer_store_dword v7, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 24
buffer_store_dword v8, v10, s[0:3], 0 offen
v_mov_b32_e32 v10, 28
buffer_store_dword v9, v10, s[0:3], 0 offen
s_endpgm
