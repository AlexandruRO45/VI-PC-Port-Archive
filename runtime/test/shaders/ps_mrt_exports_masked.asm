; Mixed masks from #3988: MRT0 exports with every lane valid. After it retires, clear EXEC
; before the MRT1 export, which has no vm. Only MRT0 has a reference here. Report MRT1's
; observed bits separately: an unwritten Vulkan shader output has no defined value.
v_mov_b32 v2, 1.0
v_mov_b32 v3, 2.0
v_mov_b32 v4, 0x40400000
v_mov_b32 v5, 4.0
v_mov_b32 v1, 1
exp mrt0 v2, v3, v4, v5 vm
s_waitcnt expcnt(0)
v_cmpx_eq_u32 0, v1
exp mrt1 v5, v4, v3, v2 done
s_waitcnt expcnt(0)
s_endpgm
