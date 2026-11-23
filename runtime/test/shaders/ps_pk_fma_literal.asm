; Expected MRT0 words: 0x40008c00, 0x42008c00, 0x4200bc00, 0x4400bc00.
; The literal is (lo=3/2, hi=65/64), v0 is (lo=63/64, hi=2).
; Red low: (65/64)*(63/64)-1 = -1/4096 (0x8c00). The product alone is
; 4095/4096, a tie that rounds to 1 in binary16, giving zero if split.
; Red high: (3/2)*2-1 = 2. Reusing the literal's high half gives 33/32.
; Green selects the zero high half of the inline addend: high becomes 3.
; Blue and alpha replace the literal with inline 2.0 (lo=2, hi=0).
; Their low half becomes -1, and their high halves are 2*2-1 and 2*2+0.
v_mov_b32 v0, 0x40003be0
v_pk_fma_f16 v1, 0x3c103e00, v0, -1.0 op_sel:[1,0,0] op_sel_hi:[0,1,0]
v_pk_fma_f16 v2, 0x3c103e00, v0, -1.0 op_sel:[1,0,0] op_sel_hi:[0,1,1]
v_pk_fma_f16 v3, 2.0, v0, -1.0 op_sel:[1,0,0] op_sel_hi:[0,1,0]
v_pk_fma_f16 v4, 2.0, v0, -1.0 op_sel:[1,0,0] op_sel_hi:[0,1,1]
exp mrt0 v1, v2, v3, v4 done vm
s_endpgm
