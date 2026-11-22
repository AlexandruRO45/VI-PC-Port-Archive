; An SV_Target that the PS output signature (OSG1) declares sint (EGEN-PS-MRT-TYPES). A compressed export carries
; two 16-bit integers per register (SINT16_ABGR export format: the halves reach a 16-bit integer target bit for bit,
; measured with tools/ps_oracle.py --format rgba16i on gfx1036). Drawn into an RGBA32_SINT target, each half arrives
; sign-extended. Until this test the translator decoded the halves as f16 into a float output.
;   v2 = 0x80017fff: x = 0x7fff (32767), y = 0x8001 (-32767)    v3 = 0xc0003c01: z = 0x3c01 (15361), w = 0xc000 (-16384)
; Expected mrt0 bits: 0x00007fff, 0xffff8001, 0x00003c01, 0xffffc000
v_mov_b32 v2, 0x80017fff
v_mov_b32 v3, 0xc0003c01
exp mrt0 v2, v2, v3, v3 done compr vm
s_endpgm
