; An SV_Target that the PS output signature (OSG1) declares uint: the export's bits are the value (EGEN-PS-MRT-TYPES).
; A compressed export of a uint target carries two 16-bit integers per register; RDNA2 writes them to a 16-bit
; integer target bit for bit (UINT16_ABGR export format, measured with tools/ps_oracle.py --format rgba16u on gfx1036:
; 0x80017fff, 0xc0003c01 came back as those words). Drawn into an RGBA32_UINT target, each half arrives
; zero-extended. Until this test the translator decoded the halves as f16 into a float output.
;   v2 = 0x80017fff: x = 0x7fff, y = 0x8001     v3 = 0xc0003c01: z = 0x3c01, w = 0xc000
; Expected mrt0 bits: 0x00007fff, 0x00008001, 0x00003c01, 0x0000c000
v_mov_b32 v2, 0x80017fff
v_mov_b32 v3, 0xc0003c01
exp mrt0 v2, v2, v3, v3 done compr vm
s_endpgm
