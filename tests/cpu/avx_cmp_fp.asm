; coverage: source cpu_aot_avx_cmp_fp
; Source-owned differential fixture for VEX AVX FP compare instructions (vcmpps, vcmppd, vcmpss, vcmpsd).
.code

CMP_XMM_CASE MACRO prefix, insn, p
prefix&_p&p& PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    insn xmm0, xmm1, xmm2, p
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
prefix&_p&p& ENDP
ENDM

CMP_YMM_CASE MACRO prefix, insn, p
prefix&_p&p& PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    insn ymm0, ymm1, ymm2, p
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
prefix&_p&p& ENDP
ENDM

CMP_ALL32_XMM MACRO prefix, insn
    CMP_XMM_CASE prefix, insn, 0
    CMP_XMM_CASE prefix, insn, 1
    CMP_XMM_CASE prefix, insn, 2
    CMP_XMM_CASE prefix, insn, 3
    CMP_XMM_CASE prefix, insn, 4
    CMP_XMM_CASE prefix, insn, 5
    CMP_XMM_CASE prefix, insn, 6
    CMP_XMM_CASE prefix, insn, 7
    CMP_XMM_CASE prefix, insn, 8
    CMP_XMM_CASE prefix, insn, 9
    CMP_XMM_CASE prefix, insn, 10
    CMP_XMM_CASE prefix, insn, 11
    CMP_XMM_CASE prefix, insn, 12
    CMP_XMM_CASE prefix, insn, 13
    CMP_XMM_CASE prefix, insn, 14
    CMP_XMM_CASE prefix, insn, 15
    CMP_XMM_CASE prefix, insn, 16
    CMP_XMM_CASE prefix, insn, 17
    CMP_XMM_CASE prefix, insn, 18
    CMP_XMM_CASE prefix, insn, 19
    CMP_XMM_CASE prefix, insn, 20
    CMP_XMM_CASE prefix, insn, 21
    CMP_XMM_CASE prefix, insn, 22
    CMP_XMM_CASE prefix, insn, 23
    CMP_XMM_CASE prefix, insn, 24
    CMP_XMM_CASE prefix, insn, 25
    CMP_XMM_CASE prefix, insn, 26
    CMP_XMM_CASE prefix, insn, 27
    CMP_XMM_CASE prefix, insn, 28
    CMP_XMM_CASE prefix, insn, 29
    CMP_XMM_CASE prefix, insn, 30
    CMP_XMM_CASE prefix, insn, 31
ENDM

CMP_ALL32_YMM MACRO prefix, insn
    CMP_YMM_CASE prefix, insn, 0
    CMP_YMM_CASE prefix, insn, 1
    CMP_YMM_CASE prefix, insn, 2
    CMP_YMM_CASE prefix, insn, 3
    CMP_YMM_CASE prefix, insn, 4
    CMP_YMM_CASE prefix, insn, 5
    CMP_YMM_CASE prefix, insn, 6
    CMP_YMM_CASE prefix, insn, 7
    CMP_YMM_CASE prefix, insn, 8
    CMP_YMM_CASE prefix, insn, 9
    CMP_YMM_CASE prefix, insn, 10
    CMP_YMM_CASE prefix, insn, 11
    CMP_YMM_CASE prefix, insn, 12
    CMP_YMM_CASE prefix, insn, 13
    CMP_YMM_CASE prefix, insn, 14
    CMP_YMM_CASE prefix, insn, 15
    CMP_YMM_CASE prefix, insn, 16
    CMP_YMM_CASE prefix, insn, 17
    CMP_YMM_CASE prefix, insn, 18
    CMP_YMM_CASE prefix, insn, 19
    CMP_YMM_CASE prefix, insn, 20
    CMP_YMM_CASE prefix, insn, 21
    CMP_YMM_CASE prefix, insn, 22
    CMP_YMM_CASE prefix, insn, 23
    CMP_YMM_CASE prefix, insn, 24
    CMP_YMM_CASE prefix, insn, 25
    CMP_YMM_CASE prefix, insn, 26
    CMP_YMM_CASE prefix, insn, 27
    CMP_YMM_CASE prefix, insn, 28
    CMP_YMM_CASE prefix, insn, 29
    CMP_YMM_CASE prefix, insn, 30
    CMP_YMM_CASE prefix, insn, 31
ENDM

CMP_ALL32_XMM vcmpps_xmm, vcmpps
CMP_ALL32_YMM vcmpps_ymm, vcmpps
CMP_ALL32_XMM vcmppd_xmm, vcmppd
CMP_ALL32_YMM vcmppd_ymm, vcmppd
CMP_ALL32_XMM vcmpss, vcmpss
CMP_ALL32_XMM vcmpsd, vcmpsd

vcmpps_xmm_mem_q_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vcmpps xmm0, xmm1, xmmword ptr [rdx], 0ch
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_xmm_mem_q_op ENDP

vcmpps_xmm_mem_s_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vcmpps xmm0, xmm1, xmmword ptr [rdx], 01h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_xmm_mem_s_op ENDP

vcmpps_ymm_mem_q_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcmpps ymm0, ymm1, ymmword ptr [rdx], 1eh
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_ymm_mem_q_op ENDP

vcmpps_ymm_mem_s_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcmpps ymm0, ymm1, ymmword ptr [rdx], 02h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_ymm_mem_s_op ENDP

vcmppd_xmm_mem_q_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vcmppd xmm0, xmm1, xmmword ptr [rdx], 00h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmppd_xmm_mem_q_op ENDP

vcmppd_xmm_mem_s_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vcmppd xmm0, xmm1, xmmword ptr [rdx], 10h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmppd_xmm_mem_s_op ENDP

vcmppd_ymm_mem_q_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcmppd ymm0, ymm1, ymmword ptr [rdx], 1dh
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmppd_ymm_mem_q_op ENDP

vcmppd_ymm_mem_s_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcmppd ymm0, ymm1, ymmword ptr [rdx], 0dh
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmppd_ymm_mem_s_op ENDP

vcmpss_mem_q_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vcmpss xmm0, xmm1, dword ptr [rdx], 11h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpss_mem_q_op ENDP

vcmpss_mem_s_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vcmpss xmm0, xmm1, dword ptr [rdx], 05h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpss_mem_s_op ENDP

vcmpsd_mem_q_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vcmpsd xmm0, xmm1, qword ptr [rdx], 04h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpsd_mem_q_op ENDP

vcmpsd_mem_s_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vcmpsd xmm0, xmm1, qword ptr [rdx], 1bh
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpsd_mem_s_op ENDP

vcmpps_ymm_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vcmpps ymm0, ymm0, ymm1, 01h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_ymm_alias_dst_src1_op ENDP

vcmpps_ymm_alias_dst_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vcmpps ymm0, ymm1, ymm0, 02h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_ymm_alias_dst_src2_op ENDP

vcmpps_ymm_alias_src1_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcmpps ymm0, ymm1, ymm1, 00h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_ymm_alias_src1_src2_op ENDP

vcmpps_ymm_alias_all_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vcmpps ymm0, ymm0, ymm0, 04h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_ymm_alias_all_op ENDP

vcmpps_xmm_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vcmpps xmm0, xmm0, xmm1, 01h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_xmm_alias_dst_src1_op ENDP

vcmpps_xmm_alias_dst_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vcmpps xmm0, xmm1, xmm0, 02h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_xmm_alias_dst_src2_op ENDP

vcmppd_xmm_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vcmppd xmm0, xmm0, xmm1, 0ch
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmppd_xmm_alias_dst_src1_op ENDP

vcmpss_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vcmpss xmm0, xmm0, xmm1, 01h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpss_alias_dst_src1_op ENDP

vcmpss_alias_dst_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vcmpss xmm0, xmm1, xmm0, 02h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpss_alias_dst_src2_op ENDP

vcmpss_alias_src1_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcmpss xmm0, xmm1, xmm1, 00h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpss_alias_src1_src2_op ENDP

vcmpsd_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vcmpsd xmm0, xmm0, xmm1, 0eh
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpsd_alias_dst_src1_op ENDP

vcmpsd_alias_dst_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vcmpsd xmm0, xmm1, xmm0, 11h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpsd_alias_dst_src2_op ENDP

vcmpps_xmm_imm_highbits_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vcmpps xmm0, xmm1, xmm2, 0e5h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_xmm_imm_highbits_op ENDP

END
