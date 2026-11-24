; coverage: source cpu_aot_float_conversions
; Source-built fixture for 34 packed and scalar SSE/SSE2/AVX float conversion instructions,
; testing full 256-bit vector state, GPR conversions, and truncation/rounding modes.
.code

cvtdq2ps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvtdq2ps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtdq2ps_op ENDP

vcvtdq2ps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtdq2ps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtdq2ps_op ENDP

cvtdq2pd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvtdq2pd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtdq2pd_op ENDP

vcvtdq2pd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtdq2pd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtdq2pd_op ENDP

cvtps2dq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvtps2dq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtps2dq_op ENDP

vcvtps2dq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtps2dq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtps2dq_op ENDP

cvttps2dq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvttps2dq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvttps2dq_op ENDP

vcvttps2dq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvttps2dq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvttps2dq_op ENDP

cvtps2pd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvtps2pd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtps2pd_op ENDP

vcvtps2pd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtps2pd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtps2pd_op ENDP

cvtpd2ps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvtpd2ps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtpd2ps_op ENDP

vcvtpd2ps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtpd2ps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtpd2ps_op ENDP

cvtpd2dq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvtpd2dq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtpd2dq_op ENDP

vcvtpd2dq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtpd2dq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtpd2dq_op ENDP

cvttpd2dq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvttpd2dq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvttpd2dq_op ENDP

vcvttpd2dq_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvttpd2dq xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvttpd2dq_op ENDP

cvtss2sd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvtss2sd xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtss2sd_op ENDP

vcvtss2sd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtss2sd xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtss2sd_op ENDP

cvtsd2ss_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    cvtsd2ss xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtsd2ss_op ENDP

vcvtsd2ss_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtsd2ss xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtsd2ss_op ENDP

cvtsi2ss_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    cvtsi2ss xmm0, edx
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtsi2ss_op ENDP

cvtsi2ss_qword_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    cvtsi2ss xmm0, rdx
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtsi2ss_qword_op ENDP

vcvtsi2ss_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vcvtsi2ss xmm0, xmm0, edx
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtsi2ss_op ENDP

vcvtsi2ss_qword_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vcvtsi2ss xmm0, xmm0, rdx
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtsi2ss_qword_op ENDP

cvtsi2sd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    cvtsi2sd xmm0, edx
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtsi2sd_op ENDP

cvtsi2sd_qword_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    cvtsi2sd xmm0, rdx
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
cvtsi2sd_qword_op ENDP

vcvtsi2sd_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vcvtsi2sd xmm0, xmm0, edx
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtsi2sd_op ENDP

vcvtsi2sd_qword_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vcvtsi2sd xmm0, xmm0, rdx
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtsi2sd_qword_op ENDP

cvtss2si_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    cvtss2si eax, xmm0
    ret
cvtss2si_op ENDP

cvtss2si_qword_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    cvtss2si rax, xmm0
    ret
cvtss2si_qword_op ENDP

vcvtss2si_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vcvtss2si eax, xmm0
    ret
vcvtss2si_op ENDP

vcvtss2si_qword_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vcvtss2si rax, xmm0
    ret
vcvtss2si_qword_op ENDP

cvttss2si_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    cvttss2si eax, xmm0
    ret
cvttss2si_op ENDP

cvttss2si_qword_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    cvttss2si rax, xmm0
    ret
cvttss2si_qword_op ENDP

vcvttss2si_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vcvttss2si eax, xmm0
    ret
vcvttss2si_op ENDP

vcvttss2si_qword_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vcvttss2si rax, xmm0
    ret
vcvttss2si_qword_op ENDP

cvtsd2si_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    cvtsd2si eax, xmm0
    ret
cvtsd2si_op ENDP

cvtsd2si_qword_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    cvtsd2si rax, xmm0
    ret
cvtsd2si_qword_op ENDP

vcvtsd2si_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vcvtsd2si eax, xmm0
    ret
vcvtsd2si_op ENDP

vcvtsd2si_qword_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vcvtsd2si rax, xmm0
    ret
vcvtsd2si_qword_op ENDP

cvttsd2si_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    cvttsd2si eax, xmm0
    ret
cvttsd2si_op ENDP

cvttsd2si_qword_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    cvttsd2si rax, xmm0
    ret
cvttsd2si_qword_op ENDP

vcvttsd2si_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vcvttsd2si eax, xmm0
    ret
vcvttsd2si_op ENDP

vcvttsd2si_qword_op PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vcvttsd2si rax, xmm0
    ret
vcvttsd2si_qword_op ENDP

vcvtph2ps_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtph2ps xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtph2ps_op ENDP

vcvtps2ph_op PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vcvtps2ph xmm0, xmm1, 0
    vmovdqu ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtps2ph_op ENDP

end
