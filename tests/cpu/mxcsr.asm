; coverage: source cpu_aot_mxcsr
; Source-owned differential fixture for LDMXCSR, STMXCSR and their VEX.128 forms VLDMXCSR and VSTMXCSR in the opt-in
; host-MXCSR profile. Each export takes (a, b, c, state) in RCX, RDX, R8 and R9; b is the requested MXCSR word.
; State layout (bytes):
;   00 caller word (STMXCSR at entry, LDMXCSR at exit)   04 guard
;   08 requested word                                     0C guard
;   10, 20 the ADDPS operands, 30 and 40 the CVTSS2SI operands (lane 0), all written by the test at run time
;   50 the ADDPS result                                   60 first CVTSS2SI result, 64 guard
;   68 second CVTSS2SI result, 6C guard                   70 STMXCSR read-back, 74 guard
;   78 the marker that mxcsr_after and mxcsr_vex_after write after their load; in the VEX and mixed rounding
;      exports the word read after the final restore     80-FF guards
; mxcsr_round restores the caller's word itself, and so do the three rounding exports of the VEX family. The
; others do not: there only the translated entry's boundary restores it, and the test calls them translated only.
; mxcsr_vex_* use the VEX forms for every MXCSR access, mxcsr_mix_lv and mxcsr_mix_vl use one legacy and one VEX
; form per pair. The four mxcsr_bad_* exports hold a VLDMXCSR or VSTMXCSR whose second byte differs by one bit from
; a valid encoding (VEX.L = 1, or vvvv = 1110); they are never run natively and only decoded and lowered.
; The shared arithmetic of the rounding exports: ADDPS and two CVTSS2SI, which round by the loaded MXCSR.
MXCSR_WORK MACRO
    movups xmm0, xmmword ptr [r9+10h]
    movups xmm1, xmmword ptr [r9+20h]
    addps xmm0, xmm1
    movups xmmword ptr [r9+50h], xmm0
    movups xmm2, xmmword ptr [r9+30h]
    cvtss2si eax, xmm2
    mov dword ptr [r9+60h], eax
    movups xmm2, xmmword ptr [r9+40h]
    cvtss2si eax, xmm2
    mov dword ptr [r9+68h], eax
ENDM

.code

mxcsr_round PROC FRAME
    .endprolog
    stmxcsr dword ptr [r9]
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    movups xmm0, xmmword ptr [r9+10h]
    movups xmm1, xmmword ptr [r9+20h]
    addps xmm0, xmm1
    movups xmmword ptr [r9+50h], xmm0
    movups xmm2, xmmword ptr [r9+30h]
    cvtss2si eax, xmm2
    mov dword ptr [r9+60h], eax
    movups xmm2, xmmword ptr [r9+40h]
    cvtss2si eax, xmm2
    mov dword ptr [r9+68h], eax
    stmxcsr dword ptr [r9+70h]
    ldmxcsr dword ptr [r9]
    mov eax, 7
    ret
mxcsr_round ENDP

; Loads the requested word and returns without restoring the caller's.
mxcsr_leak PROC FRAME
    .endprolog
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    mov eax, 7
    ret
mxcsr_leak ENDP

; Guest work after the load: the marker store must not run when the word is refused.
mxcsr_after PROC FRAME
    .endprolog
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    mov qword ptr [r9+78h], 1
    mov eax, 7
    ret
mxcsr_after ENDP

; LDMXCSR of an operand the test places on an inaccessible page: the guest read faults before any load.
mxcsr_badload PROC FRAME
    .endprolog
    ldmxcsr dword ptr [rcx]
    mov eax, 7
    ret
mxcsr_badload ENDP

; A guest load through RCX after a valid word was loaded; the test points RCX at an inaccessible page.
mxcsr_fault PROC FRAME
    .endprolog
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    mov eax, dword ptr [rcx]
    ret
mxcsr_fault ENDP


; Every MXCSR access in VEX form: entry store, load, read-back, restore and a read after the restore.
mxcsr_vex_round PROC FRAME
    .endprolog
    vstmxcsr dword ptr [r9]
    mov dword ptr [r9+8], edx
    vldmxcsr dword ptr [r9+8]
    MXCSR_WORK
    vstmxcsr dword ptr [r9+70h]
    vldmxcsr dword ptr [r9]
    vstmxcsr dword ptr [r9+78h]
    mov eax, 7
    ret
mxcsr_vex_round ENDP

; Legacy entry store and restore around a VEX load and VEX read-back.
mxcsr_mix_lv PROC FRAME
    .endprolog
    stmxcsr dword ptr [r9]
    mov dword ptr [r9+8], edx
    vldmxcsr dword ptr [r9+8]
    MXCSR_WORK
    vstmxcsr dword ptr [r9+70h]
    ldmxcsr dword ptr [r9]
    stmxcsr dword ptr [r9+78h]
    mov eax, 7
    ret
mxcsr_mix_lv ENDP

; VEX entry store and restore around a legacy load and legacy read-back.
mxcsr_mix_vl PROC FRAME
    .endprolog
    vstmxcsr dword ptr [r9]
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    MXCSR_WORK
    stmxcsr dword ptr [r9+70h]
    vldmxcsr dword ptr [r9]
    stmxcsr dword ptr [r9+78h]
    mov eax, 7
    ret
mxcsr_mix_vl ENDP

; Loads the requested word with VLDMXCSR and returns without restoring the caller's.
mxcsr_vex_leak PROC FRAME
    .endprolog
    mov dword ptr [r9+8], edx
    vldmxcsr dword ptr [r9+8]
    mov eax, 7
    ret
mxcsr_vex_leak ENDP

; Guest work after the VEX load: the marker store must not run when the word is refused.
mxcsr_vex_after PROC FRAME
    .endprolog
    mov dword ptr [r9+8], edx
    vldmxcsr dword ptr [r9+8]
    mov qword ptr [r9+78h], 1
    mov eax, 7
    ret
mxcsr_vex_after ENDP

; VLDMXCSR of an operand the test places on an inaccessible page: the guest read faults before any load.
mxcsr_vex_badload PROC FRAME
    .endprolog
    vldmxcsr dword ptr [rcx]
    mov eax, 7
    ret
mxcsr_vex_badload ENDP

; Malformed encodings, written as bytes (C5 payload AE modrm, modrm 11 = [rcx], 19 = [rcx] for the store). The
; valid payload is F8: R = 1, vvvv = 1111, L = 0, pp = 00. F8 -> FC sets L; F8 -> F0 makes vvvv 1110.
mxcsr_bad_vld_l1 PROC FRAME
    .endprolog
    db 0C5h, 0FCh, 0AEh, 011h
    ret
mxcsr_bad_vld_l1 ENDP

mxcsr_bad_vld_vvvv PROC FRAME
    .endprolog
    db 0C5h, 0F0h, 0AEh, 011h
    ret
mxcsr_bad_vld_vvvv ENDP

mxcsr_bad_vst_l1 PROC FRAME
    .endprolog
    db 0C5h, 0FCh, 0AEh, 019h
    ret
mxcsr_bad_vst_l1 ENDP

mxcsr_bad_vst_vvvv PROC FRAME
    .endprolog
    db 0C5h, 0F0h, 0AEh, 019h
    ret
mxcsr_bad_vst_vvvv ENDP

END
