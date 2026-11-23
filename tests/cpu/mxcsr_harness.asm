; Native-only harness for tests/cpu/mxcsr.py; it is never translated. get_mxcsr returns the calling thread's
; MXCSR and set_mxcsr(word) loads it, both through the caller's home space.
.code

get_mxcsr PROC FRAME
    .endprolog
    stmxcsr dword ptr [rsp+8]
    mov eax, dword ptr [rsp+8]
    ret
get_mxcsr ENDP

set_mxcsr PROC FRAME
    .endprolog
    mov dword ptr [rsp+8], ecx
    ldmxcsr dword ptr [rsp+8]
    ret
set_mxcsr ENDP

get_mxcsr_mask PROC FRAME
    sub rsp, 520
    .allocstack 520
    .endprolog
    fxsave [rsp]
    mov eax, dword ptr [rsp+28]
    add rsp, 520
    ret
get_mxcsr_mask ENDP

END
