; Native-only harness for tests/cpu/mxcsr.py and tests/cpu/mxcsr_denormals.py; it is never translated. get_mxcsr
; returns the calling thread's MXCSR and set_mxcsr(word) loads it, both through the caller's home space.
; mxcsr_mask_of(area) runs FXSAVE into the 16-byte-aligned 512-byte area and returns its MXCSR_MASK field.
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

mxcsr_mask_of PROC FRAME
    .endprolog
    fxsave64 [rcx]
    mov eax, dword ptr [rcx+28]
    ret
mxcsr_mask_of ENDP

END
