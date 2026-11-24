; coverage: source cpu_aot_xsave_state
; Source-built MASM fixture exercising 4 XSAVE state forms from docs/coverage/x64_zen2.tsv
; group XSAVE: xsave, xsave64, xrstor, and xrstor64, along with transport/ABI glue
; (vmovdqu, vzeroupper, paddd, movq, emms).
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (64-byte aligned; 960+ bytes)
;   RDX = const void* input_b (64-byte aligned; 960+ bytes)
;   R8  = void*       output  (64-byte aligned; 960+ bytes)
.code

; --- Entry-state XSAVE / XSAVE64 (opt-in canonical-entry profile) ---
; Native callers targeting lifted canonical exports must establish canonical XSAVE state
; (FCW=0x037F, FSW=0, FTW=0, FOP=0, FIP=0, FCS=0, FDP=0, FDS=0, MXCSR=0x1F80, zero ST/XMM/YMM_HI, XINUSE=0).

op_xsave_only PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xsave [r8]
    ret
op_xsave_only ENDP

op_xsave64_only PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xsave64 [r8]
    ret
op_xsave64_only ENDP

; --- XRSTOR / XRSTOR64 -> XSAVE / XSAVE64 roundtrips ---

op_xrstor_xsave PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor [rcx]
    xsave [r8]
    ret
op_xrstor_xsave ENDP

op_xrstor64_xsave64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
IFDEF ALTER_OP
    xsave [r8]
ELSE
    xsave64 [r8]
ENDIF
    ret
op_xrstor64_xsave64 ENDP

op_xrstor_xsave64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
IFDEF ALTER_OP
    xrstor64 [rcx]
ELSE
    xrstor [rcx]
ENDIF
    xsave64 [r8]
    ret
op_xrstor_xsave64 ENDP

op_xrstor64_xsave PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    xsave [r8]
    ret
op_xrstor64_xsave ENDP

op_xrstor64_disp PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
IFDEF ALTER_OP
    xrstor [rcx+64]
ELSE
    xrstor64 [rcx+64]
ENDIF
    xsave64 [r8+64]
    ret
op_xrstor64_disp ENDP

op_xrstor64_mask_sse PROC FRAME
    .endprolog
    mov eax, 2
    xor edx, edx
    xrstor64 [rcx]
IFDEF ALTER_OP
    mov eax, 7
ELSE
    mov eax, 2
ENDIF
    xsave64 [r8]
    ret
op_xrstor64_mask_sse ENDP

op_vzeroupper_xsave64 PROC FRAME
    .endprolog
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    vzeroupper
    xsave64 [r8]
    ret
op_vzeroupper_xsave64 ENDP

; --- Chained AVX + SSE + MMX + XSAVE64 pipeline and dead-read fault ordering ---

op_pipeline PROC FRAME
    .endprolog
IFDEF ALTER_OP
    mov r9, QWORD PTR [rdx+8]
ELSE
    mov r9, QWORD PTR [rdx]
ENDIF
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    movq mm0, r9
    movq rax, mm0
    movq mm7, rax
    vmovdqu ymm0, YMMWORD PTR [rcx+832]
    vmovdqu ymm1, ymm0
    paddd xmm1, XMMWORD PTR [rcx+864]
    vmovdqu YMMWORD PTR [r8+832], ymm1
    mov eax, 7
    xsave64 [r8]
    emms
    ret
op_pipeline ENDP

; The discarded xrstor64 [r9] (from rdx) must fault after [r8] and before [r8+832].
op_discarded_read PROC FRAME
    .endprolog
    mov r9, rdx
    mov eax, 7
    xor edx, edx
    xrstor64 [rcx]
    xsave64 [r8]
    xrstor64 [r9]
    vmovdqu ymm0, YMMWORD PTR [rcx+576]
    vmovdqu YMMWORD PTR [r8+832], ymm0
    ret
op_discarded_read ENDP

END
