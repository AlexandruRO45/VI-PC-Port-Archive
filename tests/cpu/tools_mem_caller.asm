; A source-built EXE that imports XMemIsToolsMemAvailable by name from xmem.dll (#870) and starts
; through exe_startup.asm, so the same body runs in a normal Windows process and from pe_map_run.
; Synthetic caller; it does not reproduce any title's startup code. Steps:
;    1  the first call returns exactly FALSE (0) or TRUE (1)
;    2  a second call returns the same value
;    3  a third call returns the same value
; Then the caller takes one of two branches and records it as the exit code:
;    7100h  FALSE: continue without tools memory (desktop without a tools partition)
;    7101h  TRUE:  take the tools-memory branch
; A failed step returns 0F00h + the step number.
EXTERN __imp_XMemIsToolsMemAvailable:QWORD

NO_TOOLS_BRANCH EQU 7100h
TOOLS_BRANCH    EQU 7101h

.code
native_entry PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 40
    .allocstack 40
    .endprolog

    mov ebx, 1                      ; step 1
    call QWORD PTR __imp_XMemIsToolsMemAvailable
    cmp eax, 1
    ja failed
    mov esi, eax
    inc ebx                         ; step 2
    call QWORD PTR __imp_XMemIsToolsMemAvailable
    cmp eax, esi
    jne failed
    inc ebx                         ; step 3
    call QWORD PTR __imp_XMemIsToolsMemAvailable
    cmp eax, esi
    jne failed

    mov eax, NO_TOOLS_BRANCH
    test esi, esi
    jz finished
    mov eax, TOOLS_BRANCH
    jmp finished
failed:
    lea eax, [rbx+0F00h]
finished:
    add rsp, 40
    pop rsi
    pop rbx
    ret
native_entry ENDP
END
