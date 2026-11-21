; A CRT-free PE entry. The existing call_mix (direct.asm, IMPORT_MIX)
; calls the named DLL import and records the two actual arguments and its output.
EXTERN call_mix:PROC
EXTERN __imp_read_selector:QWORD
EXTERN __imp_emit_trace:QWORD

.code
native_entry PROC FRAME
    sub rsp, 72
    .allocstack 72
    .endprolog
    call qword ptr [__imp_read_selector]
    cmp eax, -1
    je io_error
    test eax, eax
IFDEF FLIP_SELECTOR
    jnz wide_input
ELSE
    jz wide_input
ENDIF
    mov ecx, 41
    mov edx, 9
    jmp dispatch
wide_input:
    mov rcx, 1000002fch
    mov rdx, 100000000h
dispatch:
    lea r8, [rsp+32]
    call call_mix
    mov [rsp+56], rax
    lea rcx, [rsp+32]
    call qword ptr [__imp_emit_trace]
    test eax, eax
    jz io_error
    mov eax, [rsp+56]
    jmp finished
io_error:
    mov eax, 2
finished:
    add rsp, 72
    ret
native_entry ENDP
END
