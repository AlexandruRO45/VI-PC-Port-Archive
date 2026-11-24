; guarded_load(address, mode, trace[2]) with an exception handler in its own unwind data.
; mode 0 loads from address; an access violation at that load is caught by on_fault, which resumes
; at recover with RAX set to FAULT_VALUE. mode 1 executes ud2; on_fault declines it, so the
; exception unwinds through guarded_load to the caller.
FAULT_VALUE EQU 0fa017fa017fa017h
.code
on_fault PROC
    ; on_fault(ExceptionRecord, EstablisherFrame, ContextRecord, DispatcherContext)
    test DWORD PTR [rcx+4], 66h          ; EXCEPTION_UNWINDING | EXIT_UNWIND | TARGET_UNWIND | COLLIDED
    jnz decline
    cmp DWORD PTR [rcx], 0c0000005h      ; EXCEPTION_ACCESS_VIOLATION
    jne decline
    lea rax, load                        ; resume only a fault at the load itself, so a bad trace
    cmp [rcx+10h], rax                   ; pointer cannot loop through recover (ExceptionAddress)
    jne decline
    lea rax, recover
    mov [r8+0f8h], rax                   ; CONTEXT.Rip
    mov rax, FAULT_VALUE
    mov [r8+78h], rax                    ; CONTEXT.Rax
    xor eax, eax                         ; ExceptionContinueExecution
    ret
decline:
    mov eax, 1                           ; ExceptionContinueSearch
    ret
on_fault ENDP

guarded_load PROC FRAME:on_fault
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rbx, rcx
    mov rsi, rdx
    mov [r8], rbx
    test rsi, rsi
    jz load
    ud2
load::
    mov rax, [rbx]
recover::
    mov [r8+8], rax
    add rsp, 28h
    pop rsi
    pop rbx
    ret
guarded_load ENDP
END
