; Context contains a table pointer. Slot 1 takes (context, input) and returns a word.
.code
call_context PROC FRAME
    sub rsp, 40
    .allocstack 40
    .endprolog
    test rcx, rcx
    jz no_context
    mov rcx, [rcx]
    test rcx, rcx
    jz no_context
    mov rax, [rcx]
    call qword ptr [rax+8]
    jmp finish
no_context:
    mov eax, 6
finish:
    add rsp, 40
    ret
call_context ENDP
END
