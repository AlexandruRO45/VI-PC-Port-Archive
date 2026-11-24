; Source-built native caller. rcx=output, rdx=descriptor, r8=allocator, r9=host.
.code
PUBLIC create_body, call_provider, call_allocate, call_calloc, wrapper_oom, create_done
fixture_create PROC FRAME
    push rsi
    .pushreg rsi
    push rdi
    .pushreg rdi
    push rbx
    .pushreg rbx
    push rbp
    .pushreg rbp
    sub rsp, 40
    .allocstack 40
    .endprolog
create_body::
    test rcx, rcx
    jz invalid_args
    test rdx, rdx
    jz invalid_args
    mov rsi, rcx
    mov rbx, r8
    mov rdi, r9
    mov qword ptr [rsi], 0
    mov eax, [rdx]
    and eax, 0ff0000h
    cmp eax, 10000h
    jne unknown_effect
    mov rbp, [rdi]
    test rbp, rbp
    jz unknown_effect
    mov qword ptr [rsp+32], 0
    lea rcx, [rsp+32]
call_provider::
    call qword ptr [rbp]
    test eax, eax
    jne create_done
    mov edx, 16
    test rbx, rbx
    jz use_default
    mov rcx, [rbx]
call_allocate::
    call qword ptr [rbx+8]
    jmp wrap
use_default:
    mov ecx, 1
call_calloc::
    call qword ptr [rdi+8]
wrap:
    test rax, rax
    jz wrapper_oom
    mov [rax], rbp
    mov rdx, [rsp+32]
    mov [rax+8], rdx
    mov [rsi], rax
    xor eax, eax
    jmp create_done
unknown_effect:
    mov eax, 2
    jmp create_done
invalid_args:
    mov eax, 6
    jmp create_done
wrapper_oom::
    lea rcx, [rsp+32]
    mov rdx, rbx
    call qword ptr [rbp+8]
    mov eax, 5
create_done::
    add rsp, 40
    pop rbp
    pop rbx
    pop rdi
    pop rsi
    ret
fixture_create ENDP
END
