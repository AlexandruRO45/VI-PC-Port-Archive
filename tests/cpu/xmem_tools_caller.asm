EXTERN __imp_XMemIsToolsMemAvailable:QWORD
EXTERN __imp_xmem_tools_note:QWORD
EXTERN __imp_xmem_tools_note_value:QWORD
EXTERN __imp_xmem_tools_calls:QWORD

.code
native_entry PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    call qword ptr [__imp_XMemIsToolsMemAvailable]
    mov dword ptr [rsp+32], eax
    test eax, eax
    jz fallback
    mov ecx, 11h
    jmp note_path
fallback:
    mov ecx, 22h
note_path:
    call qword ptr [__imp_xmem_tools_note]
    call qword ptr [__imp_xmem_tools_note_value]
    mov dword ptr [rsp+40], eax
    call qword ptr [__imp_xmem_tools_calls]
    mov dword ptr [rsp+48], eax

    mov eax, 7A00h
    cmp dword ptr [rsp+32], 0
    je outcome_false
    or eax, 100h
outcome_false:
    cmp dword ptr [rsp+40], 11h
    je note_tools
    cmp dword ptr [rsp+40], 22h
    je note_fallback
    or eax, 30h
    jmp note_done
note_tools:
    or eax, 10h
    jmp note_done
note_fallback:
    or eax, 20h
note_done:
    cmp dword ptr [rsp+48], 1
    je calls_one
    or eax, 2
    jmp done
calls_one:
    or eax, 1
done:
    add rsp, 38h
    ret
native_entry ENDP
END