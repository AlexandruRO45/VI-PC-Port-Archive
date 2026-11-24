.code
; RCX points to before, after, status and flag. RDX points to the two counters.
delta MACRO entry, flag_width
LOCAL have_sample, primary_only
entry PROC FRAME
    .endprolog
    xor r8d, r8d
    cmp dword ptr [rcx+16], 0
    je have_sample
    mov r8, [rcx+8]
have_sample:
    sub r8, [rcx]
    add [rdx], r8
    cmp flag_width ptr [rcx+20], 0
    je primary_only
    add [rdx+8], r8
primary_only:
    ret
entry ENDP
ENDM
delta create_delta, dword
delta release_delta, byte
END
