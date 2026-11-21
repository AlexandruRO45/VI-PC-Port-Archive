; Source-owned differential fixture for forward string moves and stores (MOVS and STOS in
; byte, word, doubleword and quadword sizes, with and without REP) after CLD.
; Each export takes (dst, src_or_value, count), runs one string instruction with RDI = dst,
; RSI = src_or_value, RAX = src_or_value and RCX = count, and returns
; ((final RCX << 20) + (final RSI - RSI on entry)) << 20 + (final RDI - dst),
; so the element count, both pointer advances and the counter are observed as well as memory.
; RDI and RSI are nonvolatile; they are kept in R10 and R11 and restored before RET.
.code

STRING_ENTRY MACRO
    mov r10, rdi
    mov r11, rsi
    mov r9, rcx
    mov rdi, rcx
    mov rsi, rdx
    mov rax, rdx
    mov rcx, r8
    cld
ENDM

STRING_RESULT MACRO
    mov rax, rcx
    shl rax, 20
    sub rsi, rdx
    add rax, rsi
    shl rax, 20
    sub rdi, r9
    add rax, rdi
    mov rdi, r10
    mov rsi, r11
    ret
ENDM

rep_movsb PROC FRAME
    .endprolog
    STRING_ENTRY
    rep movsb
    STRING_RESULT
rep_movsb ENDP

rep_movsw PROC FRAME
    .endprolog
    STRING_ENTRY
    rep movsw
    STRING_RESULT
rep_movsw ENDP

rep_movsd PROC FRAME
    .endprolog
    STRING_ENTRY
    rep movsd
    STRING_RESULT
rep_movsd ENDP

rep_movsq PROC FRAME
    .endprolog
    STRING_ENTRY
    rep movsq
    STRING_RESULT
rep_movsq ENDP

rep_stosb PROC FRAME
    .endprolog
    STRING_ENTRY
    rep stosb
    STRING_RESULT
rep_stosb ENDP

rep_stosw PROC FRAME
    .endprolog
    STRING_ENTRY
    rep stosw
    STRING_RESULT
rep_stosw ENDP

rep_stosd PROC FRAME
    .endprolog
    STRING_ENTRY
    rep stosd
    STRING_RESULT
rep_stosd ENDP

rep_stosq PROC FRAME
    .endprolog
    STRING_ENTRY
    rep stosq
    STRING_RESULT
rep_stosq ENDP

; Without REP one element moves and RCX is left unchanged.
movsq_once PROC FRAME
    .endprolog
    STRING_ENTRY
    movsq
    STRING_RESULT
movsq_once ENDP

stosd_once PROC FRAME
    .endprolog
    STRING_ENTRY
    stosd
    STRING_RESULT
stosd_once ENDP

END
