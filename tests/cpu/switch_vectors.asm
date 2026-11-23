; coverage: record cpu_aot_switch_vectors
; Original Windows x64 fixtures for vector data moves in the control-flow translator: 128- and 256-bit moves
; between vector registers, the frame, guest buffers and image data, the XOR zeroing idiom and VZEROUPPER.
; Every export takes (destination, source, count) in RCX, RDX and R8 and returns an integer in RAX.
; The bad_* exports are refusal controls; the check translates each one alone and expects a stated reason.

.data
ALIGN 16
pattern16 QWORD 0123456789abcdefh, 0fedcba9876543210h
pattern32 QWORD 1111111111111111h, 2222222222222222h, 3333333333333333h, 4444444444444444h

.code
; Legacy SSE moves: unaligned loads, aligned register copies and stores in a different order.
copy_legacy PROC FRAME
    .endprolog
    movups xmm0, [rdx]
    movdqu xmm1, [rdx+16]
    movupd xmm2, [rdx+32]
    movaps xmm3, xmm0
    movdqa xmm4, xmm1
    movapd xmm5, xmm2
    movups [rcx+16], xmm3
    movdqu [rcx+32], xmm4
    movupd [rcx], xmm5
    mov eax, 48
    ret
copy_legacy ENDP

; Legacy aligned memory forms: every operand must be 16-byte aligned.
copy_aligned PROC FRAME
    .endprolog
    movaps xmm0, [rdx]
    movdqa xmm1, [rdx+16]
    movapd xmm2, [rdx+32]
    movaps [rcx], xmm2
    movdqa [rcx+16], xmm0
    movapd [rcx+32], xmm1
    mov eax, 1
    ret
copy_aligned ENDP

; VEX moves of 16 and 32 bytes, with register copies of both widths.
copy_vex PROC FRAME
    .endprolog
    vmovups ymm0, [rdx]
    vmovdqu xmm1, XMMWORD PTR [rdx+32]
    vmovupd ymm2, [rdx+48]
    vmovaps ymm3, ymm2
    vmovapd xmm4, xmm1
    vmovdqa ymm5, ymm0
    vmovups [rcx], ymm3
    vmovdqu YMMWORD PTR [rcx+32], ymm5
    vmovupd [rcx+64], xmm4
    vzeroupper
    mov eax, 80
    ret
copy_vex ENDP

; VEX aligned forms: the XMM operands need 16-byte and the YMM operands 32-byte alignment.
copy_vex_aligned PROC FRAME
    .endprolog
    vmovaps xmm0, [rdx]
    vmovdqa ymm1, YMMWORD PTR [rdx+32]
    vmovapd [rcx], xmm0
    vmovdqa YMMWORD PTR [rcx+32], ymm1
    vzeroupper
    mov eax, 2
    ret
copy_vex_aligned ENDP

; Bytes 16-31: a legacy write keeps them and a VEX write of an XMM register clears them.
upper_halves PROC FRAME
    .endprolog
    vmovups ymm0, [rdx]
    movups xmm0, [rdx+32]
    vmovups [rcx], ymm0
    vmovups ymm1, [rdx]
    vmovups xmm1, [rdx+48]
    vmovups [rcx+32], ymm1
    vmovups ymm2, [rdx]
    movaps xmm2, xmm1
    vmovups [rcx+64], ymm2
    vzeroupper
    xor eax, eax
    ret
upper_halves ENDP

; Zeroing idioms over registers that first hold source data. VXORPD with equal sources zeroes a third register.
zeroing PROC FRAME
    .endprolog
    vmovups ymm0, [rdx]
    vmovups ymm1, [rdx]
    vmovups ymm2, [rdx]
    vmovups ymm3, [rdx]
    vmovups ymm4, [rdx]
    xorps xmm0, xmm0
    vmovups [rcx], ymm0
    pxor xmm1, xmm1
    vmovups [rcx+32], ymm1
    vxorps xmm2, xmm2, xmm2
    vmovups [rcx+64], ymm2
    vpxor ymm3, ymm3, ymm3
    vmovups [rcx+96], ymm3
    vxorpd xmm4, xmm5, xmm5
    vmovups [rcx+128], ymm4
    xorpd xmm5, xmm5
    movups [rcx+160], xmm5
    vzeroupper
    mov eax, 7
    ret
zeroing ENDP

; 16- and 32-byte frame accesses. RSP is entry RSP - 58h, 16-byte aligned, so [rsp+30h] suits MOVAPS.
frame_moves PROC FRAME
    sub rsp, 58h
    .allocstack 58h
    .endprolog
    vmovups ymm0, [rdx]
    vmovups [rsp+8], ymm0
    movups xmm1, [rdx+32]
    movaps [rsp+30h], xmm1
    vmovdqu ymm2, YMMWORD PTR [rsp+8]
    movdqa xmm3, [rsp+30h]
    vmovups [rcx], ymm2
    movups [rcx+32], xmm3
    vzeroupper
    mov eax, 3
    add rsp, 58h
    ret
frame_moves ENDP

; A callee that changes XMM0-XMM5 and their upper halves, as the ABI allows.
clobber PROC FRAME
    .endprolog
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm5, ymm5, ymm5
    xor eax, eax
    ret
clobber ENDP

; XMM6 saved and restored through the frame around a call: its low 16 bytes survive the call.
save_xmm6 PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    vmovaps [rsp+20h], xmm6
    .savexmm128 xmm6, 20h
    .endprolog
    movups xmm6, [rdx]
    movups xmm0, [rdx+16]
    movups [rcx+16], xmm0
    mov [rsp+40h], rcx              ; the caller's home space keeps the destination across the call
    call clobber
    mov rcx, [rsp+40h]
    movups [rcx], xmm6
    vmovaps xmm6, [rsp+20h]
    mov eax, 4
    add rsp, 38h
    ret
save_xmm6 ENDP

; R8 16-byte blocks in a loop. A vector move sits between the compare and its branch.
loop_blocks PROC FRAME
    .endprolog
    xor eax, eax
    test r8, r8
    je done
next:
    movdqu xmm0, [rdx]
    add rdx, 16
    movdqu [rcx], xmm0
    add rcx, 16
    inc rax
    cmp rax, r8
    movaps xmm1, xmm0
    jb next
done:
    ret
loop_blocks ENDP

; RIP-relative constants: an aligned 16-byte and an unaligned 32-byte load.
rip_constants PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR pattern16
    vmovups ymm1, YMMWORD PTR pattern32
    vmovdqa xmm2, XMMWORD PTR pattern16
    movups [rcx], xmm0
    vmovups [rcx+16], ymm1
    vmovdqu XMMWORD PTR [rcx+48], xmm2
    vzeroupper
    mov eax, 5
    ret
rip_constants ENDP

; An RBP-anchored frame: XMM6 is saved and restored through RBP, and a block passes through an RBP slot.
; RBP is entry RSP-8 (0 mod 16) while RSP is entry RSP-30h (8 mod 16), so the aligned accesses are only
; proven aligned from the anchor.
anchored_vectors PROC FRAME
    push rbp
    .pushreg rbp
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rbp, [rsp+28h]
    movaps XMMWORD PTR [rbp-10h], xmm6
    movups xmm6, XMMWORD PTR [rdx]
    movaps XMMWORD PTR [rbp-20h], xmm6
    movdqa xmm0, XMMWORD PTR [rbp-20h]
    movups XMMWORD PTR [rcx], xmm0
    movaps xmm6, XMMWORD PTR [rbp-10h]
    mov eax, 9
    add rsp, 28h
    pop rbp
    ret
anchored_vectors ENDP

; Refusal controls.
bad_undefined PROC FRAME
    .endprolog
    movups [rcx], xmm0
    xor eax, eax
    ret
bad_undefined ENDP

bad_after_call PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    movups xmm0, [rdx]
    mov [rsp+30h], rcx
    call clobber
    mov rcx, [rsp+30h]
    movups [rcx], xmm0
    xor eax, eax
    add rsp, 28h
    ret
bad_after_call ENDP

bad_upper_after_call PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    vmovaps [rsp+20h], xmm6
    .savexmm128 xmm6, 20h
    .endprolog
    vmovups ymm6, [rdx]
    mov [rsp+40h], rcx
    call clobber
    mov rcx, [rsp+40h]
    vmovups [rcx], ymm6
    vmovaps xmm6, [rsp+20h]
    xor eax, eax
    add rsp, 38h
    ret
bad_upper_after_call ENDP

bad_vzeroupper PROC FRAME
    .endprolog
    vmovups ymm0, [rdx]
    vzeroupper
    vmovups [rcx], ymm0
    xor eax, eax
    ret
bad_vzeroupper ENDP

bad_legacy_upper PROC FRAME
    .endprolog
    movups xmm0, [rdx]
    vmovups [rcx], ymm0
    xor eax, eax
    ret
bad_legacy_upper ENDP

bad_unrestored PROC FRAME
    .endprolog
    movups xmm6, [rdx]
    movups [rcx], xmm6
    xor eax, eax
    ret
bad_unrestored ENDP

bad_caller_xmm6 PROC FRAME
    .endprolog
    movups [rcx], xmm6
    xor eax, eax
    ret
bad_caller_xmm6 ENDP

bad_saved_as_data PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    vmovaps [rsp+20h], xmm6
    .savexmm128 xmm6, 20h
    .endprolog
    mov rax, [rsp+20h]
    vmovaps xmm6, [rsp+20h]
    add rsp, 38h
    ret
bad_saved_as_data ENDP

bad_frame_misaligned PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    xorps xmm0, xmm0
    movaps [rsp+28h], xmm0
    xor eax, eax
    add rsp, 38h
    ret
bad_frame_misaligned ENDP

bad_frame_aligned32 PROC FRAME
    sub rsp, 58h
    .allocstack 58h
    .endprolog
    vxorps ymm0, ymm0, ymm0
    vmovaps [rsp+20h], ymm0
    vzeroupper
    xor eax, eax
    add rsp, 58h
    ret
bad_frame_aligned32 ENDP

; [rbp-18h] is entry RSP-20h, misaligned; the same bytes measured from RSP would look aligned.
bad_anchored_misaligned PROC FRAME
    push rbp
    .pushreg rbp
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rbp, [rsp+28h]
    xorps xmm0, xmm0
    movaps XMMWORD PTR [rbp-18h], xmm0
    xor eax, eax
    add rsp, 28h
    pop rbp
    ret
bad_anchored_misaligned ENDP

bad_rip_misaligned PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR pattern16+8
    movups [rcx], xmm0
    xor eax, eax
    ret
bad_rip_misaligned ENDP

bad_xor_logic PROC FRAME
    .endprolog
    movups xmm0, [rdx]
    movups xmm1, [rdx+16]
    xorps xmm0, xmm1
    movups [rcx], xmm0
    xor eax, eax
    ret
bad_xor_logic ENDP

bad_integer_transfer PROC FRAME
    .endprolog
    movq xmm0, rdx
    movups [rcx], xmm0
    xor eax, eax
    ret
bad_integer_transfer ENDP
END
