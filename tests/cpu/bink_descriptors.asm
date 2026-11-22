.code
group_init PROC
    lea rax, [rcx+64]
    mov [rcx+8], rax
    mov [rcx], rax
    mov r8d, edx
    lea rax, [rax+r8*8]
    mov [rcx+24], rax
    mov [rcx+16], rax
    mov [rcx+44], edx
    mov dword ptr [rcx+40], 0
    mov qword ptr [rcx+32], 0
    ret
group_init ENDP

group_append PROC
    mov eax, [rcx+40]
    cmp eax, [rcx+44]
    jne available
    int 3
available:
    ; Work in bytes. Pick the first rounded stride with the required low ten bits.
    add r8, 31
    and r8, -32
    mov r9, [rcx+32]
    add r9, 32
    and r9, 1023
    mov r10, r8
    and r10, -1024
    or r10, r9
    cmp r10, r8
    jae have_stride
    add r10, 1024
have_stride:
    mov eax, [rcx+40]
    mov r11, [rcx+8]
    mov [r11+rax*8], rdx
    mov r11, [rcx+24]
    mov [r11+rax*8], r10
    add [rcx+32], r10
    inc dword ptr [rcx+40]
    mov eax, 1
    ret
group_append ENDP
; RCX = frame table, EDX = frame index, R8 = pitch pair, R9 = request observer.
plane_request MACRO field, pitch_offset, height_offset
LOCAL skip_plane
    mov rcx, [rsp+32]
    cmp dword ptr [rcx+field+8], 0
    je skip_plane
    mov r10, [rsp+40]
    mov edx, [r10+pitch_offset]
    mov r10, [rsp+48]
    imul edx, dword ptr [r10+height_offset]
    add rcx, field
    mov r9, [rsp+56]
    call r9
    mov rcx, [rsp+32]
    mov r10, [rsp+40]
    mov eax, [r10+pitch_offset]
    mov [rcx+field+12], eax
skip_plane:
ENDM

frame_planes PROC FRAME
    sub rsp, 72
    .allocstack 72
    .endprolog
    mov [rsp+48], rcx
    imul edx, edx, 80
    lea rax, [rcx+rdx+24]
    mov [rsp+32], rax
    mov [rsp+40], r8
    mov [rsp+56], r9
    plane_request 0, 0, 8
    plane_request 16, 4, 16
    plane_request 32, 4, 16
    plane_request 48, 0, 8
    plane_request 64, 0, 8
    add rsp, 72
    ret
frame_planes ENDP
END
