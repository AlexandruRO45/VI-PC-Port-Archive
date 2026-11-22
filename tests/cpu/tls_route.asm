; A DLL with a TLS callback and an entry point. Both report each call to sink_record in
; cpu_tls_sink.dll, which outlives this image, so attach and detach order can be observed.
; The TLS directory has an empty data template; only the callback list is populated.
; The callback list has its own read-only section, so a test can make the list unreadable
; without touching the directory in .data.
EXTERN __imp_sink_record:QWORD
IFDEF FRAME_GUEST
EXTERN __imp_frame_render:QWORD
ENDIF
CALLER_TLS EQU 1
CALLER_ENTRY EQU 2

tlscb SEGMENT READ ALIAS(".tlscb") 'DATA'
callbacks   DQ tls_callback, 0
tlscb ENDS

.data
tls_index DD 0
            DD 0
PUBLIC _tls_used
_tls_used   DQ 0, 0                ; StartAddressOfRawData, EndAddressOfRawData (empty template)
            DQ tls_index           ; AddressOfIndex
            DQ callbacks           ; AddressOfCallBacks
            DD 0, 0                ; SizeOfZeroFill, Characteristics

.code
; Both routines receive (module, reason, reserved) and forward them with a caller tag.
tls_callback PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov r9d, CALLER_TLS
    call QWORD PTR [__imp_sink_record]
    add rsp, 28h
    ret
tls_callback ENDP

dll_entry PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov r9d, CALLER_ENTRY
    call QWORD PTR [__imp_sink_record]
IFDEF ENTRY_FAILS
    xor eax, eax
ELSE
    mov eax, 1
ENDIF
    add rsp, 28h
    ret
dll_entry ENDP

; Keeps the image exporting something callable after initialization.
tls_index_address PROC
    lea rax, tls_index
    ret
tls_index_address ENDP

IFDEF FRAME_GUEST
guest_draw PROC
    mov rdx, rcx
    mov ecx, 2
    jmp QWORD PTR [__imp_frame_render]
guest_draw ENDP
ENDIF
END
