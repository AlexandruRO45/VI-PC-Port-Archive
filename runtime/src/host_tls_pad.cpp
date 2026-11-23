// gta6rt.exe is the first module in the process, so it owns static TLS index 0. The title was linked expecting its
// own static TLS at that index: its CRT inlines the read of gs:[58][0] at every thread-local access (7,802 sites in
// 1.0.2.1), so whatever block sits behind index 0 has to hold the title's template. The loader therefore copies the
// title's template to the base of this program's block (pe_loader.cpp, mirror_index0).
//
// This pad reserves the room for that copy in front of the runtime's own TLS variables. MSVC's tlssup.obj puts 272
// bytes of empty `.tls` before the `.tls$` contributions, and this pad is the first of those, so the template lands on
// 272 bytes of padding plus this array. Keeping the pad at least as large as the title's template (5,248 bytes) means
// the copy ends before the first runtime variable (`_Init_thread_epoch`, the task queue's WaitTimer state), which the
// loader checks at run time. Nothing reads the pad, so the title overwriting it is harmless.
__declspec(thread) unsigned char g_guest_tls_pad[8192];

unsigned char* guest_tls_pad() { return g_guest_tls_pad; }
unsigned char* guest_tls_pad_end() { return g_guest_tls_pad + sizeof g_guest_tls_pad; }
