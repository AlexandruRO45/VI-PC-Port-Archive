// Tiny fixture EXE built with /GA, the retail executable's form of __declspec(thread): every access reads
// TEB->ThreadLocalStoragePointer[0] (gs:[58h]) instead of loading _tls_index. It is never executed by the test;
// rt::Loader maps it and the test calls the exports, including from a thread that existed before load().
#include <cstdint>
#include <intrin.h>

__declspec(thread) int tiny_ga_tls_value = 7;

// The array pointer the patched instruction reads, without dereferencing it: on a thread the loader gave no /GA
// view this is what a real /GA access would dereference, so the test can report it instead of faulting.
extern "C" __declspec(dllexport) uintptr_t __cdecl tiny_ga_tls_vector() {
    return static_cast<uintptr_t>(__readgsqword(0x58));
}

// A real /GA TLS access: vector[0] plus the variable's offset in the block.
extern "C" __declspec(dllexport) uintptr_t __cdecl tiny_ga_tls_address() {
    return reinterpret_cast<uintptr_t>(&tiny_ga_tls_value);
}

int main() { return 0; }
