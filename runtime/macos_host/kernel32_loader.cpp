// The kernel32 calls Leonida's own runtime (gta6rt.exe) makes on top of the CRT's, to load and run a game
// image inside the process: environment, modules, memory queries, function tables.
#include "host.h"

#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace host {
using HANDLE = void*;
namespace {

uint32_t GetEnvironmentVariableW(const char16_t* name, char16_t* buf, uint32_t n) {
    const char* v = std::getenv(narrow(name).c_str());
    if (!v) return set_last_error(203 /* ERROR_ENVVAR_NOT_FOUND */), 0;
    std::u16string w = widen(v);
    uint32_t len = static_cast<uint32_t>(w.size());
    if (n <= len) return len + 1;
    std::memcpy(buf, w.c_str(), (len + 1) * 2);
    return len;
}

int FlushInstructionCache(HANDLE, const void*, uint64_t) { return 1; }  // Rosetta watches code pages itself
int DisableThreadLibraryCalls(HANDLE) { return 1; }
int TlsFree(uint32_t) { return 1; }  // ponytail: slots are never reused
int DeleteFileW(const char16_t* path) {
    if (unlink(to_host_path(narrow(path).c_str()).c_str()) == 0) return 1;
    return set_last_error(errno == ENOENT ? 2 : 5), 0;
}
int DeleteFileA(const char* path) {
    if (unlink(to_host_path(path).c_str()) == 0) return 1;
    return set_last_error(errno == ENOENT ? 2 : 5), 0;
}

}  // namespace

#define K(name) EXPORT("kernel32.dll", name)
Exports kernel32_loader_exports() {
    return {K(GetEnvironmentVariableW), K(FlushInstructionCache), K(DisableThreadLibraryCalls), K(TlsFree), K(DeleteFileW), K(DeleteFileA)};
}

}  // namespace host
