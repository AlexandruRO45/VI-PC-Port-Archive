// D3D12ConfigureVirtualMemoryX resolves through the shim registry and returns S_OK, the measured
// console ABI being a no-parameter policy call (docs/d3d12_xs_surface.md; SCARAB's bridge, ordinal 4).
// The retail title calls it during D3D12 init, and with no binding the unimplemented policy return made
// it exit before D3D12XboxCreateDevice (gta6rt --run, RT-5). No game data and no GPU involved.
#include <windows.h>

#include <cstdio>
#include <string>

#include "../shims/shims.h"

namespace {

int g_failed = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failed;
        std::fprintf(stderr, "FAIL %s\n", what);
    }
}

}  // namespace

int main() {
    void* fn = rt::shims::find("d3d12_xs.dll", "D3D12ConfigureVirtualMemoryX");
    check(fn != nullptr, "the export resolves through the shim registry");
    if (fn) {
        // The measured call takes no parameters and succeeds; call it three times the way any caller would.
        using VmFn = HRESULT(WINAPI*)();
        const auto configure = reinterpret_cast<VmFn>(fn);
        for (int i = 0; i < 3; ++i)
            check(configure() == S_OK, "D3D12ConfigureVirtualMemoryX returns S_OK");
    }
    // The other d3d12_xs exports stay reachable.
    check(rt::shims::find("d3d12_xs.dll", "D3D12XboxCreateDevice") != nullptr,
          "D3D12XboxCreateDevice still resolves");
    check(rt::shims::find("d3d12_xs.dll", "D3D12SerializeRootSignature") != nullptr,
          "D3D12SerializeRootSignature still resolves");
    if (g_failed) {
        std::fprintf(stderr, "d3d12_xs_vmem_test: %d failure(s)\n", g_failed);
        return 1;
    }
    std::printf("d3d12_xs_vmem_test: OK\n");
    return 0;
}
