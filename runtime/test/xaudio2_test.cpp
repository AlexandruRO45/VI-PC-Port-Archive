#include <windows.h>
#include <xaudio2.h>

#include <cstdio>
#include <string>

#include "shims.h"
#include "xaudio2_module.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

void test_factory(const char* dll) {
    const std::string name = std::string(dll) + "!XAudio2Create resolves";
    auto raw = rt::shims::find(dll, "XAudio2Create");
    check(raw != nullptr, name.c_str());
    const std::string module_name = std::string(dll) + " is served by the shim registry for run-time loads";
    check(rt::shims::owns_dll(dll), module_name.c_str());
    if (!raw) return;

    using CreateFn = HRESULT(WINAPI*)(IXAudio2**, UINT32, XAUDIO2_PROCESSOR);
    auto create = reinterpret_cast<CreateFn>(raw);
    IXAudio2* engine = nullptr;
    const HRESULT hr = create(&engine, 0, XAUDIO2_DEFAULT_PROCESSOR);
    check(SUCCEEDED(hr), "XAudio2Create initializes the host engine without creating an output device");
    check(engine != nullptr, "XAudio2Create returns an engine on success");
    if (engine) engine->Release();
}

void test_load_error_is_cached_with_module_result() {
    SetLastError(ERROR_MOD_NOT_FOUND);
    const auto load = rt::shims::xaudio2::detail::load_module([]() -> HMODULE { return nullptr; });
    SetLastError(ERROR_INVALID_HANDLE);
    check(load.module == nullptr, "failed host load remains represented as a null module");
    check(load.error == ERROR_MOD_NOT_FOUND, "failed host load preserves the original error");
}

}  // namespace

int main() {
    test_load_error_is_cached_with_module_result();
    test_factory("xaudio2_9.dll");
    test_factory("xaudio2_9redist.dll");
    if (g_failures) {
        std::printf("xaudio2_test: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }
    std::printf("xaudio2_test: OK (%d checks)\n", g_checks);
    return 0;
}
