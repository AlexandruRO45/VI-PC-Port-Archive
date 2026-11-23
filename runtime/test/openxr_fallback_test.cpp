// LEONIDA_VR=openxr with no OpenXR runtime (built with -DLEONIDA_OPENXR=ON; docs/vr.md). CTest points XR_RUNTIME_JSON
// at a file that does not exist, so the OpenXR loader finds no runtime whatever is installed on the machine, and no
// headset is needed. Checks:
//   1. OpenXrSession::create() returns null and VR is reported unavailable once, however often it is asked.
//   2. A device from create_vulkan_device_standalone() and a window presenter on it work as without VR: a frame is
//      presented. Without a Vulkan device this half is left out (the test still passes on the first half).
#include "../shims/d3d12_com.h"
#include "../shims/openxr_session.h"

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

void pump() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

// Run by hand, the test sets what CTest sets: VR on, and a runtime manifest that is not there.
void ensure_env(const char* name, const std::string& value) {
    char* current = nullptr;
    std::size_t length = 0;
    const bool set = _dupenv_s(&current, &length, name) == 0 && current && *current;
    std::free(current);
    if (!set) _putenv_s(name, value.c_str());
}

}  // namespace

int main() {
    char temp[MAX_PATH] = {};
    GetTempPathA(MAX_PATH, temp);
    ensure_env("LEONIDA_VR", "openxr");
    ensure_env("XR_RUNTIME_JSON", std::string(temp) + "leonida_no_openxr_runtime.json");

    // 1. No runtime: null, reported once.
    expect(rt::vr::OpenXrSession::unavailable_reports() == 0, "nothing reported before the first try");
    expect(!rt::vr::OpenXrSession::create(), "no runtime: no session");
    expect(rt::vr::OpenXrSession::unavailable_reports() == 1, "no runtime: reported once");
    expect(!rt::vr::OpenXrSession::create(), "second try: no session");
    expect(rt::vr::OpenXrSession::unavailable_reports() == 1, "second try: not reported again");
    _putenv_s("LEONIDA_VR", "0");
    expect(!rt::vr::OpenXrSession::create(), "LEONIDA_VR=0: no session");
    _putenv_s("LEONIDA_VR", "openxr");

    // 2. The device and a window presenter carry on without VR.
    ID3D12Device* device = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device))) || !device) {
        std::printf("no Vulkan device: the device half is left out\n");
    } else {
        expect(rt::vr::OpenXrSession::unavailable_reports() == 1, "device: still reported once");
        WNDCLASSW window_class{};
        window_class.lpfnWndProc = window_proc;
        window_class.hInstance = GetModuleHandleW(nullptr);
        window_class.lpszClassName = L"LeonidaOpenXrFallbackTest";
        RegisterClassW(&window_class);
        HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida: VR fallback", WS_OVERLAPPEDWINDOW, 100,
                                      100, 320, 240, nullptr, nullptr, window_class.hInstance, nullptr);
        expect(window != nullptr, "window");
        ShowWindow(window, SW_SHOWNOACTIVATE);
        pump();
        auto presenter = rt::d3d12::create_window_presenter(device, window);
        if (!presenter) {
            std::printf("no window presenter on this device: the present is left out\n");
        } else {
            expect(SUCCEEDED(presenter->present_blank(1)), "a frame is presented without VR");
            expect(presenter->presented() == 1, "the window took the frame");
            presenter.reset();
        }
        device->Release();
        DestroyWindow(window);
        pump();
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("openxr_fallback: VR unavailable without a runtime, reported once; the window works as without VR\n");
    return 0;
}
