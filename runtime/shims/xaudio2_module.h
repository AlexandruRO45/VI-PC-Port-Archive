#pragma once

#include <windows.h>

namespace rt::shims::xaudio2::detail {

struct ModuleLoad {
    HMODULE module;
    DWORD error;
};

template <typename Loader>
ModuleLoad load_module(Loader&& loader) {
    HMODULE module = loader();
    return {module, module ? ERROR_SUCCESS : GetLastError()};
}

}  // namespace rt::shims::xaudio2::detail
