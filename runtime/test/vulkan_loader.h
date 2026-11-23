// Loads Vulkan at run time. The macOS test tools default to MoltenVK; LEONIDA_VULKAN_LOADER overrides the library
// path or name on every platform.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

inline PFN_vkGetInstanceProcAddr load_vulkan_from(const char* requested) {
    const bool overridden = requested && *requested;
#ifdef _WIN32
    const char* name = overridden ? requested : "vulkan-1.dll";
    HMODULE library = nullptr;
    if (overridden) {
        if (std::strchr(requested, '\\') || std::strchr(requested, '/')) {
            char full_path[MAX_PATH];
            const DWORD len = GetFullPathNameA(requested, MAX_PATH, full_path, nullptr);
            const char* load_path = (len > 0 && len < MAX_PATH) ? full_path : requested;
            library = LoadLibraryExA(load_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        } else {
            library = LoadLibraryExA(requested, nullptr, 0);
        }
    } else {
        library = LoadLibraryExA(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if (!library) {
        std::fprintf(stderr, "could not load %s (error %lu)\n", name, static_cast<unsigned long>(GetLastError()));
        return nullptr;
    }
    const FARPROC symbol = GetProcAddress(library, "vkGetInstanceProcAddr");
    if (!symbol) {
        std::fprintf(stderr, "%s has no vkGetInstanceProcAddr\n", name);
        FreeLibrary(library);
        return nullptr;
    }
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void (*)()>(symbol));
#else
#ifdef __APPLE__
    // MoltenVK compiles MSL with fast-math on by default, which lets Metal fold an f32 compare of a NaN with
    // itself to "ordered" (v_cmpx_eq/le_f32 on (NaN, NaN) give true; ps_vcmpx_forms). Translated shaders need
    // IEEE results, so turn it off before MoltenVK reads its configuration, unless the user set the variable
    // (docs/design.md, "MoltenVK fast-math"). The same variable reaches MoltenVK behind the Vulkan loader.
    setenv("MVK_CONFIG_FAST_MATH_ENABLED", "0", 0);
    // The default is direct MoltenVK: a portability driver would otherwise require instance opt-in.
    constexpr const char* default_primary = "libMoltenVK.dylib";
    constexpr const char* default_fallback = "/opt/homebrew/lib/libMoltenVK.dylib";
#else
    constexpr const char* default_primary = "libvulkan.so.1";
    constexpr const char* default_fallback = "libvulkan.so";
#endif
    const char* name = overridden ? requested : default_primary;
    dlerror();
    void* library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (!library && !overridden) {
        name = default_fallback;
        dlerror();
        library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    }
    if (!library) {
        const char* err = dlerror();
        std::fprintf(stderr, "could not load %s: %s\n", name, err ? err : "unknown error");
        return nullptr;
    }
    void* symbol = dlsym(library, "vkGetInstanceProcAddr");
    if (!symbol) {
        std::fprintf(stderr, "%s has no vkGetInstanceProcAddr\n", name);
        dlclose(library);
        return nullptr;
    }
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(symbol);
#endif
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once, at startup
#endif
inline PFN_vkGetInstanceProcAddr load_vulkan() {
    // Keep the Vulkan library loaded for the lifetime of the tool, and report a missing loader at most once.
    static const PFN_vkGetInstanceProcAddr entry = load_vulkan_from(std::getenv("LEONIDA_VULKAN_LOADER"));
    return entry;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif
