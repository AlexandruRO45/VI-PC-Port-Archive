// Exercise the factory's private data and adapter event allocation without a GPU or a swapchain.
#include "../shims/dxgi_com.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

static thread_local int fail_after = -1;

void* operator new(std::size_t size) {
    if (fail_after >= 0 && fail_after-- == 0) throw std::bad_alloc{};
    if (void* memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

namespace {

using namespace std::chrono_literals;
constexpr GUID kValue{0x5eed0101, 0x1d3d, 0x4c12, {0x9a, 0x41, 0x77, 0x10, 0x2b, 0x6c, 0, 1}};
constexpr GUID kInterface{0x5eed0102, 0x1d3d, 0x4c12, {0x9a, 0x41, 0x77, 0x10, 0x2b, 0x6c, 0, 2}};
constexpr GUID kOther{0x5eed0103, 0x1d3d, 0x4c12, {0x9a, 0x41, 0x77, 0x10, 0x2b, 0x6c, 0, 3}};
constexpr UINT kValueData = 81;

template<class T = IDXGIFactory4>
struct FactoryRef {
    T* ptr = nullptr;
    ~FactoryRef() { if (ptr) ptr->Release(); }
};

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL no_device_proc(VkDevice, const char*) { return nullptr; }

template<class T>
HRESULT create_layer_factory(leonida::VulkanContext& context, FactoryRef<T>& factory) {
    return rt::dxgi::create_factory(context, nullptr, VK_NULL_HANDLE, VK_NULL_HANDLE, nullptr, VK_NULL_HANDLE, 0,
                                    IID_PPV_ARGS(&factory.ptr));
}

HRESULT create_native_factory(FactoryRef<>& factory) {
    using CreateFactory2 = HRESULT(WINAPI*)(UINT, REFIID, void**);
    // The module stays loaded for the lifetime of the short-lived comparison process.
    const HMODULE module = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = module ? reinterpret_cast<CreateFactory2>(
                                     reinterpret_cast<void*>(GetProcAddress(module, "CreateDXGIFactory2"))) : nullptr;
    return create ? create(0, IID_PPV_ARGS(&factory.ptr)) : E_FAIL;
}

struct Payload final : IUnknown {
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown)) return E_NOINTERFACE;
        *out = this;
        AddRef();
        return S_OK;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
        if (owner) {
            UINT size = sizeof(UINT), value = 0;
            const HRESULT read = owner->GetPrivateData(kValue, &size, &value);
            // Replace the very entry the outer call is reading before it acquires its COM reference.
            const HRESULT replace = owner->SetPrivateData(kInterface, sizeof(value), &value);
            if (check_layer_guarantees)
                callback_ok = read == S_OK && size == sizeof(value) && value == kValueData &&
                              replace == S_OK && refs == 1;
        }
        if (entered) {
            entered->set_value();
            const bool removed_in_time = removed.wait_for(3s) == std::future_status::ready;
            if (check_layer_guarantees) callback_ok = removed_in_time && refs == 1;
        }
        return ++refs;
    }

    ULONG STDMETHODCALLTYPE Release() override { return --refs; }

    // Stack lifetime lets the test report an early or leaked Release without dereferencing freed storage.
    std::atomic<ULONG> refs{1};
    IDXGIObject* owner = nullptr;
    std::promise<void>* entered = nullptr;
    std::future<void> removed;
    bool check_layer_guarantees = true;
    bool callback_ok = false;
};

bool reentrant_replace(IDXGIObject& factory, Payload& payload) {
    if (factory.SetPrivateData(kValue, sizeof(kValueData), &kValueData) != S_OK ||
        factory.SetPrivateDataInterface(kInterface, &payload) != S_OK) return false;
    const ULONG refs = payload.Release();
    if (payload.check_layer_guarantees && refs != 1) return false;
    payload.owner = &factory;
    UINT size = sizeof(IUnknown*);
    IUnknown* result = nullptr;
    const HRESULT hr = factory.GetPrivateData(kInterface, &size, &result);
    if (!payload.check_layer_guarantees) {
        if (SUCCEEDED(hr) && result == &payload) result->Release();
        return true;
    }
    const bool read_ok = hr == S_OK && size == sizeof(result) && result == &payload && payload.callback_ok;
    const bool released = result == &payload && result->Release() == 0;
    UINT value = 0;
    size = sizeof(value);
    return read_ok && released && factory.GetPrivateData(kInterface, &size, &value) == S_OK &&
           size == sizeof(value) && value == kValueData;
}

bool concurrent_remove(IDXGIObject& factory, Payload& payload) {
    if (factory.SetPrivateDataInterface(kInterface, &payload) != S_OK) return false;
    const ULONG refs = payload.Release();
    if (payload.check_layer_guarantees && refs != 1) return false;
    std::promise<void> entered, removed;
    auto started = entered.get_future();
    payload.entered = &entered;
    payload.removed = removed.get_future();

    UINT size = sizeof(IUnknown*);
    IUnknown* result = nullptr;
    HRESULT hr = E_FAIL;
    std::thread reader([&] { hr = factory.GetPrivateData(kInterface, &size, &result); });
    const bool started_read = started.wait_for(3s) == std::future_status::ready;
    const bool erased = started_read && factory.SetPrivateDataInterface(kInterface, nullptr) == S_OK;
    const bool retained = !payload.check_layer_guarantees || payload.refs == 1;
    removed.set_value();
    reader.join();

    if (!payload.check_layer_guarantees) {
        if (SUCCEEDED(hr) && result == &payload) result->Release();
        return started_read;
    }
    const bool read_ok = hr == S_OK && size == sizeof(result) && result == &payload && payload.callback_ok;
    const bool released = result == &payload && result->Release() == 0;
    size = sizeof(IUnknown*);
    return erased && retained && read_ok && released &&
           factory.GetPrivateData(kInterface, &size, nullptr) == DXGI_ERROR_NOT_FOUND;
}

// `allocations` counts the operator new calls that succeed before the injected failure. The count follows the
// standard library's allocation order (a debug iterator level adds its own), so a case's name describes the release
// build; every case still requires E_OUTOFMEMORY.
// An interface whose Release reads and writes the store that is dropping it. The store must release a replaced or
// removed reference after unlocking, or this call relocks its std::mutex on the same thread (a hang or an abort).
struct ReleaseProbe final : IUnknown {
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown)) return E_NOINTERFACE;
        *out = this;
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs;
        if (owner && left == 1) {  // the store's reference going away; the test still holds its own
            IDXGIObject* store = std::exchange(owner, nullptr);
            UINT size = sizeof(UINT), value = 0;
            const HRESULT read = store->GetPrivateData(kValue, &size, &value);
            const UINT other = kValueData + 2;
            const HRESULT write = store->SetPrivateData(kOther, sizeof(other), &other);
            callback_ok = read == S_OK && size == sizeof(value) && value == kValueData && write == S_OK;
            callback_ran = true;
        }
        return left;
    }

    std::atomic<ULONG> refs{1};
    IDXGIObject* owner = nullptr;
    bool callback_ran = false;
    bool callback_ok = false;
};

// The entry holding `probe` is replaced by bytes for the same key (or removed); the probe's Release then reenters.
bool release_reentry(IDXGIObject& factory, bool remove) {
    ReleaseProbe probe;
    if (factory.SetPrivateData(kValue, sizeof(kValueData), &kValueData) != S_OK ||
        factory.SetPrivateDataInterface(kInterface, &probe) != S_OK || probe.refs != 2) return false;
    probe.owner = &factory;
    const UINT replacement = kValueData + 1;
    const HRESULT hr = remove ? factory.SetPrivateDataInterface(kInterface, nullptr)
                              : factory.SetPrivateData(kInterface, sizeof(replacement), &replacement);
    UINT size = sizeof(UINT), value = 0;
    const HRESULT after = factory.GetPrivateData(kInterface, &size, &value);
    const bool entry_ok = remove ? after == DXGI_ERROR_NOT_FOUND && size == 0
                                 : after == S_OK && size == sizeof(value) && value == replacement;
    size = sizeof(UINT);
    value = 0;
    const bool other_ok = factory.GetPrivateData(kOther, &size, &value) == S_OK && value == kValueData + 2;
    return hr == S_OK && probe.callback_ran && probe.callback_ok && probe.refs == 1 && entry_ok && other_ok;
}

template<class Call>
bool reports_oom(int allocations, Call call) {
    fail_after = allocations;
    HRESULT hr = E_FAIL;
    try {
        hr = call();
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "bad_alloc escaped the DXGI call\n");
    }
    const bool injected = fail_after < 0;
    fail_after = -1;
    return injected && hr == E_OUTOFMEMORY;
}

bool failed_insert(leonida::VulkanContext& context, int allocations, bool with_interface) {
    Payload payload;
    FactoryRef<> factory;
    if (FAILED(create_layer_factory(context, factory))) return false;
    const bool oom = reports_oom(allocations, [&] {
        return with_interface ? factory.ptr->SetPrivateDataInterface(kInterface, &payload)
                              : factory.ptr->SetPrivateData(kInterface, sizeof(kValueData), &kValueData);
    });
    UINT size = sizeof(IUnknown*);
    const HRESULT lookup = factory.ptr->GetPrivateData(kInterface, &size, nullptr);
    return oom && lookup == DXGI_ERROR_NOT_FOUND && size == 0 && payload.refs == 1;
}

bool failed_replace_interface(leonida::VulkanContext& context, bool with_interface) {
    Payload original, replacement;
    FactoryRef<> factory;
    if (FAILED(create_layer_factory(context, factory)) ||
        factory.ptr->SetPrivateDataInterface(kInterface, &original) != S_OK) return false;
    const bool oom = reports_oom(0, [&] {
        return with_interface ? factory.ptr->SetPrivateDataInterface(kInterface, &replacement)
                              : factory.ptr->SetPrivateData(kInterface, sizeof(kValueData), &kValueData);
    });
    UINT size = sizeof(IUnknown*);
    IUnknown* result = nullptr;
    const HRESULT lookup = factory.ptr->GetPrivateData(kInterface, &size, &result);
    const bool kept = lookup == S_OK && size == sizeof(result) && result == &original;
    const ULONG refs = lookup == S_OK && (result == &original || result == &replacement) ? result->Release() : 0;
    const HRESULT erase = factory.ptr->SetPrivateDataInterface(kInterface, nullptr);
    return oom && kept && refs == 2 && erase == S_OK && original.refs == 1 && replacement.refs == 1;
}

bool failed_replace_bytes(leonida::VulkanContext& context, bool with_interface) {
    Payload replacement;
    FactoryRef<> factory;
    if (FAILED(create_layer_factory(context, factory)) ||
        factory.ptr->SetPrivateData(kValue, sizeof(kValueData), &kValueData) != S_OK) return false;
    const UINT changed = kValueData + 1;
    const bool oom = reports_oom(0, [&] {
        return with_interface ? factory.ptr->SetPrivateDataInterface(kValue, &replacement)
                              : factory.ptr->SetPrivateData(kValue, sizeof(changed), &changed);
    });
    UINT value = 0, size = sizeof(value);
    return oom && replacement.refs == 1 && factory.ptr->GetPrivateData(kValue, &size, &value) == S_OK &&
           size == sizeof(value) && value == kValueData;
}

bool concurrent_updates(leonida::VulkanContext& context) {
    constexpr std::size_t thread_count = 4;
    std::array<Payload, thread_count> values;
    std::atomic<bool> ok{true};
    {
        FactoryRef<> factory;
        if (FAILED(create_layer_factory(context, factory))) return false;
        std::array<std::jthread, thread_count> threads;
        for (std::size_t t = 0; t < threads.size(); ++t) {
            threads[t] = std::jthread([&, t] {
                for (unsigned i = 0; i < 1000; ++i) {
                    if (factory.ptr->SetPrivateDataInterface(kInterface, &values[t]) != S_OK) ok = false;
                    UINT size = sizeof(IUnknown*);
                    IUnknown* value = nullptr;
                    const HRESULT hr = factory.ptr->GetPrivateData(kInterface, &size, &value);
                    if (hr == S_OK) {
                        bool found = false;
                        for (auto& candidate : values) found |= value == &candidate;
                        if (found) value->Release();
                        if (!found || size != sizeof(value)) ok = false;
                    } else if (hr != DXGI_ERROR_NOT_FOUND || size != 0) {
                        ok = false;
                    }
                    if (factory.ptr->SetPrivateDataInterface(kInterface, nullptr) != S_OK) ok = false;
                }
            });
        }
    }
    return ok && std::all_of(values.begin(), values.end(), [](const Payload& value) { return value.refs == 1; });
}

bool failed_adapter_event_registration(leonida::VulkanContext& context) {
    FactoryRef<IDXGIFactory7> factory;
    if (FAILED(create_layer_factory(context, factory))) return false;
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    constexpr DWORD sentinel = 0xDEAD;
    DWORD cookie = sentinel;
    const bool oom = reports_oom(0, [&] { return factory.ptr->RegisterAdaptersChangedEvent(event, &cookie); });
    const bool unchanged = cookie == sentinel;
    const HRESULT retry = factory.ptr->RegisterAdaptersChangedEvent(event, &cookie);
    const HRESULT unregister = retry == S_OK ? factory.ptr->UnregisterAdaptersChangedEvent(cookie) : E_FAIL;
    CloseHandle(event);
    return oom && unchanged && retry == S_OK && cookie == 1 && unregister == S_OK;
}

bool run_child(const wchar_t* scenario, bool native) {
    std::wstring exe(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, exe.data(), DWORD(exe.size()));
    if (!length || length == exe.size()) {
        std::fprintf(stderr, "could not resolve the test executable path\n");
        return false;
    }
    exe.resize(length);
    std::wstring command = L"\"" + exe + L"\" " + scenario + (native ? L" native" : L" layer");
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process)) {
        std::fprintf(stderr, "CreateProcessW failed: %lu\n", GetLastError());
        return false;
    }
    CloseHandle(process.hThread);
    const DWORD wait = WaitForSingleObject(process.hProcess, 10000);
    DWORD status = 1;
    if (wait == WAIT_OBJECT_0) {
        GetExitCodeProcess(process.hProcess, &status);
        if (status != 0) std::fprintf(stderr, "%ls (%s): exit 0x%08lx\n", scenario,
                                      native ? "native" : "layer", status);
    } else {
        status = wait == WAIT_TIMEOUT ? 124 : 1;
        std::fprintf(stderr, "%ls (%s): %s\n", scenario, native ? "native" : "layer",
                     wait == WAIT_TIMEOUT ? "watchdog timed out" : "watchdog wait failed");
        TerminateProcess(process.hProcess, status);
        WaitForSingleObject(process.hProcess, 1000);
    }
    CloseHandle(process.hProcess);
    return status == 0;
}

bool report(const char* scenario, bool ok) {
    std::printf("%s: %s\n", scenario, ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if (argc == 3) {
        const std::wstring_view scenario = argv[1], implementation = argv[2];
        const bool release_case = scenario == L"release-replace" || scenario == L"release-remove";
        if ((scenario != L"reentrant" && scenario != L"concurrent" && !release_case) ||
            (implementation != L"native" && implementation != L"layer")) return 2;
        leonida::VulkanContext context(no_device_proc, VK_NULL_HANDLE, false);
        Payload payload;
        payload.check_layer_guarantees = implementation == L"layer";
        FactoryRef<> factory;
        const HRESULT hr = implementation == L"native" ? create_native_factory(factory)
                                                       : create_layer_factory(context, factory);
        if (FAILED(hr) || !factory.ptr) {
            std::fprintf(stderr, "factory creation failed: 0x%08lx\n", static_cast<unsigned long>(hr));
            return 1;
        }
        const bool ok = release_case             ? release_reentry(*factory.ptr, scenario == L"release-remove")
                        : scenario == L"reentrant" ? reentrant_replace(*factory.ptr, payload)
                                                   : concurrent_remove(*factory.ptr, payload);
        return ok ? 0 : 1;
    }
    const bool native = argc == 2 && std::wstring_view(argv[1]) == L"--native";
    if (argc != 1 && !native) return 2;
    if (native) {
        const bool reentrant = run_child(L"reentrant", true);
        const bool concurrent = run_child(L"concurrent", true);
        if (!reentrant || !concurrent) return 1;
        std::puts("native: completes (layer-only guarantees not checked)");
        return 0;
    }
    const bool reentrant = report("AddRef reentry and replacement", run_child(L"reentrant", false));
    const bool concurrent = report("Concurrent removal during AddRef", run_child(L"concurrent", false));
    // In a child too: with the reference released under the lock, these relock the store's mutex and hang or abort.
    const bool release_replace = report("Release of a replaced interface reenters the store",
                                        run_child(L"release-replace", false));
    const bool release_remove = report("Release of a removed interface reenters the store",
                                       run_child(L"release-remove", false));

    leonida::VulkanContext context(no_device_proc, VK_NULL_HANDLE, false);
    const bool owner = report("New interface key OOM while acquiring the owner", failed_insert(context, 0, true));
    const bool growth = report("New interface key OOM while growing the store", failed_insert(context, 1, true));
    const bool copy = report("New byte key OOM while copying the bytes", failed_insert(context, 0, false));
    const bool copy_growth = report("New byte key OOM while growing the store", failed_insert(context, 1, false));
    const bool interface_to_interface = report("Interface to interface replacement OOM",
                                               failed_replace_interface(context, true));
    const bool interface_to_bytes = report("Interface to bytes replacement OOM",
                                          failed_replace_interface(context, false));
    const bool bytes_to_interface = report("Bytes to interface replacement OOM", failed_replace_bytes(context, true));
    const bool bytes_to_bytes = report("Bytes to bytes replacement OOM", failed_replace_bytes(context, false));
    const bool updates = report("Concurrent updates keep reference counts balanced", concurrent_updates(context));
    const bool registration = report("Adapter event registration recovers from OOM",
                                     failed_adapter_event_registration(context));
    return reentrant && concurrent && release_replace && release_remove && owner && growth && copy && copy_growth && interface_to_interface &&
           interface_to_bytes && bytes_to_interface && bytes_to_bytes && updates && registration ? 0 : 1;
}
