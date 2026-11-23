// Xbox Series D3D12 imports forwarded to the Windows D3D12 exports.
// The SDK types describe the Windows side. The measured guest ABI differs in the root-signature flag word: Xbox
// sets bits in the high byte (observed 0x03000000) and the PC serializer rejects them, so descriptors are copied
// with that byte cleared before forwarding. On a host that serializes only up to 1.1 (Windows 10's built-in
// d3d12.dll), version 1.2 descriptions are rewritten as 1.1 when no static sampler uses 1.2's flags.
// D3D12XboxCreateDevice returns an Xbox-ABI proxy (d3d12_xs_device.h) over the PC device; only the measured slots
// are overridden, the rest follow the PC vtable.
#include <windows.h>
#include <d3d12.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#ifdef LEONIDA_VULKAN_DEVICE
#include "d3d12_com.h"
#endif
#include "d3d12_trace.h"
#include "d3d12_xs_device.h"
#include "d3d12_xs_rootsig.h"
#include "shims.h"

namespace {

FARPROC host(const char* name) {
    static HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    FARPROC p = d3d12 ? GetProcAddress(d3d12, name) : nullptr;
    if (!p) std::fprintf(stderr, "[rt] d3d12_xs: host d3d12.dll has no %s\n", name);
    return p;
}

using SerializeFn = decltype(&::D3D12SerializeVersionedRootSignature);
using DeserializerFn = decltype(&::D3D12CreateRootSignatureDeserializer);
using LegacySerializeFn = decltype(&::D3D12SerializeRootSignature);
using CreateDeviceFn = decltype(&::D3D12CreateDevice);

// Asked once: does the host serializer take version 1.2 at all? An empty 1.2 description is valid everywhere 1.2 is
// supported; Windows 10's built-in d3d12.dll rejects it with E_INVALIDARG.
bool host_serializes_1_2(SerializeFn fn) {
    static const bool supported = [fn] {
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty{};
        empty.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
        ID3DBlob* blob = nullptr;
        ID3DBlob* error = nullptr;
        const HRESULT hr = fn(&empty, &blob, &error);
        if (blob) blob->Release();
        if (error) error->Release();
        return SUCCEEDED(hr);
    }();
    return supported;
}

HRESULT WINAPI serialize_root_signature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc,
                                        ID3DBlob** blob, ID3DBlob** error) {
    static auto fn = reinterpret_cast<SerializeFn>(host("D3D12SerializeVersionedRootSignature"));
    if (!fn) return E_NOTIMPL;
    if (!desc) return fn(desc, blob, error);

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    if (!rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(*desc, cleared)) {
        std::fprintf(stderr,
                     "[rt] d3d12_xs: D3D12SerializeVersionedRootSignature: unknown descriptor version %d; "
                     "forwarding unchanged (Xbox flags not cleared)\n",
                     static_cast<int>(desc->Version));
        return fn(desc, blob, error);
    }
    if (cleared.Version == D3D_ROOT_SIGNATURE_VERSION_1_2 && !host_serializes_1_2(fn)) {
        // Windows 10's d3d12.dll stops at 1.1 (d3d12_xs_rootsig.h). Rewrite as 1.1 when nothing is lost.
        std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC v11{};
        if (!rt::d3d12_xs::downgrade_root_signature_1_2_to_1_1(cleared, samplers, v11)) {
            std::fprintf(stderr,
                         "[rt] d3d12_xs: D3D12SerializeVersionedRootSignature: version 1.2 with static sampler flags, "
                         "and the host d3d12.dll serializes only up to 1.1; refused\n");
            if (blob) *blob = nullptr;
            if (error) *error = nullptr;
            return E_INVALIDARG;
        }
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true))
            std::fprintf(stderr,
                         "[rt] d3d12_xs: the host d3d12.dll serializes root signatures only up to 1.1; version 1.2 "
                         "descriptions without static sampler flags are serialized as 1.1\n");
        return fn(&v11, blob, error);
    }
    return fn(&cleared, blob, error);
}

// The package DLLs import the versioned serializer; the legacy entry point is kept because it is trivial to forward
// with the same flag fix and the test uses it.
HRESULT WINAPI serialize_root_signature_legacy(const D3D12_ROOT_SIGNATURE_DESC* desc,
                                               D3D_ROOT_SIGNATURE_VERSION version, ID3DBlob** blob,
                                               ID3DBlob** error) {
    static auto fn = reinterpret_cast<LegacySerializeFn>(host("D3D12SerializeRootSignature"));
    if (!fn) return E_NOTIMPL;
    if (!desc) return fn(desc, version, blob, error);

    D3D12_ROOT_SIGNATURE_DESC cleared{};
    rt::d3d12_xs::copy_root_signature_desc_clear_xbox_flags(*desc, cleared);
    return fn(&cleared, version, blob, error);
}

HRESULT WINAPI create_root_signature_deserializer(const void* data, SIZE_T size, REFIID iid, void** out) {
    static auto fn = reinterpret_cast<DeserializerFn>(host("D3D12CreateRootSignatureDeserializer"));
    if (!fn) return E_NOTIMPL;

    // The host deserializer rejects a blob whose RTS0 flags carry the Xbox high
    // byte, so forward a copy with that byte cleared when present (the same fix
    // the serializer applies to descriptors).
    std::vector<uint8_t> cleared;
    if (rt::d3d12_xs::copy_rts0_blob_clear_xbox_flags(data, static_cast<size_t>(size), cleared)) {
        return fn(cleared.data(), cleared.size(), iid, out);
    }
    return fn(data, size, iid, out);
}

// LEONIDA_DRED=1: device-removed diagnostics for the PC device behind the proxy. DRED (auto-breadcrumbs and page
// faults) is switched on before the device is created; a fence waits for UINT64_MAX, the value D3D12 gives every
// fence of a removed device, and a thread then prints the removal reason, each command list's operations around the
// last one the GPU completed, and the page-fault address with the allocations near it.
bool dred_requested() {
    char value[8] = {};
    size_t length = 0;
    return getenv_s(&length, value, "LEONIDA_DRED") == 0 && length > 1 && std::strcmp(value, "0") != 0;
}

void enable_dred() {
    using GetDebugInterfaceFn = decltype(&::D3D12GetDebugInterface);
    static auto get = reinterpret_cast<GetDebugInterfaceFn>(host("D3D12GetDebugInterface"));
    ID3D12DeviceRemovedExtendedDataSettings* settings = nullptr;
    if (!get || FAILED(get(__uuidof(ID3D12DeviceRemovedExtendedDataSettings), reinterpret_cast<void**>(&settings))) ||
        !settings) {
        std::fprintf(stderr, "[rt] d3d12_xs: LEONIDA_DRED: DRED settings not available on this host\n");
        return;
    }
    settings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    settings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    settings->Release();
    std::fprintf(stderr, "[rt] d3d12_xs: LEONIDA_DRED: auto-breadcrumbs and page-fault reporting on\n");
}

void print_dred_allocations(const char* what, const D3D12_DRED_ALLOCATION_NODE* node) {
    for (int n = 0; node && n < 16; node = node->pNext, ++n)
        std::fprintf(stderr, "[rt] d3d12_xs: DRED   %s allocation type %d name %ls\n", what,
                     static_cast<int>(node->AllocationType), node->ObjectNameW ? node->ObjectNameW : L"(none)");
}

void report_device_removed(ID3D12Device* device) {
    const HRESULT reason = device->GetDeviceRemovedReason();
    std::fprintf(stderr, "[rt] d3d12_xs: DRED: device removed, reason 0x%08lx\n", static_cast<unsigned long>(reason));
    ID3D12DeviceRemovedExtendedData* dred = nullptr;
    if (FAILED(device->QueryInterface(__uuidof(ID3D12DeviceRemovedExtendedData), reinterpret_cast<void**>(&dred))) ||
        !dred) {
        std::fprintf(stderr, "[rt] d3d12_xs: DRED: no ID3D12DeviceRemovedExtendedData (was DRED on?)\n");
        return;
    }
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT crumbs{};
    if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&crumbs))) {
        int lists = 0;
        for (const auto* node = crumbs.pHeadAutoBreadcrumbNode; node && lists < 32; node = node->pNext, ++lists) {
            const UINT done = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0;
            if (done >= node->BreadcrumbCount) continue;  // this list finished: not where the GPU stopped
            std::fprintf(stderr, "[rt] d3d12_xs: DRED list %ls on queue %ls: %u of %u operations completed\n",
                         node->pCommandListDebugNameW ? node->pCommandListDebugNameW : L"(unnamed)",
                         node->pCommandQueueDebugNameW ? node->pCommandQueueDebugNameW : L"(unnamed)", done,
                         node->BreadcrumbCount);
            // shortcut: ops are printed as D3D12_AUTO_BREADCRUMB_OP numbers (d3d12.h names them), not as names.
            const UINT first = done > 4 ? done - 4 : 0;
            const UINT last = done + 4 < node->BreadcrumbCount ? done + 4 : node->BreadcrumbCount - 1;
            for (UINT i = first; i <= last; ++i)
                std::fprintf(stderr, "[rt] d3d12_xs: DRED   op %u: %d%s\n", i, static_cast<int>(node->pCommandHistory[i]),
                             i == done ? "  <- first not completed" : "");
        }
    }
    D3D12_DRED_PAGE_FAULT_OUTPUT fault{};
    if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&fault)) && fault.PageFaultVA) {
        std::fprintf(stderr, "[rt] d3d12_xs: DRED page fault at GPU VA 0x%llx\n",
                     static_cast<unsigned long long>(fault.PageFaultVA));
        print_dred_allocations("existing", fault.pHeadExistingAllocationNode);
        print_dred_allocations("recently freed", fault.pHeadRecentFreedAllocationNode);
    }
    dred->Release();
    std::fflush(stderr);
}

void watch_device_removal(ID3D12Device* device) {
    ID3D12Fence* fence = nullptr;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&fence))))
        return;
    HANDLE removed = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!removed || FAILED(fence->SetEventOnCompletion(UINT64_MAX, removed))) {
        if (removed) CloseHandle(removed);
        fence->Release();
        return;
    }
    // One watcher per device, stopped and released by an atexit handler: a thread still calling into the device while
    // the process tears d3d12.dll down ends the process with an error code.
    struct Watch {
        ID3D12Device* device;
        ID3D12Fence* fence;
        HANDLE removed;
        HANDLE stop;
        HANDLE thread;
    };
    static std::atomic<Watch*> watches[4] = {};
    static std::atomic<int> watch_count{0};
    const int index = watch_count.fetch_add(1);
    HANDLE stop = index < 4 ? CreateEventW(nullptr, TRUE, FALSE, nullptr) : nullptr;
    if (!stop) {
        CloseHandle(removed);
        fence->Release();
        return;
    }
    device->AddRef();
    auto* watch = new Watch{device, fence, removed, stop, nullptr};
    watch->thread = CreateThread(
        nullptr, 0,
        [](void* p) -> DWORD {
            auto* w = static_cast<Watch*>(p);
            // The fence event, or the reason itself every 20 ms: a title that stops itself right after a removed
            // device can end the process before the fence's event is delivered.
            const HANDLE wake[2] = {w->removed, w->stop};
            DWORD woke = WAIT_TIMEOUT;
            while ((woke = WaitForMultipleObjects(2, wake, FALSE, 20)) == WAIT_TIMEOUT &&
                   w->device->GetDeviceRemovedReason() == S_OK) {
            }
            if (woke != WAIT_OBJECT_0 + 1) report_device_removed(w->device);
            return 0;
        },
        watch, 0, nullptr);
    if (!watch->thread) {
        CloseHandle(stop);
        CloseHandle(removed);
        fence->Release();
        device->Release();
        delete watch;
        return;
    }
    watches[index].store(watch);
    static const bool registered = [] {
        std::atexit([] {
            for (auto& slot : watches) {
                Watch* w = slot.exchange(nullptr);
                if (!w) continue;
                SetEvent(w->stop);
                WaitForSingleObject(w->thread, 1000);
                CloseHandle(w->thread);
                CloseHandle(w->stop);
                CloseHandle(w->removed);
                w->fence->Release();
                w->device->Release();
                delete w;
            }
        });
        return true;
    }();
    (void)registered;
}

// Measured shape: D3D12XboxCreateDevice(adapter, params, riid, out), where params is a 24-word block that is only
// logged and not used to create the device. The struct's total size is unknown, so the log reads 24 x 32-bit words
// (96 bytes) and no more. The PC D3D12CreateDevice needs a minimum feature level instead, so 12_0 is used as the
// least restrictive floor; higher feature levels are still exposed by the created device.
HRESULT WINAPI D3D12XboxCreateDevice(IUnknown* adapter, const void* params, const IID* riid, void** out) {
    constexpr auto method = "D3D12XboxCreateDevice";
    using rt::d3d12_trace::result;
    if (!out) return result(method, E_POINTER);
    *out = nullptr;
    if (!riid) return result(method, E_POINTER);

    const bool tracing = rt::d3d12_trace::enabled();
    if (!tracing)
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice adapter=%p params=%p riid=%08lx-%04x-%04x\n",
                     adapter, params, static_cast<unsigned long>(riid->Data1), riid->Data2, riid->Data3);
    // The 24 words are the title's device parameters (docs/boot.md, question 1). Unlike the pointers above they are the
    // same on every run, so they are printed with --trace-d3d12 as well, and one traced run shows them.
    if (params) {
        const uint32_t* words = static_cast<const uint32_t*>(params);
        char line[24 * 11 + 1];
        line[0] = '\0';
        size_t used = 0;
        for (int i = 0; i < 24 && used < sizeof(line); ++i) {
            int n = std::snprintf(line + used, sizeof(line) - used, "%s%08x", i ? " " : "",
                                  static_cast<unsigned>(words[i]));
            if (n < 0) break;
            used += static_cast<size_t>(n);
        }
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice params[24] =%s\n", line);
    } else {
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice params = null\n");
    }

    static std::atomic<bool> proxy_logged{false};
    if (!proxy_logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] d3d12_xs: D3D12XboxCreateDevice: wrapping the PC ID3D12Device in the Xbox-ABI proxy "
                     "(slot 62 -> CreateComputePipelineState, slot 76 -> CreateCommittedResource).\n");
    }

#ifdef LEONIDA_VULKAN_DEVICE
    // Opt-in until the Vulkan device covers the first frame: LEONIDA_D3D12=vulkan puts the Vulkan-backed device
    // behind the same proxy instead of the host's D3D12 device.
    char backend[16] = {};
    size_t backend_length = 0;
    if (getenv_s(&backend_length, backend, "LEONIDA_D3D12") == 0 && std::strcmp(backend, "vulkan") == 0) {
        ID3D12Device* vulkan = nullptr;
        HRESULT vhr = rt::d3d12::create_vulkan_device_standalone(
            __uuidof(ID3D12Device), reinterpret_cast<void**>(&vulkan), rt::d3d12::QueueTestSettings::ignore);
        if (FAILED(vhr)) {
            std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice: Vulkan device failed (hr=0x%08lx)\n",
                         static_cast<unsigned long>(vhr));
            return result(method, vhr);
        }
        vhr = rt::d3d12_xs::create_device_proxy(vulkan, *riid, out);
        if (FAILED(vhr)) {
            vulkan->Release();
            std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice: proxy over the Vulkan device failed "
                                 "(hr=0x%08lx)\n", static_cast<unsigned long>(vhr));
            return result(method, vhr);
        }
        rt::d3d12::set_device_outer(vulkan, static_cast<IUnknown*>(*out));
        if (!tracing)
            std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy %p (Vulkan device %p)\n", *out,
                         static_cast<void*>(vulkan));
        return result(method, S_OK);
    }
#endif

    static auto create = reinterpret_cast<CreateDeviceFn>(host("D3D12CreateDevice"));
    if (!create) return result(method, E_NOTIMPL);

    const bool dred = dred_requested();
    if (dred) enable_dred();
    ID3D12Device* real = nullptr;
    HRESULT hr = create(adapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device),
                        reinterpret_cast<void**>(&real));
    if (SUCCEEDED(hr) && real && dred) watch_device_removal(real);
    if (FAILED(hr) || !real) {
        if (real) real->Release();
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice: PC D3D12CreateDevice failed (hr=0x%08lx)\n",
                     static_cast<unsigned long>(hr));
        return result(method, FAILED(hr) ? hr : E_FAIL);
    }

    hr = rt::d3d12_xs::create_device_proxy(real, *riid, out);
    if (FAILED(hr)) {
        // create_device_proxy keeps the caller's reference on failure.
        real->Release();
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice: proxy creation failed (hr=0x%08lx)\n",
                     static_cast<unsigned long>(hr));
        return result(method, hr);
    }
    if (!tracing)
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy %p (PC device %p)\n", *out,
                     static_cast<void*>(real));
    return result(method, S_OK);
}

static_assert(std::is_same_v<decltype(&serialize_root_signature), SerializeFn>);
static_assert(std::is_same_v<decltype(&create_root_signature_deserializer), DeserializerFn>);
static_assert(std::is_same_v<decltype(&serialize_root_signature_legacy), LegacySerializeFn>);

// Measured call (RT-5, 1.0.2.1): the title calls it once at startup with one pointer to a zeroed 32-byte block whose
// dword at +4 it sets (a count), and only checks the result for failure. The PC/Vulkan side owns virtual memory,
// so the policy is accepted and ignored.
HRESULT WINAPI configure_virtual_memory(void* /*desc*/) { return S_OK; }

}  // namespace

SHIM_TABLE(d3d12_xs,
    {"d3d12_xs.dll", "D3D12SerializeVersionedRootSignature", reinterpret_cast<void*>(&serialize_root_signature)},
    {"d3d12_xs.dll", "D3D12SerializeRootSignature", reinterpret_cast<void*>(&serialize_root_signature_legacy)},
    {"d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer",
     reinterpret_cast<void*>(&create_root_signature_deserializer)},
    {"d3d12_xs.dll", "D3D12XboxCreateDevice", reinterpret_cast<void*>(&D3D12XboxCreateDevice)},
    {"d3d12_xs.dll", "D3D12ConfigureVirtualMemoryX", reinterpret_cast<void*>(&configure_virtual_memory)},
);
