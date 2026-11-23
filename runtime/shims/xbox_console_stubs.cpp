// Bring-up shims for console-exclusive Xbox Series X|S DLLs:
//   - xcrdapi.dll  (DirectStorage factory)
//   - xg_xs.dll    (Xbox Graphics texture tiling / swizzle)
//   - chad.dll     (Xbox Audio DSP)
//   - xapu.dll     (Xbox Audio Processing Unit)
//   - acphal.dll   (Audio Co-Processor HAL & APU heap)
//
// All entry points log on first call so bring-up traces record call order cleanly.
#include <windows.h>
#include <unknwn.h>
#include <malloc.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "file_system.h"
#include "shims.h"

namespace {

void log_once(std::atomic<bool>& flag, const char* dll, const char* fn) {
    if (!flag.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] %s: %s called (bring-up stub, logged once)\n", dll, fn);
    }
}

void log_iid(const char* tag, REFIID riid) {
    std::fprintf(stderr,
                 "[rt] xcrdapi: %s IID={%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}\n",
                 tag, static_cast<unsigned long>(riid.Data1), riid.Data2, riid.Data3, riid.Data4[0], riid.Data4[1],
                 riid.Data4[2], riid.Data4[3], riid.Data4[4], riid.Data4[5], riid.Data4[6], riid.Data4[7]);
}

// ---- xcrdapi.dll (DirectStorage COM bring-up shim) --------------------------------------------------------------

class DStorageFileShim : public IUnknown {
public:
    explicit DStorageFileShim(HANDLE h) : handle_(h) {}
    ~DStorageFileShim() { close_handle(); }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = this;
        AddRef();
        (void)riid;
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (left == 0) delete this;
        return left;
    }

    // IDStorageFileX vtable: slot 3 = Close, slot 4 = GetFileInformation
    virtual void STDMETHODCALLTYPE Close() {
        std::fprintf(stderr, "[rt] xcrdapi: File::Close()\n");
        close_handle();
    }
    virtual HRESULT STDMETHODCALLTYPE GetFileInformation(BY_HANDLE_FILE_INFORMATION* info) {
        if (!info) return E_POINTER;
        std::lock_guard<std::mutex> lock(mutex_);
        if (handle_ == INVALID_HANDLE_VALUE) return E_HANDLE;
        const BOOL ok = GetFileInformationByHandle(handle_, info);
        const HRESULT hr = ok ? S_OK : HRESULT_FROM_WIN32(GetLastError());
        std::fprintf(stderr, "[rt] xcrdapi: File::GetFileInformation -> hr=0x%08lx size=%llu\n",
                     static_cast<unsigned long>(hr),
                     (static_cast<unsigned long long>(info->nFileSizeHigh) << 32) | info->nFileSizeLow);
        return hr;
    }

    HANDLE native_handle() const { return handle_; }

private:
    void close_handle() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
    }

    std::atomic<ULONG> refs_{1};
    std::mutex mutex_;
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

class DStorageStatusArrayShim : public IUnknown {
public:
    explicit DStorageStatusArrayShim(UINT32 capacity) : results_(capacity ? capacity : 1, S_OK) {}
    ~DStorageStatusArrayShim() = default;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = this;
        AddRef();
        (void)riid;
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (left == 0) delete this;
        return left;
    }

    // IDStorageStatusArrayX vtable: slot 3 = IsComplete, slot 4 = GetHResult
    virtual BOOL STDMETHODCALLTYPE IsComplete(UINT32 /*index*/) { return TRUE; }
    virtual HRESULT STDMETHODCALLTYPE GetHResult(UINT32 index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= results_.size()) return E_INVALIDARG;
        return results_[index];
    }

private:
    std::atomic<ULONG> refs_{1};
    std::mutex mutex_;
    std::vector<HRESULT> results_;
};

class DStorageQueueShim : public IUnknown {
public:
    DStorageQueueShim() { error_event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr); }
    ~DStorageQueueShim() {
        if (error_event_) CloseHandle(error_event_);
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = this;
        AddRef();
        (void)riid;
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (left == 0) delete this;
        return left;
    }

    // IDStorageQueueX / X1 / X2 vtable
    virtual void STDMETHODCALLTYPE EnqueueRequest(const void* /*request*/) {
        static std::atomic<bool> logged{false};
        log_once(logged, "xcrdapi", "IDStorageQueueX::EnqueueRequest");
    }
    virtual void STDMETHODCALLTYPE EnqueueStatus(IUnknown* /*statusArray*/, UINT32 /*index*/) {
        static std::atomic<bool> logged{false};
        log_once(logged, "xcrdapi", "IDStorageQueueX::EnqueueStatus");
    }
    virtual void STDMETHODCALLTYPE EnqueueSignal(IUnknown* /*fence*/, UINT64 /*value*/) {
        static std::atomic<bool> logged{false};
        log_once(logged, "xcrdapi", "IDStorageQueueX::EnqueueSignal");
    }
    virtual void STDMETHODCALLTYPE Submit() {
        static std::atomic<bool> logged{false};
        log_once(logged, "xcrdapi", "IDStorageQueueX::Submit");
        std::lock_guard<std::mutex> lock(mutex_);
        for (HANDLE ev : pending_events_) {
            if (ev && ev != INVALID_HANDLE_VALUE) SetEvent(ev);
        }
        pending_events_.clear();
    }
    virtual void STDMETHODCALLTYPE CancelRequestsWithTag(UINT64 /*mask*/, UINT64 /*value*/) {}
    virtual void STDMETHODCALLTYPE Close() {}
    virtual HANDLE STDMETHODCALLTYPE GetErrorEvent() { return error_event_; }
    virtual void STDMETHODCALLTYPE RetrieveErrorRecord(void* record) {
        if (record) std::memset(record, 0, 128);
    }
    virtual void STDMETHODCALLTYPE Query(void* info) {
        if (info) std::memset(info, 0, 48);
    }
    // IDStorageQueueX1
    virtual void STDMETHODCALLTYPE EnqueueSetEvent(HANDLE handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_events_.push_back(handle);
    }
    // IDStorageQueueX2
    virtual void STDMETHODCALLTYPE Query1(void* info) {
        if (info) std::memset(info, 0, 64);
    }

private:
    std::atomic<ULONG> refs_{1};
    std::mutex mutex_;
    HANDLE error_event_ = nullptr;
    std::vector<HANDLE> pending_events_;
};

std::wstring resolve_dstorage_path(const wchar_t* path) {
    if (!path) return {};
    std::wstring s(path);
    auto strip_seps = [](std::wstring rest) {
        while (!rest.empty() && (rest.front() == L'\\' || rest.front() == L'/')) rest.erase(rest.begin());
        return rest;
    };
    if (s.size() >= 5 && _wcsnicmp(s.c_str(), L"GAME:", 5) == 0) {
        const auto game = rt::fs::game_dir();
        if (!game.empty()) return (game / strip_seps(s.substr(5))).wstring();
    } else if (s.size() >= 2 && (s[0] == L'G' || s[0] == L'g') && s[1] == L':') {
        const auto game = rt::fs::game_dir();
        if (!game.empty()) return (game / strip_seps(s.substr(2))).wstring();
    } else if (s.size() >= 2 && (s[0] == L'T' || s[0] == L't') && s[1] == L':') {
        const auto ov = rt::fs::overlay_dir();
        if (!ov.empty()) return (ov / L".gta6rt-temp" / strip_seps(s.substr(2))).wstring();
    }
    return s;
}

class DStorageFactoryShim : public IUnknown {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        log_iid("Factory::QueryInterface", riid);
        *ppv = this;
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        return left;  // singleton factory
    }

    // IDStorageFactoryX / X1 / X2 vtable
    virtual HRESULT STDMETHODCALLTYPE CreateQueue(const void* /*desc*/, REFIID riid, void** ppv) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            log_iid("Factory::CreateQueue", riid);
        }
        if (!ppv) return E_POINTER;
        *ppv = new DStorageQueueShim();
        return S_OK;
    }
    virtual HRESULT STDMETHODCALLTYPE OpenFile(const WCHAR* path, REFIID riid, void** ppv) {
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        if (!path) return E_INVALIDARG;
        (void)riid;
        const std::wstring resolved = resolve_dstorage_path(path);
        HANDLE h = CreateFileW(resolved.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            const DWORD err = GetLastError();
            std::fprintf(stderr, "[rt] xcrdapi: OpenFile failed (err=%lu) for '%ls'\n",
                         static_cast<unsigned long>(err), resolved.c_str());
            return HRESULT_FROM_WIN32(err);
        }
        static std::atomic<int> open_log_count{0};
        if (open_log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
            std::fprintf(stderr, "[rt] xcrdapi: OpenFile OK '%ls'\n", resolved.c_str());
        }
        *ppv = new DStorageFileShim(h);
        return S_OK;
    }
    virtual HRESULT STDMETHODCALLTYPE CreateStatusArray(UINT32 capacity, PCSTR /*name*/, REFIID /*riid*/, void** ppv) {
        static std::atomic<bool> logged{false};
        log_once(logged, "xcrdapi", "Factory::CreateStatusArray");
        if (!ppv) return E_POINTER;
        *ppv = new DStorageStatusArrayShim(capacity);
        return S_OK;
    }
    virtual void STDMETHODCALLTYPE SetDebugFlags(UINT32 flags) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xcrdapi: Factory::SetDebugFlags(0x%x)\n", flags);
        }
    }
    virtual HRESULT STDMETHODCALLTYPE SetCpuAffinity(UINT64 affinity) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xcrdapi: Factory::SetCpuAffinity(0x%llx)\n",
                         static_cast<unsigned long long>(affinity));
        }
        return S_OK;
    }
    virtual HRESULT STDMETHODCALLTYPE SetStagingBufferSize(UINT32 size) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xcrdapi: Factory::SetStagingBufferSize(%u)\n", size);
        }
        return S_OK;
    }
    // IDStorageFactoryX1
    virtual void STDMETHODCALLTYPE GetStats(void* stats) {
        static std::atomic<bool> logged{false};
        log_once(logged, "xcrdapi", "Factory::GetStats");
        if (stats) std::memset(stats, 0, 64);
    }
    // IDStorageFactoryX2
    virtual HRESULT STDMETHODCALLTYPE CreateQueue1(const void* desc, REFIID riid, void** ppv) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            log_iid("Factory::CreateQueue1", riid);
        }
        (void)desc;
        if (!ppv) return E_POINTER;
        *ppv = new DStorageQueueShim();
        return S_OK;
    }

private:
    std::atomic<ULONG> refs_{1};
};

DStorageFactoryShim g_dstorage_factory;

HRESULT WINAPI DStorageGetFactory_stub(REFIID riid, void** out) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        log_iid("DStorageGetFactory -> S_OK", riid);
    }
    if (!out) return E_POINTER;
    g_dstorage_factory.AddRef();
    *out = &g_dstorage_factory;
    return S_OK;
}

// ---- xg_xs.dll (Xbox Graphics texture tiling) -------------------------------------------------------------------

HRESULT WINAPI XGCreateTextureComputer_stub(const void* /*desc*/, void** out) {
    static std::atomic<bool> logged{false};
    log_once(logged, "xg_xs", "XGCreateTextureComputer -> E_NOTIMPL");
    if (out) *out = nullptr;
    return E_NOTIMPL;
}

uint32_t WINAPI XGSuggestSwizzleMode_stub(const void* /*desc*/) {
    static std::atomic<bool> logged{false};
    log_once(logged, "xg_xs", "XGSuggestSwizzleMode -> 0 (linear)");
    return 0;
}

// ---- chad.dll (Xbox Audio DSP) ----------------------------------------------------------------------------------

HRESULT WINAPI XDspActivate_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspActivate -> S_OK");
    return S_OK;
}

HRESULT WINAPI XDspConnectWithMaximumStreamLimit_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspConnectWithMaximumStreamLimit -> S_OK");
    return S_OK;
}

HRESULT WINAPI XDspDeactivate_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspDeactivate -> S_OK");
    return S_OK;
}

HRESULT WINAPI XDspDisconnect_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspDisconnect -> S_OK");
    return S_OK;
}

HRESULT WINAPI XDspSubmitCommand_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "chad", "XDspSubmitCommand -> S_OK");
    return S_OK;
}

// ---- xapu.dll (Xbox Audio Processing Unit) ----------------------------------------------------------------------

HRESULT WINAPI XApuConnect_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuConnect -> S_OK");
    return S_OK;
}

HRESULT WINAPI XApuDisconnect_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuDisconnect -> S_OK");
    return S_OK;
}

HRESULT WINAPI XApuEnqueueCommand_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuEnqueueCommand -> S_OK");
    return S_OK;
}

HRESULT WINAPI XApuDequeueResult_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuDequeueResult -> S_FALSE");
    return S_FALSE;
}

uint32_t WINAPI XApuGetResultCount_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "xapu", "XApuGetResultCount -> 0");
    return 0;
}

// ---- acphal.dll (Audio Co-Processor HAL) ------------------------------------------------------------------------

HRESULT WINAPI AcpHalCreate_stub(void** out) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "AcpHalCreate -> E_NOTIMPL");
    if (out) *out = nullptr;
    return E_NOTIMPL;
}

HRESULT WINAPI AcpHalAllocateShapeContexts_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "AcpHalAllocateShapeContexts -> S_OK");
    return S_OK;
}

HRESULT WINAPI AcpHalReleaseShapeContexts_stub() {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "AcpHalReleaseShapeContexts -> S_OK");
    return S_OK;
}

HRESULT WINAPI ApuCreateHeap_stub(uint32_t /*cached_size*/, uint32_t /*noncached_size*/) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuCreateHeap -> S_OK");
    return S_OK;
}

HRESULT WINAPI ApuAlloc_stub(void** virtual_address, uint32_t* apu_address, uint32_t size, uint32_t alignment) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuAlloc -> aligned heap");
    const size_t align = (alignment >= 64 && (alignment & (alignment - 1)) == 0) ? alignment : 64;
    void* p = _aligned_malloc(size ? size : 1, align);
    if (!p) return E_OUTOFMEMORY;
    if (virtual_address) *virtual_address = p;
    if (apu_address) *apu_address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) & 0xFFFFFFFFu);
    return S_OK;
}

void WINAPI ApuFree_stub(void* virtual_address) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuFree");
    _aligned_free(virtual_address);
}

BOOL WINAPI ApuIsVirtualAddressValid_stub(const void* virtual_address, uint32_t /*size*/) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuIsVirtualAddressValid");
    return virtual_address != nullptr ? TRUE : FALSE;
}

uint32_t WINAPI ApuMapVirtualAddress_stub(const void* virtual_address) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuMapVirtualAddress");
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(virtual_address) & 0xFFFFFFFFu);
}

void* WINAPI ApuMapApuAddress_stub(uint32_t /*apu_address*/) {
    static std::atomic<bool> logged{false};
    log_once(logged, "acphal", "ApuMapApuAddress -> nullptr");
    return nullptr;
}

}  // namespace

SHIM_TABLE(xbox_console_stubs,
    {"xcrdapi.dll", "DStorageGetFactory", reinterpret_cast<void*>(&DStorageGetFactory_stub)},
    {"chad.dll", "XDspActivate", reinterpret_cast<void*>(&XDspActivate_stub)},
    {"chad.dll", "XDspConnectWithMaximumStreamLimit", reinterpret_cast<void*>(&XDspConnectWithMaximumStreamLimit_stub)},
    {"chad.dll", "XDspDeactivate", reinterpret_cast<void*>(&XDspDeactivate_stub)},
    {"chad.dll", "XDspDisconnect", reinterpret_cast<void*>(&XDspDisconnect_stub)},
    {"chad.dll", "XDspSubmitCommand", reinterpret_cast<void*>(&XDspSubmitCommand_stub)},
    {"xapu.dll", "XApuConnect", reinterpret_cast<void*>(&XApuConnect_stub)},
    {"xapu.dll", "XApuDisconnect", reinterpret_cast<void*>(&XApuDisconnect_stub)},
    {"xapu.dll", "XApuEnqueueCommand", reinterpret_cast<void*>(&XApuEnqueueCommand_stub)},
    {"xapu.dll", "XApuDequeueResult", reinterpret_cast<void*>(&XApuDequeueResult_stub)},
    {"xapu.dll", "XApuGetResultCount", reinterpret_cast<void*>(&XApuGetResultCount_stub)},
    {"acphal.dll", "AcpHalCreate", reinterpret_cast<void*>(&AcpHalCreate_stub)},
    {"acphal.dll", "AcpHalAllocateShapeContexts", reinterpret_cast<void*>(&AcpHalAllocateShapeContexts_stub)},
    {"acphal.dll", "AcpHalReleaseShapeContexts", reinterpret_cast<void*>(&AcpHalReleaseShapeContexts_stub)},
    {"acphal.dll", "ApuCreateHeap", reinterpret_cast<void*>(&ApuCreateHeap_stub)},
    {"acphal.dll", "ApuAlloc", reinterpret_cast<void*>(&ApuAlloc_stub)},
    {"acphal.dll", "ApuFree", reinterpret_cast<void*>(&ApuFree_stub)},
    {"acphal.dll", "ApuIsVirtualAddressValid", reinterpret_cast<void*>(&ApuIsVirtualAddressValid_stub)},
    {"acphal.dll", "ApuMapVirtualAddress", reinterpret_cast<void*>(&ApuMapVirtualAddress_stub)},
    {"acphal.dll", "ApuMapApuAddress", reinterpret_cast<void*>(&ApuMapApuAddress_stub)},
);
