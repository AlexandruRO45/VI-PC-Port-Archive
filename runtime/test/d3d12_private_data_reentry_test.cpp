// Exercise the production COM implementation without creating a GPU device.
#include "../shims/d3d12_com_object.h"

#include <chrono>
#include <future>
#include <thread>

namespace rt::d3d12 {
void stub(const char*) {}
}

namespace {

using namespace std::chrono_literals;
constexpr GUID kValue{0x5eed0001, 0x1d3d, 0x4c12, {0x9a, 0x41, 0x77, 0x10, 0x2b, 0x6c, 0, 1}};
constexpr GUID kInterface{0x5eed0002, 0x1d3d, 0x4c12, {0x9a, 0x41, 0x77, 0x10, 0x2b, 0x6c, 0, 1}};
constexpr UINT kExpected = 81;

class Owner final : public rt::d3d12::Object<Owner, ID3D12Object, ID3D12Object> {
public:
    static constexpr GUID kTag{0x5eed0003, 0x1d3d, 0x4c12, {0x9a, 0x41, 0x77, 0x10, 0x2b, 0x6c, 0, 1}};
};

class Payload final : public IUnknown {
public:
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
            callback_ok = owner->GetPrivateData(kValue, &size, &value) == S_OK &&
                          size == sizeof(value) && value == kExpected;
            // Replace the exact slot the outer call is reading, before it acquires the returned COM reference.
            callback_ok &= owner->SetPrivateData(kInterface, sizeof(value), &value) == S_OK && refs == 1;
        }
        if (entered) {
            entered->set_value();
            callback_ok = removed.wait_for(5s) == std::future_status::ready && refs == 1;
        }
        return ++refs;
    }
    ULONG STDMETHODCALLTYPE Release() override { return --refs; }

    // Keep the fixture on the stack so a lost COM reference shows up as an assertion, not undefined behavior.
    std::atomic<ULONG> refs{1};
    ID3D12Object* owner = nullptr;
    std::promise<void>* entered = nullptr;
    std::future<void> removed;
    bool callback_ok = false;
};

bool reentrant_replace() {
    Payload payload;
    Owner owner;
    if (owner.SetPrivateData(kValue, sizeof(kExpected), &kExpected) != S_OK ||
        owner.SetPrivateDataInterface(kInterface, &payload) != S_OK ||
        payload.Release() != 1) return false;
    payload.owner = &owner;
    UINT size = sizeof(IUnknown*);
    IUnknown* result = nullptr;
    const HRESULT hr = owner.GetPrivateData(kInterface, &size, &result);
    const bool read_ok = hr == S_OK && size == sizeof(result) && result == &payload && payload.callback_ok;
    const bool released = result == &payload && result->Release() == 0;
    UINT value = 0;
    size = sizeof(value);
    return read_ok && released && owner.GetPrivateData(kInterface, &size, &value) == S_OK &&
           size == sizeof(value) && value == kExpected;
}

bool concurrent_remove() {
    Payload payload;
    Owner owner;
    if (owner.SetPrivateDataInterface(kInterface, &payload) != S_OK || payload.Release() != 1) return false;
    std::promise<void> entered, removed;
    auto started = entered.get_future();
    payload.entered = &entered;
    payload.removed = removed.get_future();

    UINT size = sizeof(IUnknown*);
    IUnknown* result = nullptr;
    HRESULT hr = E_FAIL;
    std::thread reader([&] { hr = owner.GetPrivateData(kInterface, &size, &result); });
    const bool started_read = started.wait_for(5s) == std::future_status::ready;
    const bool erased = started_read && owner.SetPrivateData(kInterface, 0, nullptr) == S_OK;
    const bool retained = payload.refs == 1;
    removed.set_value();
    reader.join();

    const bool read_ok = payload.callback_ok && hr == S_OK && size == sizeof(result) && result == &payload;
    const bool released = result == &payload && result->Release() == 0;
    return erased && retained && read_ok && released &&
           owner.GetPrivateData(kInterface, &size, nullptr) == DXGI_ERROR_NOT_FOUND && size == 0;
}

}  // namespace

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    const bool reentry = reentrant_replace();
    const bool removal = concurrent_remove();
    std::printf("AddRef reentry and replacement: %s\n", reentry ? "PASS" : "FAIL");
    std::printf("Concurrent removal during AddRef: %s\n", removal ? "PASS" : "FAIL");
    return reentry && removal ? 0 : 1;
}
