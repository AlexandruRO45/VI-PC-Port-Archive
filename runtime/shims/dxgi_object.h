// IUnknown and IDXGIObject for the layer's DXGI objects (factory, adapter, swapchain). The D3D12 Object base in
// d3d12_com.cpp does not fit: IDXGIObject has GetParent and SetPrivateDataInterface where ID3D12Object has SetName.
// Same idea: one inheritance chain, so every listed interface shares the object pointer; Derived::kTag returns the
// implementation; an unknown IID is logged and refused. The parent (what GetParent returns) is held by reference.
#pragma once

#include <windows.h>
#include <dxgi.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

namespace rt::d3d12 {
void stub(const char* method);
}

namespace rt::dxgi {

inline void log_unknown_iid(const char* what, REFIID riid) {
    std::fprintf(stderr, "[rt] dxgi: %s: unknown IID {%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}\n", what,
                 static_cast<unsigned long>(riid.Data1), riid.Data2, riid.Data3, riid.Data4[0], riid.Data4[1],
                 riid.Data4[2], riid.Data4[3], riid.Data4[4], riid.Data4[5], riid.Data4[6], riid.Data4[7]);
}

template<class Derived, class Stubs, class... Interfaces>
class DxgiObject : public Stubs {
public:
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete static_cast<Derived*>(this);
        return left;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == Derived::kTag) {
            *out = static_cast<Derived*>(this);
        } else if (riid == __uuidof(IUnknown) || ((riid == __uuidof(Interfaces)) || ...)) {
            *out = static_cast<Stubs*>(this);
        } else {
            log_unknown_iid("QueryInterface", riid);
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT size, const void* data) override {
        std::lock_guard lock(data_mutex_);
        std::erase_if(data_, [&](const auto& entry) { return entry.first == guid; });
        if (data && size) {
            const auto* bytes = static_cast<const std::uint8_t*>(data);
            data_.emplace_back(guid, std::vector<std::uint8_t>(bytes, bytes + size));
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* size, void* data) override {
        if (!size) return E_INVALIDARG;
        std::lock_guard lock(data_mutex_);
        for (const auto& [key, bytes] : data_) {
            if (key != guid) continue;
            if (data) {
                if (*size < bytes.size()) return DXGI_ERROR_MORE_DATA;
                std::memcpy(data, bytes.data(), bytes.size());
            }
            *size = static_cast<UINT>(bytes.size());
            return S_OK;
        }
        *size = 0;
        return DXGI_ERROR_NOT_FOUND;
    }

    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!parent_) {
            rt::d3d12::stub("IDXGIObject::GetParent (no parent)");
            return E_NOINTERFACE;
        }
        return parent_->QueryInterface(riid, out);
    }

protected:
    explicit DxgiObject(IUnknown* parent = nullptr) : parent_(parent) {
        if (parent_) parent_->AddRef();
    }
    virtual ~DxgiObject() {
        if (parent_) parent_->Release();
    }

    template<class T>
    static T* from(IUnknown* object) {
        if (!object) return nullptr;
        void* implementation = nullptr;
        if (FAILED(object->QueryInterface(T::kTag, &implementation))) return nullptr;
        static_cast<T*>(implementation)->Release();  // the caller holds its own reference
        return static_cast<T*>(implementation);
    }

private:
    std::atomic<ULONG> refs_{1};
    IUnknown* parent_;
    std::mutex data_mutex_;
    std::vector<std::pair<GUID, std::vector<std::uint8_t>>> data_;
};

}  // namespace rt::dxgi
