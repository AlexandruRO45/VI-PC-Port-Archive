// Media Foundation (MFPlat.DLL / MFReadWrite.dll) shims with console-path resolution and offline fallbacks.
//
// game_scarlett_master.exe imports MFStartup, MFShutdown, MFCreateAttributes, MFCreateMediaType, MFCreateSourceResolver,
// MFGetSystemTime (MFPlat.DLL), and MFCreateSourceReaderFromMediaSource (MFReadWrite.dll). Bound directly to the host:
//   - IMFSourceResolver::CreateObjectFromURL / BeginCreateObjectFromURL fail on GAME:, G:, and T: console paths;
//   - unbalanced or guest-triggered MFShutdown drops host Media Foundation refcounts;
//   - hosts without Media Feature Pack (Windows N / Server / minimal Wine) lack mfplat.dll or mfreadwrite.dll.
//
// This shim balances guest MFStartup/MFShutdown state in-process, wraps IMFSourceResolver to translate console paths
// via rt::fs roots (without creating overlay directories on read queries and enforcing traversal/deletion rules), and
// falls back cleanly to pure-C++ IMFAttributes / IMFMediaType / IMFSourceResolver / MFGetSystemTime implementations when
// host Media Foundation is unavailable or LEONIDA_MF_OFFLINE=1 is set.
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mferror.h>
#include <mfreadwrite.h>
#include <objbase.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "file_system.h"
#include "shims.h"

namespace rt::mf {
namespace {

struct GuidLess {
    bool operator()(const GUID& a, const GUID& b) const { return std::memcmp(&a, &b, sizeof(GUID)) < 0; }
};

constexpr GUID kMfMtMajorType{0x48eba18e, 0xf8c9, 0x4687, {0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f}};
constexpr GUID kMfMtSubtype{0xf7e34c9a, 0x42e8, 0x4714, {0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5}};
constexpr GUID kMfMtAllSamplesIndependent{
    0xc9173739, 0x5e56, 0x461c, {0xb7, 0x13, 0x46, 0xfb, 0x99, 0x5c, 0xb9, 0x5f}};

using MfStartupFn = HRESULT(STDAPICALLTYPE*)(ULONG, DWORD);
using MfShutdownFn = HRESULT(STDAPICALLTYPE*)();
using MfCreateAttributesFn = HRESULT(STDAPICALLTYPE*)(IMFAttributes**, UINT32);
using MfCreateMediaTypeFn = HRESULT(STDAPICALLTYPE*)(IMFMediaType**);
using MfCreateSourceResolverFn = HRESULT(STDAPICALLTYPE*)(IMFSourceResolver**);
using MfGetSystemTimeFn = MFTIME(STDAPICALLTYPE*)();
using MfCreateSourceReaderFromMediaSourceFn = HRESULT(STDAPICALLTYPE*)(IMFMediaSource*, IMFAttributes*,
                                                                       IMFSourceReader**);

struct HostMfApi {
    bool offline_forced = false;
    HMODULE mfplat = nullptr;
    HMODULE mfreadwrite = nullptr;
    MfStartupFn startup = nullptr;
    MfShutdownFn shutdown = nullptr;
    MfCreateAttributesFn create_attributes = nullptr;
    MfCreateMediaTypeFn create_media_type = nullptr;
    MfCreateSourceResolverFn create_source_resolver = nullptr;
    MfGetSystemTimeFn get_system_time = nullptr;
    MfCreateSourceReaderFromMediaSourceFn create_source_reader = nullptr;
};

INIT_ONCE g_api_once = INIT_ONCE_STATIC_INIT;
HostMfApi g_host_api{};

BOOL CALLBACK init_host_api_once(PINIT_ONCE, PVOID, PVOID*) {
    char env_buf[16]{};
    const DWORD env_len = GetEnvironmentVariableA("LEONIDA_MF_OFFLINE", env_buf, static_cast<DWORD>(sizeof(env_buf)));
    if (env_len > 0 && env_buf[0] != '\0' && std::strcmp(env_buf, "0") != 0) {
        g_host_api.offline_forced = true;
        return TRUE;
    }

    g_host_api.mfplat = LoadLibraryExW(L"mfplat.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (g_host_api.mfplat) {
        g_host_api.startup = reinterpret_cast<MfStartupFn>(GetProcAddress(g_host_api.mfplat, "MFStartup"));
        g_host_api.shutdown = reinterpret_cast<MfShutdownFn>(GetProcAddress(g_host_api.mfplat, "MFShutdown"));
        g_host_api.create_attributes =
            reinterpret_cast<MfCreateAttributesFn>(GetProcAddress(g_host_api.mfplat, "MFCreateAttributes"));
        g_host_api.create_media_type =
            reinterpret_cast<MfCreateMediaTypeFn>(GetProcAddress(g_host_api.mfplat, "MFCreateMediaType"));
        g_host_api.create_source_resolver =
            reinterpret_cast<MfCreateSourceResolverFn>(GetProcAddress(g_host_api.mfplat, "MFCreateSourceResolver"));
        g_host_api.get_system_time =
            reinterpret_cast<MfGetSystemTimeFn>(GetProcAddress(g_host_api.mfplat, "MFGetSystemTime"));
    }

    g_host_api.mfreadwrite = LoadLibraryExW(L"mfreadwrite.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (g_host_api.mfreadwrite) {
        g_host_api.create_source_reader = reinterpret_cast<MfCreateSourceReaderFromMediaSourceFn>(
            GetProcAddress(g_host_api.mfreadwrite, "MFCreateSourceReaderFromMediaSource"));
    }
    return TRUE;
}

const HostMfApi& host_api() {
    InitOnceExecuteOnce(&g_api_once, init_host_api_once, nullptr, nullptr);
    return g_host_api;
}

SRWLOCK g_startup_lock = SRWLOCK_INIT;
uint32_t g_guest_startup_depth = 0;
bool g_host_started = false;

bool ascii_iequals(wchar_t a, wchar_t b) {
    if (a >= L'A' && a <= L'Z') a = static_cast<wchar_t>(a + 32);
    if (b >= L'A' && b <= L'Z') b = static_cast<wchar_t>(b + 32);
    return a == b;
}

bool starts_with_icase(const std::wstring& text, const wchar_t* prefix) {
    const size_t n = std::wcslen(prefix);
    if (text.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (!ascii_iequals(text[i], prefix[i])) return false;
    }
    return true;
}

// Resolves a GAME:, G:, or T: URL/path for read-only media loading without creating overlay or scratch directories.
// Sets `handled` true when `url` names a configured console drive (`out_hr` carries the failure on false return).
bool resolve_console_media_url(LPCWSTR url, bool& handled, std::wstring& out_host_path, HRESULT& out_hr) {
    handled = false;
    out_hr = S_OK;
    if (!url) return false;

    std::wstring path(url);
    if (starts_with_icase(path, L"file:///")) {
        path.erase(0, 8);
    }

    enum class Drive { None, Game, Scratch };
    Drive drive = Drive::None;
    size_t prefix_len = 0;
    if (path.size() >= 5 && starts_with_icase(path, L"game:")) {
        drive = Drive::Game;
        prefix_len = 5;
    } else if (path.size() >= 2 && (ascii_iequals(path[0], L'g') || ascii_iequals(path[0], L't')) && path[1] == L':') {
        drive = ascii_iequals(path[0], L'g') ? Drive::Game : Drive::Scratch;
        prefix_len = 2;
    }
    if (drive == Drive::None) return false;

    const std::filesystem::path game_root = rt::fs::game_dir();
    const std::filesystem::path overlay_root = rt::fs::overlay_dir();
    if (drive == Drive::Game && game_root.empty()) return false;
    if (drive == Drive::Scratch && overlay_root.empty()) return false;

    handled = true;

    std::vector<std::wstring> parts;
    size_t pos = prefix_len;
    if (pos < path.size() && (path[pos] == L'\\' || path[pos] == L'/')) ++pos;
    while (pos <= path.size()) {
        size_t next = path.find_first_of(L"\\/", pos);
        if (next == std::wstring::npos) next = path.size();
        const std::wstring seg = path.substr(pos, next - pos);
        if (seg == L"..") {
            if (parts.empty()) {
                out_hr = HRESULT_FROM_WIN32(ERROR_INVALID_NAME);
                return false;
            }
            parts.pop_back();
        } else if (!seg.empty() && seg != L".") {
            parts.push_back(seg);
        }
        if (next == path.size()) break;
        pos = next + 1;
    }

    auto join_parts = [](std::filesystem::path base, const std::vector<std::wstring>& segs) {
        for (const auto& s : segs) base /= s;
        return base;
    };

    std::error_code ec;
    if (drive == Drive::Scratch) {
        const std::filesystem::path candidate = join_parts(overlay_root / L".gta6rt-temp", parts);
        if (!std::filesystem::exists(candidate, ec)) {
            out_hr = HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
            return false;
        }
        out_host_path = candidate.wstring();
        return true;
    }

    if (!overlay_root.empty() && !parts.empty()) {
        const std::filesystem::path deleted_marker = join_parts(overlay_root / L".gta6rt-deleted", parts);
        if (std::filesystem::exists(deleted_marker, ec)) {
            out_hr = HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
            return false;
        }
        const std::filesystem::path overlay_candidate = join_parts(overlay_root, parts);
        if (std::filesystem::exists(overlay_candidate, ec)) {
            out_host_path = overlay_candidate.wstring();
            return true;
        }
    }

    const std::filesystem::path game_candidate = join_parts(game_root, parts);
    if (!std::filesystem::exists(game_candidate, ec)) {
        out_hr = HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
        return false;
    }
    out_host_path = game_candidate.wstring();
    return true;
}

// Pure-C++ IMFAttributes / IMFMediaType fallback for hosts without mfplat.dll or offline mode.
class OfflineMediaType final : public IMFMediaType {
public:
    explicit OfflineMediaType(bool as_media_type) : is_media_type_(as_media_type) {}

    ~OfflineMediaType() { DeleteAllItems(); }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        if (IsEqualIID(riid, __uuidof(IUnknown)) || IsEqualIID(riid, __uuidof(IMFAttributes))) {
            *ppv = static_cast<IMFAttributes*>(this);
        } else if (is_media_type_ && IsEqualIID(riid, __uuidof(IMFMediaType))) {
            *ppv = static_cast<IMFMediaType*>(this);
        } else {
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }

    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&refs_)); }

    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG n = static_cast<ULONG>(InterlockedDecrement(&refs_));
        if (n == 0) delete this;
        return n;
    }

    HRESULT STDMETHODCALLTYPE GetItem(REFGUID guidKey, PROPVARIANT* pValue) override {
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (!pValue) {
            ReleaseSRWLockShared(&lock_);
            return S_OK;
        }
        const HRESULT hr = PropVariantCopy(pValue, &it->second);
        ReleaseSRWLockShared(&lock_);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE GetItemType(REFGUID guidKey, MF_ATTRIBUTE_TYPE* pType) override {
        if (!pType) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        *pType = vt_to_attr_type(it->second.vt);
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE CompareItem(REFGUID guidKey, REFPROPVARIANT Value, BOOL* pbResult) override {
        if (!pbResult) return E_POINTER;
        *pbResult = FALSE;
        PROPVARIANT stored{};
        PropVariantInit(&stored);
        if (FAILED(GetItem(guidKey, &stored))) return S_OK;
        *pbResult = (stored.vt == Value.vt && compare_propvar(stored, Value)) ? TRUE : FALSE;
        PropVariantClear(&stored);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE Compare(IMFAttributes* pTheirs, MF_ATTRIBUTES_MATCH_TYPE MatchType,
                                      BOOL* pbResult) override {
        if (!pTheirs || !pbResult) return E_POINTER;
        *pbResult = FALSE;
        UINT32 count = 0;
        GetCount(&count);
        if (MatchType == MF_ATTRIBUTES_MATCH_ALL_ITEMS) {
            UINT32 their_count = 0;
            pTheirs->GetCount(&their_count);
            if (their_count != count) return S_OK;
        }
        for (UINT32 i = 0; i < count; ++i) {
            GUID key{};
            PROPVARIANT val{};
            PropVariantInit(&val);
            if (FAILED(GetItemByIndex(i, &key, &val))) return S_OK;
            BOOL same = FALSE;
            const HRESULT hr = pTheirs->CompareItem(key, val, &same);
            PropVariantClear(&val);
            if (FAILED(hr) || !same) {
                if (MatchType == MF_ATTRIBUTES_MATCH_OUR_ITEMS || MatchType == MF_ATTRIBUTES_MATCH_ALL_ITEMS)
                    return S_OK;
            }
        }
        *pbResult = TRUE;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetUINT32(REFGUID guidKey, UINT32* punValue) override {
        if (!punValue) return E_POINTER;
        return get_typed<UINT32>(guidKey, VT_UI4, *punValue);
    }

    HRESULT STDMETHODCALLTYPE GetUINT64(REFGUID guidKey, UINT64* punValue) override {
        if (!punValue) return E_POINTER;
        return get_typed<UINT64>(guidKey, VT_UI8, *punValue);
    }

    HRESULT STDMETHODCALLTYPE GetDouble(REFGUID guidKey, double* pfValue) override {
        if (!pfValue) return E_POINTER;
        return get_typed<double>(guidKey, VT_R8, *pfValue);
    }

    HRESULT STDMETHODCALLTYPE GetGUID(REFGUID guidKey, GUID* pguidValue) override {
        if (!pguidValue) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != VT_CLSID || !it->second.puuid) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        *pguidValue = *it->second.puuid;
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetStringLength(REFGUID guidKey, UINT32* pcchLength) override {
        if (!pcchLength) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != VT_LPWSTR || !it->second.pwszVal) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        *pcchLength = static_cast<UINT32>(std::wcslen(it->second.pwszVal));
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetString(REFGUID guidKey, LPWSTR pwszValue, UINT32 cchBufSize,
                                        UINT32* pcchLength) override {
        if (!pwszValue) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != VT_LPWSTR || !it->second.pwszVal) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        const size_t len = std::wcslen(it->second.pwszVal);
        if (pcchLength) *pcchLength = static_cast<UINT32>(len);
        if (cchBufSize <= len) {
            ReleaseSRWLockShared(&lock_);
            return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
        }
        std::wmemcpy(pwszValue, it->second.pwszVal, len + 1);
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetAllocatedString(REFGUID guidKey, LPWSTR* ppwszValue, UINT32* pcchLength) override {
        if (!ppwszValue || !pcchLength) return E_POINTER;
        *ppwszValue = nullptr;
        *pcchLength = 0;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != VT_LPWSTR || !it->second.pwszVal) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        const size_t len = std::wcslen(it->second.pwszVal);
        wchar_t* copy = static_cast<wchar_t*>(CoTaskMemAlloc((len + 1) * sizeof(wchar_t)));
        if (!copy) {
            ReleaseSRWLockShared(&lock_);
            return E_OUTOFMEMORY;
        }
        std::wmemcpy(copy, it->second.pwszVal, len + 1);
        *ppwszValue = copy;
        *pcchLength = static_cast<UINT32>(len);
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetBlobSize(REFGUID guidKey, UINT32* pcbBlobSize) override {
        if (!pcbBlobSize) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != (VT_VECTOR | VT_UI1)) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        *pcbBlobSize = it->second.caub.cElems;
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetBlob(REFGUID guidKey, UINT8* pBuf, UINT32 cbBufSize, UINT32* pcbBlobSize) override {
        if (!pBuf) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != (VT_VECTOR | VT_UI1)) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        if (pcbBlobSize) *pcbBlobSize = it->second.caub.cElems;
        if (cbBufSize < it->second.caub.cElems) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_BUFFERTOOSMALL;
        }
        if (it->second.caub.cElems > 0) std::memcpy(pBuf, it->second.caub.pElems, it->second.caub.cElems);
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetAllocatedBlob(REFGUID guidKey, UINT8** ppBuf, UINT32* pcbSize) override {
        if (!ppBuf || !pcbSize) return E_POINTER;
        *ppBuf = nullptr;
        *pcbSize = 0;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != (VT_VECTOR | VT_UI1)) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        const UINT32 sz = it->second.caub.cElems;
        UINT8* copy = static_cast<UINT8*>(CoTaskMemAlloc(sz ? sz : 1));
        if (!copy) {
            ReleaseSRWLockShared(&lock_);
            return E_OUTOFMEMORY;
        }
        if (sz > 0) std::memcpy(copy, it->second.caub.pElems, sz);
        *ppBuf = copy;
        *pcbSize = sz;
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetUnknown(REFGUID guidKey, REFIID riid, LPVOID* ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != VT_UNKNOWN || !it->second.punkVal) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        const HRESULT hr = it->second.punkVal->QueryInterface(riid, ppv);
        ReleaseSRWLockShared(&lock_);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE SetItem(REFGUID guidKey, REFPROPVARIANT Value) override {
        PROPVARIANT copy{};
        PropVariantInit(&copy);
        const HRESULT hr = PropVariantCopy(&copy, &Value);
        if (FAILED(hr)) return hr;
        AcquireSRWLockExclusive(&lock_);
        auto it = items_.find(guidKey);
        if (it != items_.end()) {
            PropVariantClear(&it->second);
            it->second = copy;
        } else {
            items_.emplace(guidKey, copy);
        }
        ReleaseSRWLockExclusive(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE DeleteItem(REFGUID guidKey) override {
        AcquireSRWLockExclusive(&lock_);
        auto it = items_.find(guidKey);
        if (it != items_.end()) {
            PropVariantClear(&it->second);
            items_.erase(it);
        }
        ReleaseSRWLockExclusive(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE DeleteAllItems() override {
        AcquireSRWLockExclusive(&lock_);
        for (auto& kv : items_) PropVariantClear(&kv.second);
        items_.clear();
        ReleaseSRWLockExclusive(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE SetUINT32(REFGUID guidKey, UINT32 unValue) override {
        PROPVARIANT v{};
        v.vt = VT_UI4;
        v.ulVal = unValue;
        return SetItem(guidKey, v);
    }

    HRESULT STDMETHODCALLTYPE SetUINT64(REFGUID guidKey, UINT64 unValue) override {
        PROPVARIANT v{};
        v.vt = VT_UI8;
        v.uhVal.QuadPart = unValue;
        return SetItem(guidKey, v);
    }

    HRESULT STDMETHODCALLTYPE SetDouble(REFGUID guidKey, double fValue) override {
        PROPVARIANT v{};
        v.vt = VT_R8;
        v.dblVal = fValue;
        return SetItem(guidKey, v);
    }

    HRESULT STDMETHODCALLTYPE SetGUID(REFGUID guidKey, REFGUID guidValue) override {
        PROPVARIANT v{};
        v.vt = VT_CLSID;
        GUID tmp = guidValue;
        v.puuid = &tmp;
        return SetItem(guidKey, v);
    }

    HRESULT STDMETHODCALLTYPE SetString(REFGUID guidKey, LPCWSTR wszValue) override {
        if (!wszValue) return E_INVALIDARG;
        PROPVARIANT v{};
        v.vt = VT_LPWSTR;
        v.pwszVal = const_cast<LPWSTR>(wszValue);
        return SetItem(guidKey, v);
    }

    HRESULT STDMETHODCALLTYPE SetBlob(REFGUID guidKey, const UINT8* pBuf, UINT32 cbBufSize) override {
        if (cbBufSize && !pBuf) return E_INVALIDARG;
        PROPVARIANT v{};
        v.vt = VT_VECTOR | VT_UI1;
        v.caub.cElems = cbBufSize;
        v.caub.pElems = const_cast<UINT8*>(pBuf);
        return SetItem(guidKey, v);
    }

    HRESULT STDMETHODCALLTYPE SetUnknown(REFGUID guidKey, IUnknown* pUnknown) override {
        PROPVARIANT v{};
        v.vt = VT_UNKNOWN;
        v.punkVal = pUnknown;
        return SetItem(guidKey, v);
    }

    HRESULT STDMETHODCALLTYPE LockStore() override {
        AcquireSRWLockExclusive(&store_lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE UnlockStore() override {
        ReleaseSRWLockExclusive(&store_lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetCount(UINT32* pcItems) override {
        if (!pcItems) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        *pcItems = static_cast<UINT32>(items_.size());
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetItemByIndex(UINT32 unIndex, GUID* pguidKey, PROPVARIANT* pValue) override {
        if (!pguidKey) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        if (unIndex >= items_.size()) {
            ReleaseSRWLockShared(&lock_);
            return E_INVALIDARG;
        }
        auto it = items_.begin();
        std::advance(it, unIndex);
        *pguidKey = it->first;
        HRESULT hr = S_OK;
        if (pValue) hr = PropVariantCopy(pValue, &it->second);
        ReleaseSRWLockShared(&lock_);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE CopyAllItems(IMFAttributes* pDest) override {
        if (!pDest) return E_POINTER;
        AcquireSRWLockShared(&lock_);
        HRESULT hr = S_OK;
        for (const auto& kv : items_) {
            hr = pDest->SetItem(kv.first, kv.second);
            if (FAILED(hr)) break;
        }
        ReleaseSRWLockShared(&lock_);
        return hr;
    }

    // IMFMediaType methods
    HRESULT STDMETHODCALLTYPE GetMajorType(GUID* pguidMajorType) override {
        return GetGUID(kMfMtMajorType, pguidMajorType);
    }

    HRESULT STDMETHODCALLTYPE IsCompressedFormat(BOOL* pfCompressed) override {
        if (!pfCompressed) return E_POINTER;
        UINT32 all_samples_indep = TRUE;
        if (SUCCEEDED(GetUINT32(kMfMtAllSamplesIndependent, &all_samples_indep))) {
            *pfCompressed = all_samples_indep ? FALSE : TRUE;
        } else {
            *pfCompressed = FALSE;
        }
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE IsEqual(IMFMediaType* pIMediaType, DWORD* pdwFlags) override {
        if (!pIMediaType || !pdwFlags) return E_POINTER;
        *pdwFlags = 0;
        GUID mine{}, theirs{};
        if (SUCCEEDED(GetMajorType(&mine)) && SUCCEEDED(pIMediaType->GetMajorType(&theirs)) && IsEqualGUID(mine, theirs)) {
            *pdwFlags |= MF_MEDIATYPE_EQUAL_MAJOR_TYPES;
        }
        if (SUCCEEDED(GetGUID(kMfMtSubtype, &mine)) && SUCCEEDED(pIMediaType->GetGUID(kMfMtSubtype, &theirs)) &&
            IsEqualGUID(mine, theirs)) {
            *pdwFlags |= MF_MEDIATYPE_EQUAL_FORMAT_TYPES;
        }
        BOOL same_data = FALSE;
        if (SUCCEEDED(Compare(pIMediaType, MF_ATTRIBUTES_MATCH_ALL_ITEMS, &same_data)) && same_data) {
            *pdwFlags |= MF_MEDIATYPE_EQUAL_FORMAT_DATA;
        }
        return (*pdwFlags == (MF_MEDIATYPE_EQUAL_MAJOR_TYPES | MF_MEDIATYPE_EQUAL_FORMAT_TYPES |
                              MF_MEDIATYPE_EQUAL_FORMAT_DATA))
                   ? S_OK
                   : S_FALSE;
    }

    HRESULT STDMETHODCALLTYPE GetRepresentation(GUID, LPVOID* ppv) override {
        if (ppv) *ppv = nullptr;
        return MF_E_UNSUPPORTED_REPRESENTATION;
    }

    HRESULT STDMETHODCALLTYPE FreeRepresentation(GUID, LPVOID) override { return E_NOTIMPL; }

private:
    static MF_ATTRIBUTE_TYPE vt_to_attr_type(VARTYPE vt) {
        switch (vt) {
            case VT_UI4: return MF_ATTRIBUTE_UINT32;
            case VT_UI8: return MF_ATTRIBUTE_UINT64;
            case VT_R8: return MF_ATTRIBUTE_DOUBLE;
            case VT_CLSID: return MF_ATTRIBUTE_GUID;
            case VT_LPWSTR: return MF_ATTRIBUTE_STRING;
            case VT_VECTOR | VT_UI1: return MF_ATTRIBUTE_BLOB;
            case VT_UNKNOWN: return MF_ATTRIBUTE_IUNKNOWN;
            default: return static_cast<MF_ATTRIBUTE_TYPE>(0);
        }
    }

    static bool compare_propvar(const PROPVARIANT& a, const PROPVARIANT& b) {
        switch (a.vt) {
            case VT_UI4: return a.ulVal == b.ulVal;
            case VT_UI8: return a.uhVal.QuadPart == b.uhVal.QuadPart;
            case VT_R8: return a.dblVal == b.dblVal;
            case VT_CLSID: return a.puuid && b.puuid && IsEqualGUID(*a.puuid, *b.puuid);
            case VT_LPWSTR: return a.pwszVal && b.pwszVal && std::wcscmp(a.pwszVal, b.pwszVal) == 0;
            case VT_VECTOR | VT_UI1:
                return a.caub.cElems == b.caub.cElems &&
                       (a.caub.cElems == 0 || std::memcmp(a.caub.pElems, b.caub.pElems, a.caub.cElems) == 0);
            case VT_UNKNOWN: return a.punkVal == b.punkVal;
            default: return false;
        }
    }

    template <typename T>
    HRESULT get_typed(REFGUID guidKey, VARTYPE expected_vt, T& out) {
        AcquireSRWLockShared(&lock_);
        const auto it = items_.find(guidKey);
        if (it == items_.end()) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_ATTRIBUTENOTFOUND;
        }
        if (it->second.vt != expected_vt) {
            ReleaseSRWLockShared(&lock_);
            return MF_E_INVALIDTYPE;
        }
        if constexpr (sizeof(T) == 4) out = static_cast<T>(it->second.ulVal);
        else if constexpr (std::is_same_v<T, double>) out = it->second.dblVal;
        else out = static_cast<T>(it->second.uhVal.QuadPart);
        ReleaseSRWLockShared(&lock_);
        return S_OK;
    }

    volatile LONG refs_ = 1;
    bool is_media_type_;
    SRWLOCK lock_ = SRWLOCK_INIT;
    SRWLOCK store_lock_ = SRWLOCK_INIT;
    std::map<GUID, PROPVARIANT, GuidLess> items_;
};

// Wraps host IMFSourceResolver (when available) so console URLs (GAME:/G:/T:) resolve through rt::fs before opening.
class SourceResolverProxy final : public IMFSourceResolver {
public:
    explicit SourceResolverProxy(IMFSourceResolver* host) : host_(host) {
        if (host_) host_->AddRef();
    }

    ~SourceResolverProxy() {
        if (host_) host_->Release();
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        if (IsEqualIID(riid, __uuidof(IUnknown)) || IsEqualIID(riid, __uuidof(IMFSourceResolver))) {
            *ppv = static_cast<IMFSourceResolver*>(this);
            AddRef();
            return S_OK;
        }
        return host_ ? host_->QueryInterface(riid, ppv) : E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&refs_)); }

    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG n = static_cast<ULONG>(InterlockedDecrement(&refs_));
        if (n == 0) delete this;
        return n;
    }

    HRESULT STDMETHODCALLTYPE CreateObjectFromURL(LPCWSTR pwszURL, DWORD dwFlags, IPropertyStore* pProps,
                                                  MF_OBJECT_TYPE* pObjectType, IUnknown** ppObject) override {
        if (!ppObject) return E_POINTER;
        *ppObject = nullptr;
        if (pObjectType) *pObjectType = MF_OBJECT_INVALID;
        if (!pwszURL) return E_INVALIDARG;

        bool handled = false;
        std::wstring resolved;
        HRESULT resolve_hr = S_OK;
        const bool ok = resolve_console_media_url(pwszURL, handled, resolved, resolve_hr);
        if (handled && !ok) return resolve_hr;

        if (!host_) return MF_E_UNSUPPORTED_BYTESTREAM_TYPE;
        return host_->CreateObjectFromURL(handled ? resolved.c_str() : pwszURL, dwFlags, pProps, pObjectType, ppObject);
    }

    HRESULT STDMETHODCALLTYPE CreateObjectFromByteStream(IMFByteStream* pByteStream, LPCWSTR pwszURL, DWORD dwFlags,
                                                         IPropertyStore* pProps, MF_OBJECT_TYPE* pObjectType,
                                                         IUnknown** ppObject) override {
        if (!ppObject) return E_POINTER;
        *ppObject = nullptr;
        if (pObjectType) *pObjectType = MF_OBJECT_INVALID;
        if (!host_) return MF_E_UNSUPPORTED_BYTESTREAM_TYPE;
        return host_->CreateObjectFromByteStream(pByteStream, pwszURL, dwFlags, pProps, pObjectType, ppObject);
    }

    HRESULT STDMETHODCALLTYPE BeginCreateObjectFromURL(LPCWSTR pwszURL, DWORD dwFlags, IPropertyStore* pProps,
                                                       IUnknown** ppIUnknownCancelCookie, IMFAsyncCallback* pCallback,
                                                       IUnknown* punkState) override {
        if (ppIUnknownCancelCookie) *ppIUnknownCancelCookie = nullptr;
        if (!pwszURL) return E_INVALIDARG;

        bool handled = false;
        std::wstring resolved;
        HRESULT resolve_hr = S_OK;
        const bool ok = resolve_console_media_url(pwszURL, handled, resolved, resolve_hr);
        if (handled && !ok) return resolve_hr;

        if (!host_) return MF_E_UNSUPPORTED_BYTESTREAM_TYPE;
        return host_->BeginCreateObjectFromURL(handled ? resolved.c_str() : pwszURL, dwFlags, pProps,
                                               ppIUnknownCancelCookie, pCallback, punkState);
    }

    HRESULT STDMETHODCALLTYPE EndCreateObjectFromURL(IMFAsyncResult* pResult, MF_OBJECT_TYPE* pObjectType,
                                                     IUnknown** ppObject) override {
        if (!host_) return E_NOTIMPL;
        return host_->EndCreateObjectFromURL(pResult, pObjectType, ppObject);
    }

    HRESULT STDMETHODCALLTYPE BeginCreateObjectFromByteStream(IMFByteStream* pByteStream, LPCWSTR pwszURL, DWORD dwFlags,
                                                              IPropertyStore* pProps, IUnknown** ppIUnknownCancelCookie,
                                                              IMFAsyncCallback* pCallback,
                                                              IUnknown* punkState) override {
        if (ppIUnknownCancelCookie) *ppIUnknownCancelCookie = nullptr;
        if (!host_) return MF_E_UNSUPPORTED_BYTESTREAM_TYPE;
        return host_->BeginCreateObjectFromByteStream(pByteStream, pwszURL, dwFlags, pProps, ppIUnknownCancelCookie,
                                                      pCallback, punkState);
    }

    HRESULT STDMETHODCALLTYPE EndCreateObjectFromByteStream(IMFAsyncResult* pResult, MF_OBJECT_TYPE* pObjectType,
                                                            IUnknown** ppObject) override {
        if (!host_) return E_NOTIMPL;
        return host_->EndCreateObjectFromByteStream(pResult, pObjectType, ppObject);
    }

    HRESULT STDMETHODCALLTYPE CancelObjectCreation(IUnknown* pIUnknownCancelCookie) override {
        return host_ ? host_->CancelObjectCreation(pIUnknownCancelCookie) : E_NOTIMPL;
    }

private:
    volatile LONG refs_ = 1;
    IMFSourceResolver* host_ = nullptr;
};

HRESULT ensure_host_started(const HostMfApi& api) {
    if (api.offline_forced || !api.startup) return S_OK;
    AcquireSRWLockExclusive(&g_startup_lock);
    HRESULT hr = S_OK;
    if (!g_host_started) {
        hr = api.startup(MF_VERSION, MFSTARTUP_LITE);
        if (SUCCEEDED(hr)) g_host_started = true;
    }
    ReleaseSRWLockExclusive(&g_startup_lock);
    return hr;
}

HRESULT STDAPICALLTYPE mf_startup(ULONG version, DWORD flags) {
    const HostMfApi& api = host_api();
    AcquireSRWLockExclusive(&g_startup_lock);
    if (!api.offline_forced && api.startup && !g_host_started) {
        const HRESULT hr = api.startup(version, flags);
        if (FAILED(hr)) {
            ReleaseSRWLockExclusive(&g_startup_lock);
            return hr;
        }
        g_host_started = true;
    }
    ++g_guest_startup_depth;
    ReleaseSRWLockExclusive(&g_startup_lock);
    return S_OK;
}

HRESULT STDAPICALLTYPE mf_shutdown() {
    const HostMfApi& api = host_api();
    AcquireSRWLockExclusive(&g_startup_lock);
    if (g_guest_startup_depth == 0) {
        ReleaseSRWLockExclusive(&g_startup_lock);
        return MF_E_SHUTDOWN;
    }
    --g_guest_startup_depth;
    HRESULT hr = S_OK;
    if (g_guest_startup_depth == 0 && g_host_started && api.shutdown) {
        hr = api.shutdown();
        g_host_started = false;
    }
    ReleaseSRWLockExclusive(&g_startup_lock);
    return hr;
}

HRESULT STDAPICALLTYPE mf_create_attributes(IMFAttributes** ppMFAttributes, UINT32 cInitialSize) {
    if (!ppMFAttributes) return E_POINTER;
    *ppMFAttributes = nullptr;
    const HostMfApi& api = host_api();
    if (!api.offline_forced && api.create_attributes && SUCCEEDED(ensure_host_started(api))) {
        const HRESULT hr = api.create_attributes(ppMFAttributes, cInitialSize);
        if (SUCCEEDED(hr)) return hr;
    }
    auto* obj = new (std::nothrow) OfflineMediaType(false);
    if (!obj) return E_OUTOFMEMORY;
    *ppMFAttributes = static_cast<IMFAttributes*>(obj);
    return S_OK;
}

HRESULT STDAPICALLTYPE mf_create_media_type(IMFMediaType** ppMFMediaType) {
    if (!ppMFMediaType) return E_POINTER;
    *ppMFMediaType = nullptr;
    const HostMfApi& api = host_api();
    if (!api.offline_forced && api.create_media_type && SUCCEEDED(ensure_host_started(api))) {
        const HRESULT hr = api.create_media_type(ppMFMediaType);
        if (SUCCEEDED(hr)) return hr;
    }
    auto* obj = new (std::nothrow) OfflineMediaType(true);
    if (!obj) return E_OUTOFMEMORY;
    *ppMFMediaType = obj;
    return S_OK;
}

HRESULT STDAPICALLTYPE mf_create_source_resolver(IMFSourceResolver** ppISourceResolver) {
    if (!ppISourceResolver) return E_POINTER;
    *ppISourceResolver = nullptr;
    const HostMfApi& api = host_api();
    IMFSourceResolver* host_resolver = nullptr;
    if (!api.offline_forced && api.create_source_resolver && SUCCEEDED(ensure_host_started(api))) {
        api.create_source_resolver(&host_resolver);
    }
    auto* proxy = new (std::nothrow) SourceResolverProxy(host_resolver);
    if (host_resolver) host_resolver->Release();
    if (!proxy) return E_OUTOFMEMORY;
    *ppISourceResolver = proxy;
    return S_OK;
}

MFTIME STDAPICALLTYPE mf_get_system_time() {
    const HostMfApi& api = host_api();
    if (!api.offline_forced && api.get_system_time) return api.get_system_time();
    FILETIME ft{};
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER ui{};
    ui.LowPart = ft.dwLowDateTime;
    ui.HighPart = ft.dwHighDateTime;
    return static_cast<MFTIME>(ui.QuadPart);
}

HRESULT STDAPICALLTYPE mf_create_source_reader_from_media_source(IMFMediaSource* pMediaSource,
                                                                 IMFAttributes* pAttributes,
                                                                 IMFSourceReader** ppSourceReader) {
    if (!ppSourceReader) return E_POINTER;
    *ppSourceReader = nullptr;
    if (!pMediaSource) return E_INVALIDARG;
    const HostMfApi& api = host_api();
    if (!api.offline_forced && api.create_source_reader && SUCCEEDED(ensure_host_started(api))) {
        return api.create_source_reader(pMediaSource, pAttributes, ppSourceReader);
    }
    return MF_E_UNSUPPORTED_BYTESTREAM_TYPE;
}

}  // namespace
}  // namespace rt::mf

namespace {

using namespace rt::mf;

#define MFPLAT_API(dll)                                                                           \
    {dll, "MFStartup", reinterpret_cast<void*>(&mf_startup)},                                     \
        {dll, "MFShutdown", reinterpret_cast<void*>(&mf_shutdown)},                               \
        {dll, "MFCreateAttributes", reinterpret_cast<void*>(&mf_create_attributes)},              \
        {dll, "MFCreateMediaType", reinterpret_cast<void*>(&mf_create_media_type)},               \
        {dll, "MFCreateSourceResolver", reinterpret_cast<void*>(&mf_create_source_resolver)},     \
        {dll, "MFGetSystemTime", reinterpret_cast<void*>(&mf_get_system_time)}

}  // namespace

SHIM_TABLE(mf_offline,
    MFPLAT_API("mfplat.dll"),
    {"mfreadwrite.dll", "MFCreateSourceReaderFromMediaSource",
     reinterpret_cast<void*>(&mf_create_source_reader_from_media_source)},
);
