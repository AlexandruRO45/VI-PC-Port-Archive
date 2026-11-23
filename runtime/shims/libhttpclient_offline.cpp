// libhttpclient_offline: offline Microsoft libHttpClient (libHttpClient.GDK.dll) shim for mapped titles.
//
// Nothing here touches the network. HCHttpCallCreate/Request/Response manage local refcounted call state,
// HCHttpCallPerformAsync completes the caller's XAsyncBlock on xtaskqueue with no network I/O and reports
// ERROR_WINHTTP_CANNOT_CONNECT via HCHttpCallResponseGetNetworkErrorCode (matching winhttp_offline), and
// HCInitialize / HCCleanupAsync track library initialization refcounts cleanly.
// Signatures follow Microsoft's public MIT-licensed libHttpClient C API (httpClient/httpClient.h, trace.h).
#include <windows.h>

#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <XAsync.h>
#include <XAsyncProvider.h>

#include "shims.h"

namespace rt::libhttpclient {
namespace {

constexpr DWORD kErrorWinHttpCannotConnect = 12029;
constexpr HRESULT kHrCannotConnect = HRESULT_FROM_WIN32(kErrorWinHttpCannotConnect);

struct Call;
using HCCallHandle = Call*;
struct HCInitArgs;
struct HCTraceArea;

struct Header {
    std::string name;
    std::string value;
};

struct Call {
    std::atomic<long> refs{1};
    std::mutex mutex;
    std::string method = "GET";
    std::string url;
    std::vector<Header> request_headers;
    std::vector<std::uint8_t> request_body;
    std::string request_body_str;
    bool retry_allowed = true;
    std::uint32_t retry_cache_id = 0;
    std::uint32_t retry_delay = 2;
    std::uint32_t timeout = 30;
    std::uint32_t timeout_window = 20;
    bool trace_call = true;

    std::uint32_t status_code = 0;
    HRESULT network_error_code = S_OK;
    std::uint32_t platform_network_error_code = 0;
    std::string platform_error_msg;
    std::string response_string;
    std::vector<std::uint8_t> response_body;
    std::vector<Header> response_headers;
};

// Intentionally immortal: mapped images can clean up during DLL_PROCESS_DETACH after normal statics run.
std::mutex& live_mutex() {
    static auto* m = new std::mutex;
    return *m;
}

std::set<Call*>& live_calls() {
    static auto* s = new std::set<Call*>;
    return *s;
}

std::mutex& log_mutex() {
    static auto* m = new std::mutex;
    return *m;
}

std::set<std::string>& logged_urls() {
    static auto* s = new std::set<std::string>;
    return *s;
}

std::atomic<long>& init_refs() {
    static auto* r = new std::atomic<long>{0};
    return *r;
}

bool equals_ascii_nocase(const std::string& a, const char* b) {
    if (!b) return false;
    const size_t n = std::strlen(b);
    if (a.size() != n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    }
    return true;
}

Call* retain_call(HCCallHandle handle) {
    if (!handle) return nullptr;
    std::lock_guard lock(live_mutex());
    if (live_calls().count(handle) == 0) return nullptr;
    handle->refs.fetch_add(1, std::memory_order_relaxed);
    return handle;
}

void release_call(Call* call) {
    if (!call) return;
    if (call->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        {
            std::lock_guard lock(live_mutex());
            live_calls().erase(call);
        }
        delete call;
    }
}

void log_refused_once(const std::string& url) {
    std::lock_guard lock(log_mutex());
    const std::string key = url.empty() ? std::string("<empty>") : url;
    if (logged_urls().insert(key).second)
        std::fprintf(stderr, "[rt] libhttpclient: offline, refused %s\n", key.c_str());
}

struct PerformContext {
    Call* call = nullptr;
};

HRESULT CALLBACK perform_async_callback(XAsyncOp op, const XAsyncProviderData* data) {
    auto* ctx = static_cast<PerformContext*>(data->context);
    switch (op) {
    case XAsyncOp::DoWork: {
        std::string url;
        {
            std::lock_guard lock(ctx->call->mutex);
            url = ctx->call->url;
            ctx->call->status_code = 0;
            ctx->call->network_error_code = kHrCannotConnect;
            ctx->call->platform_network_error_code = kErrorWinHttpCannotConnect;
            ctx->call->platform_error_msg = "offline: network disabled by Leonida runtime";
            ctx->call->response_string.clear();
            ctx->call->response_body.clear();
            ctx->call->response_headers.clear();
        }
        log_refused_once(url);
        XAsyncComplete(data->async, S_OK, 0);
        return S_OK;
    }
    case XAsyncOp::Cleanup:
        release_call(ctx->call);
        delete ctx;
        return S_OK;
    default:
        return S_OK;
    }
}

HRESULT WINAPI shim_HCInitialize(HCInitArgs*) {
    init_refs().fetch_add(1, std::memory_order_relaxed);
    return S_OK;
}

HRESULT WINAPI shim_HCCleanupAsync(XAsyncBlock* async) {
    if (!async) return E_INVALIDARG;
    long prev = init_refs().load(std::memory_order_relaxed);
    while (prev > 0 && !init_refs().compare_exchange_weak(prev, prev - 1, std::memory_order_acq_rel)) {
    }
    return XAsyncRun(async, [](XAsyncBlock*) -> HRESULT { return S_OK; });
}

HRESULT WINAPI shim_HCSettingsGetTraceLevel(std::uint32_t* trace_level) {
    if (!trace_level) return E_INVALIDARG;
    *trace_level = 0;  // HCTraceLevel::Off
    return S_OK;
}

void WINAPI shim_HCTraceSetEtwEnabled(bool) {}

void WINAPIV shim_HCTraceImplMessage(HCTraceArea*, std::uint32_t, const char*, ...) {}

HRESULT WINAPI shim_HCHttpCallSetTracing(HCCallHandle handle, bool trace_call) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        c->trace_call = trace_call;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallCreate(HCCallHandle* out) {
    if (!out) return E_INVALIDARG;
    auto* c = new Call();
    {
        std::lock_guard lock(live_mutex());
        live_calls().insert(c);
    }
    *out = c;
    return S_OK;
}

HCCallHandle WINAPI shim_HCHttpCallDuplicateHandle(HCCallHandle handle) {
    return retain_call(handle);
}

HRESULT WINAPI shim_HCHttpCallCloseHandle(HCCallHandle handle) {
    if (!handle) return E_INVALIDARG;
    {
        std::lock_guard lock(live_mutex());
        if (live_calls().count(handle) == 0) return E_INVALIDARG;
    }
    release_call(handle);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestSetUrl(HCCallHandle handle, const char* method, const char* url) {
    if (!method || !url) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        c->method = method;
        c->url = url;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestGetUrl(HCCallHandle handle, const char** method, const char** url) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        if (method) *method = c->method.c_str();
        if (url) *url = c->url.c_str();
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallGetRequestUrl(HCCallHandle handle, const char** url) {
    if (!url) return E_INVALIDARG;
    return shim_HCHttpCallRequestGetUrl(handle, nullptr, url);
}

HRESULT WINAPI shim_HCHttpCallRequestSetHeader(HCCallHandle handle, const char* name, const char* value, bool) {
    if (!name || !value) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        bool updated = false;
        for (auto& h : c->request_headers) {
            if (equals_ascii_nocase(h.name, name)) {
                h.name = name;
                h.value = value;
                updated = true;
                break;
            }
        }
        if (!updated) c->request_headers.push_back({name, value});
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestGetNumHeaders(HCCallHandle handle, std::uint32_t* num_headers) {
    if (!num_headers) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *num_headers = static_cast<std::uint32_t>(c->request_headers.size());
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestGetHeaderAtIndex(HCCallHandle handle, std::uint32_t index, const char** name,
                                                     const char** value) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    HRESULT hr = S_OK;
    {
        std::lock_guard lock(c->mutex);
        if (index >= c->request_headers.size()) {
            hr = E_INVALIDARG;
        } else {
            if (name) *name = c->request_headers[index].name.c_str();
            if (value) *value = c->request_headers[index].value.c_str();
        }
    }
    release_call(c);
    return hr;
}

HRESULT WINAPI shim_HCHttpCallRequestSetRequestBodyBytes(HCCallHandle handle, const std::uint8_t* bytes,
                                                         std::uint32_t size) {
    if (size > 0 && !bytes) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        if (size == 0) {
            c->request_body.clear();
            c->request_body_str.clear();
        } else {
            c->request_body.assign(bytes, bytes + size);
            c->request_body_str.assign(reinterpret_cast<const char*>(bytes), size);
        }
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestSetRequestBodyString(HCCallHandle handle, const char* body) {
    if (!body) return E_INVALIDARG;
    const size_t len = std::strlen(body);
    return shim_HCHttpCallRequestSetRequestBodyBytes(
        handle, reinterpret_cast<const std::uint8_t*>(body), static_cast<std::uint32_t>(len));
}

HRESULT WINAPI shim_HCHttpCallRequestGetRequestBodyBytes(HCCallHandle handle, const std::uint8_t** bytes,
                                                         std::uint32_t* size) {
    if (!bytes || !size) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *size = static_cast<std::uint32_t>(c->request_body.size());
        *bytes = c->request_body.empty() ? nullptr : c->request_body.data();
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestSetRetryAllowed(HCCallHandle handle, bool retry_allowed) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        c->retry_allowed = retry_allowed;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestGetRetryAllowed(HCCallHandle handle, bool* retry_allowed) {
    if (!retry_allowed) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *retry_allowed = c->retry_allowed;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestSetRetryCacheId(HCCallHandle handle, std::uint32_t id) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        c->retry_cache_id = id;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestGetRetryCacheId(HCCallHandle handle, std::uint32_t* id) {
    if (!id) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *id = c->retry_cache_id;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestSetRetryDelay(HCCallHandle handle, std::uint32_t seconds) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        c->retry_delay = seconds;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestGetRetryDelay(HCCallHandle handle, std::uint32_t* seconds) {
    if (!seconds) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *seconds = c->retry_delay;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestSetTimeout(HCCallHandle handle, std::uint32_t seconds) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        c->timeout = seconds;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestGetTimeout(HCCallHandle handle, std::uint32_t* seconds) {
    if (!seconds) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *seconds = c->timeout;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestSetTimeoutWindow(HCCallHandle handle, std::uint32_t seconds) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        c->timeout_window = seconds;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallRequestGetTimeoutWindow(HCCallHandle handle, std::uint32_t* seconds) {
    if (!seconds) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *seconds = c->timeout_window;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallPerformAsync(HCCallHandle handle, XAsyncBlock* async) {
    if (!async) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    auto* ctx = new PerformContext{c};
    HRESULT hr = XAsyncBegin(async, ctx, reinterpret_cast<const void*>(&shim_HCHttpCallPerformAsync),
                             "HCHttpCallPerformAsync", perform_async_callback);
    if (FAILED(hr)) {
        release_call(c);
        delete ctx;
        return hr;
    }
    hr = XAsyncSchedule(async, 0);
    if (FAILED(hr)) {
        // Cleanup callback will run when XAsyncBegin's state tears down if needed, or we return hr.
        return hr;
    }
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallResponseGetStatusCode(HCCallHandle handle, std::uint32_t* status_code) {
    if (!status_code) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *status_code = c->status_code;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallResponseGetNetworkErrorCode(HCCallHandle handle, HRESULT* network_error_code,
                                                          std::uint32_t* platform_network_error_code) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        if (network_error_code) *network_error_code = c->network_error_code;
        if (platform_network_error_code) *platform_network_error_code = c->platform_network_error_code;
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallResponseGetPlatformNetworkErrorMessage(HCCallHandle handle, const char** message) {
    if (!message) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *message = c->platform_error_msg.c_str();
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallResponseGetResponseString(HCCallHandle handle, const char** response_string) {
    if (!response_string) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *response_string = c->response_string.c_str();
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallResponseGetResponseBodyBytesSize(HCCallHandle handle, size_t* buffer_size) {
    if (!buffer_size) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *buffer_size = c->response_body.size();
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallResponseGetResponseBodyBytes(HCCallHandle handle, size_t buffer_size,
                                                          std::uint8_t* buffer, size_t* buffer_used) {
    if (buffer_size > 0 && !buffer) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    HRESULT hr = S_OK;
    {
        std::lock_guard lock(c->mutex);
        if (buffer_size < c->response_body.size()) {
            hr = E_INVALIDARG;
        } else {
            if (!c->response_body.empty())
                std::memcpy(buffer, c->response_body.data(), c->response_body.size());
            if (buffer_used) *buffer_used = c->response_body.size();
        }
    }
    release_call(c);
    return hr;
}

HRESULT WINAPI shim_HCHttpCallResponseGetNumHeaders(HCCallHandle handle, std::uint32_t* num_headers) {
    if (!num_headers) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *num_headers = static_cast<std::uint32_t>(c->response_headers.size());
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallResponseGetHeader(HCCallHandle handle, const char* header_name,
                                                const char** header_value) {
    if (!header_name || !header_value) return E_INVALIDARG;
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    {
        std::lock_guard lock(c->mutex);
        *header_value = nullptr;
        for (const auto& h : c->response_headers) {
            if (equals_ascii_nocase(h.name, header_name)) {
                *header_value = h.value.c_str();
                break;
            }
        }
    }
    release_call(c);
    return S_OK;
}

HRESULT WINAPI shim_HCHttpCallResponseGetHeaderAtIndex(HCCallHandle handle, std::uint32_t index, const char** name,
                                                      const char** value) {
    Call* c = retain_call(handle);
    if (!c) return E_INVALIDARG;
    HRESULT hr = S_OK;
    {
        std::lock_guard lock(c->mutex);
        if (index >= c->response_headers.size()) {
            hr = E_INVALIDARG;
        } else {
            if (name) *name = c->response_headers[index].name.c_str();
            if (value) *value = c->response_headers[index].value.c_str();
        }
    }
    release_call(c);
    return hr;
}

#define HC(fn) {"libhttpclient.gdk.dll", #fn, reinterpret_cast<void*>(&shim_##fn)}

SHIM_TABLE(libhttpclient,
           HC(HCCleanupAsync),
           HC(HCHttpCallCloseHandle),
           HC(HCHttpCallCreate),
           HC(HCHttpCallDuplicateHandle),
           HC(HCHttpCallGetRequestUrl),
           HC(HCHttpCallPerformAsync),
           HC(HCHttpCallRequestGetHeaderAtIndex),
           HC(HCHttpCallRequestGetNumHeaders),
           HC(HCHttpCallRequestGetRequestBodyBytes),
           HC(HCHttpCallRequestGetRetryAllowed),
           HC(HCHttpCallRequestGetRetryCacheId),
           HC(HCHttpCallRequestGetRetryDelay),
           HC(HCHttpCallRequestGetTimeout),
           HC(HCHttpCallRequestGetTimeoutWindow),
           HC(HCHttpCallRequestGetUrl),
           HC(HCHttpCallRequestSetHeader),
           HC(HCHttpCallRequestSetRequestBodyBytes),
           HC(HCHttpCallRequestSetRequestBodyString),
           HC(HCHttpCallRequestSetRetryAllowed),
           HC(HCHttpCallRequestSetRetryCacheId),
           HC(HCHttpCallRequestSetRetryDelay),
           HC(HCHttpCallRequestSetTimeout),
           HC(HCHttpCallRequestSetTimeoutWindow),
           HC(HCHttpCallRequestSetUrl),
           HC(HCHttpCallResponseGetHeader),
           HC(HCHttpCallResponseGetHeaderAtIndex),
           HC(HCHttpCallResponseGetNetworkErrorCode),
           HC(HCHttpCallResponseGetNumHeaders),
           HC(HCHttpCallResponseGetPlatformNetworkErrorMessage),
           HC(HCHttpCallResponseGetResponseBodyBytes),
           HC(HCHttpCallResponseGetResponseBodyBytesSize),
           HC(HCHttpCallResponseGetResponseString),
           HC(HCHttpCallResponseGetStatusCode),
           HC(HCHttpCallSetTracing),
           HC(HCInitialize),
           HC(HCSettingsGetTraceLevel),
           HC(HCTraceImplMessage),
           HC(HCTraceSetEtwEnabled));

#undef HC

}  // namespace
}  // namespace rt::libhttpclient
