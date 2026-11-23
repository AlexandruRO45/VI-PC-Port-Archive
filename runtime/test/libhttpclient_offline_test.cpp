// Synthetic unit tests for the offline libHttpClient.GDK.dll shim.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <XAsync.h>
#include <XTaskQueue.h>

#include "shims.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class Fn>
Fn lookup(const char* name) {
    void* p = rt::shims::find("libhttpclient.gdk.dll", name);
    if (!p) {
        std::fprintf(stderr, "FAIL: missing shim %s\n", name);
        ++g_failures;
    }
    return reinterpret_cast<Fn>(p);
}

using FnInit = HRESULT(WINAPI*)(void*);
using FnCleanupAsync = HRESULT(WINAPI*)(XAsyncBlock*);
using FnGetTraceLevel = HRESULT(WINAPI*)(std::uint32_t*);
using FnSetEtw = void(WINAPI*)(bool);
using FnTraceMsg = void(WINAPIV*)(void*, std::uint32_t, const char*, ...);
using FnSetTracing = HRESULT(WINAPI*)(void*, bool);
using FnCreate = HRESULT(WINAPI*)(void**);
using FnDup = void*(WINAPI*)(void*);
using FnClose = HRESULT(WINAPI*)(void*);
using FnSetUrl = HRESULT(WINAPI*)(void*, const char*, const char*);
using FnGetReqUrl = HRESULT(WINAPI*)(void*, const char**, const char**);
using FnGetUrl = HRESULT(WINAPI*)(void*, const char**);
using FnSetHeader = HRESULT(WINAPI*)(void*, const char*, const char*, bool);
using FnNumHeaders = HRESULT(WINAPI*)(void*, std::uint32_t*);
using FnHeaderAt = HRESULT(WINAPI*)(void*, std::uint32_t, const char**, const char**);
using FnSetBodyBytes = HRESULT(WINAPI*)(void*, const std::uint8_t*, std::uint32_t);
using FnSetBodyStr = HRESULT(WINAPI*)(void*, const char*);
using FnGetBodyBytes = HRESULT(WINAPI*)(void*, const std::uint8_t**, std::uint32_t*);
using FnSetBool = HRESULT(WINAPI*)(void*, bool);
using FnGetBool = HRESULT(WINAPI*)(void*, bool*);
using FnSetU32 = HRESULT(WINAPI*)(void*, std::uint32_t);
using FnGetU32 = HRESULT(WINAPI*)(void*, std::uint32_t*);
using FnPerformAsync = HRESULT(WINAPI*)(void*, XAsyncBlock*);
using FnGetNetErr = HRESULT(WINAPI*)(void*, HRESULT*, std::uint32_t*);
using FnGetErrMsg = HRESULT(WINAPI*)(void*, const char**);
using FnGetRespStr = HRESULT(WINAPI*)(void*, const char**);
using FnGetRespSize = HRESULT(WINAPI*)(void*, size_t*);
using FnGetRespBytes = HRESULT(WINAPI*)(void*, size_t, std::uint8_t*, size_t*);
using FnGetRespHeader = HRESULT(WINAPI*)(void*, const char*, const char**);

}  // namespace

int main() {
    check(rt::shims::owns_dll("libhttpclient.gdk.dll"), "owns libhttpclient.gdk.dll");

    auto init = lookup<FnInit>("HCInitialize");
    auto cleanup = lookup<FnCleanupAsync>("HCCleanupAsync");
    auto get_trace = lookup<FnGetTraceLevel>("HCSettingsGetTraceLevel");
    auto set_etw = lookup<FnSetEtw>("HCTraceSetEtwEnabled");
    auto trace_msg = lookup<FnTraceMsg>("HCTraceImplMessage");
    auto set_tracing = lookup<FnSetTracing>("HCHttpCallSetTracing");
    auto create = lookup<FnCreate>("HCHttpCallCreate");
    auto dup = lookup<FnDup>("HCHttpCallDuplicateHandle");
    auto close = lookup<FnClose>("HCHttpCallCloseHandle");
    auto set_url = lookup<FnSetUrl>("HCHttpCallRequestSetUrl");
    auto get_req_url = lookup<FnGetReqUrl>("HCHttpCallRequestGetUrl");
    auto get_url = lookup<FnGetUrl>("HCHttpCallGetRequestUrl");
    auto set_header = lookup<FnSetHeader>("HCHttpCallRequestSetHeader");
    auto get_num_headers = lookup<FnNumHeaders>("HCHttpCallRequestGetNumHeaders");
    auto get_header_at = lookup<FnHeaderAt>("HCHttpCallRequestGetHeaderAtIndex");
    auto set_body_bytes = lookup<FnSetBodyBytes>("HCHttpCallRequestSetRequestBodyBytes");
    auto set_body_str = lookup<FnSetBodyStr>("HCHttpCallRequestSetRequestBodyString");
    auto get_body_bytes = lookup<FnGetBodyBytes>("HCHttpCallRequestGetRequestBodyBytes");
    auto set_retry_allowed = lookup<FnSetBool>("HCHttpCallRequestSetRetryAllowed");
    auto get_retry_allowed = lookup<FnGetBool>("HCHttpCallRequestGetRetryAllowed");
    auto set_retry_id = lookup<FnSetU32>("HCHttpCallRequestSetRetryCacheId");
    auto get_retry_id = lookup<FnGetU32>("HCHttpCallRequestGetRetryCacheId");
    auto set_retry_delay = lookup<FnSetU32>("HCHttpCallRequestSetRetryDelay");
    auto get_retry_delay = lookup<FnGetU32>("HCHttpCallRequestGetRetryDelay");
    auto set_timeout = lookup<FnSetU32>("HCHttpCallRequestSetTimeout");
    auto get_timeout = lookup<FnGetU32>("HCHttpCallRequestGetTimeout");
    auto set_timeout_win = lookup<FnSetU32>("HCHttpCallRequestSetTimeoutWindow");
    auto get_timeout_win = lookup<FnGetU32>("HCHttpCallRequestGetTimeoutWindow");
    auto perform = lookup<FnPerformAsync>("HCHttpCallPerformAsync");
    auto get_status = lookup<FnGetU32>("HCHttpCallResponseGetStatusCode");
    auto get_net_err = lookup<FnGetNetErr>("HCHttpCallResponseGetNetworkErrorCode");
    auto get_err_msg = lookup<FnGetErrMsg>("HCHttpCallResponseGetPlatformNetworkErrorMessage");
    auto get_resp_str = lookup<FnGetRespStr>("HCHttpCallResponseGetResponseString");
    auto get_resp_size = lookup<FnGetRespSize>("HCHttpCallResponseGetResponseBodyBytesSize");
    auto get_resp_bytes = lookup<FnGetRespBytes>("HCHttpCallResponseGetResponseBodyBytes");
    auto get_resp_num_headers = lookup<FnNumHeaders>("HCHttpCallResponseGetNumHeaders");
    auto get_resp_header = lookup<FnGetRespHeader>("HCHttpCallResponseGetHeader");
    auto get_resp_header_at = lookup<FnHeaderAt>("HCHttpCallResponseGetHeaderAtIndex");

    if (g_failures) return 1;

    XTaskQueueHandle queue = nullptr;
    check(SUCCEEDED(XTaskQueueCreate(XTaskQueueDispatchMode::ThreadPool, XTaskQueueDispatchMode::ThreadPool, &queue)),
          "create task queue");

    check(SUCCEEDED(init(nullptr)), "HCInitialize");
    std::uint32_t trace_lvl = 99;
    check(SUCCEEDED(get_trace(&trace_lvl)) && trace_lvl == 0, "HCSettingsGetTraceLevel");
    set_etw(false);
    trace_msg(nullptr, 0, "test %d", 1);

    void* call = nullptr;
    check(SUCCEEDED(create(&call)) && call != nullptr, "HCHttpCallCreate");
    check(SUCCEEDED(set_tracing(call, false)), "HCHttpCallSetTracing");

    check(SUCCEEDED(set_url(call, "POST", "https://example.invalid/api/test")), "HCHttpCallRequestSetUrl");
    const char* method = nullptr;
    const char* url = nullptr;
    check(SUCCEEDED(get_req_url(call, &method, &url)) && std::strcmp(method, "POST") == 0 &&
              std::strcmp(url, "https://example.invalid/api/test") == 0,
          "HCHttpCallRequestGetUrl");
    const char* url2 = nullptr;
    check(SUCCEEDED(get_url(call, &url2)) && std::strcmp(url2, "https://example.invalid/api/test") == 0,
          "HCHttpCallGetRequestUrl");

    check(SUCCEEDED(set_header(call, "Content-Type", "application/json", true)), "SetHeader 1");
    check(SUCCEEDED(set_header(call, "content-type", "text/plain", false)), "SetHeader overwrite");
    std::uint32_t nh = 0;
    check(SUCCEEDED(get_num_headers(call, &nh)) && nh == 1, "GetNumHeaders == 1");
    const char* hn = nullptr;
    const char* hv = nullptr;
    check(SUCCEEDED(get_header_at(call, 0, &hn, &hv)) && std::strcmp(hv, "text/plain") == 0, "GetHeaderAtIndex");
    check(get_header_at(call, 1, &hn, &hv) == E_INVALIDARG, "GetHeaderAtIndex OOB");

    check(SUCCEEDED(set_body_str(call, "hello")), "SetRequestBodyString");
    const std::uint8_t* body_ptr = nullptr;
    std::uint32_t body_sz = 0;
    check(SUCCEEDED(get_body_bytes(call, &body_ptr, &body_sz)) && body_sz == 5 &&
              std::memcmp(body_ptr, "hello", 5) == 0,
          "GetRequestBodyBytes after string");
    const std::uint8_t raw[] = {1, 2, 3};
    check(SUCCEEDED(set_body_bytes(call, raw, 3)), "SetRequestBodyBytes");
    check(SUCCEEDED(get_body_bytes(call, &body_ptr, &body_sz)) && body_sz == 3 && body_ptr[2] == 3,
          "GetRequestBodyBytes after bytes");

    bool retry = true;
    std::uint32_t u32 = 0;
    check(SUCCEEDED(set_retry_allowed(call, false)) && SUCCEEDED(get_retry_allowed(call, &retry)) && !retry,
          "RetryAllowed");
    check(SUCCEEDED(set_retry_id(call, 7)) && SUCCEEDED(get_retry_id(call, &u32)) && u32 == 7, "RetryCacheId");
    check(SUCCEEDED(set_retry_delay(call, 9)) && SUCCEEDED(get_retry_delay(call, &u32)) && u32 == 9, "RetryDelay");
    check(SUCCEEDED(set_timeout(call, 15)) && SUCCEEDED(get_timeout(call, &u32)) && u32 == 15, "Timeout");
    check(SUCCEEDED(set_timeout_win(call, 45)) && SUCCEEDED(get_timeout_win(call, &u32)) && u32 == 45,
          "TimeoutWindow");

    void* call_dup = dup(call);
    check(call_dup == call, "HCHttpCallDuplicateHandle");
    check(SUCCEEDED(close(call_dup)), "close dup");

    XAsyncBlock async{};
    async.queue = queue;
    check(SUCCEEDED(perform(call, &async)), "HCHttpCallPerformAsync");
    check(SUCCEEDED(XAsyncGetStatus(&async, true)), "XAsyncGetStatus completed");

    std::uint32_t status_code = 200;
    HRESULT net_hr = S_OK;
    std::uint32_t plat_err = 0;
    const char* err_msg = nullptr;
    const char* resp_str = nullptr;
    size_t resp_sz = 99;
    size_t resp_used = 99;
    std::uint8_t buf[8]{};
    const char* rh = "sentinel";

    check(SUCCEEDED(get_status(call, &status_code)) && status_code == 0, "ResponseGetStatusCode == 0");
    check(SUCCEEDED(get_net_err(call, &net_hr, &plat_err)) && net_hr == HRESULT_FROM_WIN32(12029) && plat_err == 12029,
          "ResponseGetNetworkErrorCode == ERROR_WINHTTP_CANNOT_CONNECT");
    check(SUCCEEDED(get_err_msg(call, &err_msg)) && err_msg && err_msg[0] != '\0',
          "ResponseGetPlatformNetworkErrorMessage");
    check(SUCCEEDED(get_resp_str(call, &resp_str)) && resp_str && resp_str[0] == '\0', "ResponseGetResponseString");
    check(SUCCEEDED(get_resp_size(call, &resp_sz)) && resp_sz == 0, "ResponseGetResponseBodyBytesSize");
    check(SUCCEEDED(get_resp_bytes(call, sizeof(buf), buf, &resp_used)) && resp_used == 0,
          "ResponseGetResponseBodyBytes");
    check(SUCCEEDED(get_resp_num_headers(call, &nh)) && nh == 0, "ResponseGetNumHeaders");
    check(SUCCEEDED(get_resp_header(call, "X-Test", &rh)) && rh == nullptr, "ResponseGetHeader");
    check(get_resp_header_at(call, 0, &hn, &hv) == E_INVALIDARG, "ResponseGetHeaderAtIndex OOB");

    check(SUCCEEDED(close(call)), "HCHttpCallCloseHandle");
    check(close(call) == E_INVALIDARG, "double close rejected");

    XAsyncBlock cleanup_async{};
    cleanup_async.queue = queue;
    check(SUCCEEDED(cleanup(&cleanup_async)), "HCCleanupAsync");
    check(SUCCEEDED(XAsyncGetStatus(&cleanup_async, true)), "HCCleanupAsync status");

    XTaskQueueTerminate(queue, true, nullptr, nullptr);
    XTaskQueueCloseHandle(queue);

    std::printf("libhttpclient_offline_test: %d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
