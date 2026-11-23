// crypt32_offline: CryptoAPI certificate-store shims for mapped titles, with no host store access.
//
// The package's XCurl.dll and libHttpClient.GDK.dll import certificate-store functions from crypt32.dll
// (CertOpenStore, CertOpenSystemStoreW, CertFindCertificateInStore, CertFindChainInStore,
// CertFreeCertificateContext, CertFreeCertificateChain, CertCloseStore, CryptStringToBinaryW). Bound to the host's
// crypt32.dll, a mapped title can open and enumerate the user's own certificate stores. Because nothing here goes
// online, crypt32.dll is owned by the shims:
//   - CertOpenStore and CertOpenSystemStoreW never open a host store; they log once and fail with
//     ERROR_ACCESS_DENIED.
//   - CertFindCertificateInStore and CertFindChainInStore fail with CRYPT_E_NOT_FOUND without touching host stores.
//   - CertFreeCertificateContext, CertFreeCertificateChain, and CertCloseStore accept null/ignored handles safely
//     without calling the host.
//   - CryptStringToBinaryW only decodes an in-memory string (base64/hex), so it delegates to the host's crypt32.dll
//     loaded from System32 on first use.
#include <windows.h>
#include <wincrypt.h>

#include <atomic>
#include <cstdio>
#include <type_traits>

#include "shims.h"

namespace rt::crypt32 {
namespace {

using OpenStoreFn = HCERTSTORE(WINAPI*)(LPCSTR, DWORD, HCRYPTPROV_LEGACY, DWORD, const void*);
using OpenSystemStoreWFn = HCERTSTORE(WINAPI*)(HCRYPTPROV_LEGACY, LPCWSTR);
using FindCertificateInStoreFn = PCCERT_CONTEXT(WINAPI*)(HCERTSTORE, DWORD, DWORD, DWORD, const void*,
                                                         PCCERT_CONTEXT);
using FindChainInStoreFn = BOOL(WINAPI*)(HCERTSTORE, DWORD, DWORD, DWORD, const void*, PCCERT_CHAIN_CONTEXT,
                                         PCCERT_CHAIN_CONTEXT*);
using FreeCertificateContextFn = BOOL(WINAPI*)(PCCERT_CONTEXT);
using FreeCertificateChainFn = void(WINAPI*)(PCCERT_CHAIN_CONTEXT);
using CloseStoreFn = BOOL(WINAPI*)(HCERTSTORE, DWORD);
using StringToBinaryWFn = BOOL(WINAPI*)(LPCWSTR, DWORD, DWORD, BYTE*, DWORD*, DWORD*, DWORD*);

HCERTSTORE WINAPI open_store(LPCSTR /*store_provider*/, DWORD /*encoding_type*/, HCRYPTPROV_LEGACY /*crypt_prov*/,
                             DWORD /*flags*/, const void* /*para*/) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] crypt32: offline, refused CertOpenStore\n");
    SetLastError(ERROR_ACCESS_DENIED);
    return nullptr;
}

HCERTSTORE WINAPI open_system_store_w(HCRYPTPROV_LEGACY /*prov*/, LPCWSTR subsystem_protocol) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fwprintf(stderr, L"[rt] crypt32: offline, refused CertOpenSystemStoreW(%ls)\n",
                      subsystem_protocol ? subsystem_protocol : L"(null)");
    }
    SetLastError(ERROR_ACCESS_DENIED);
    return nullptr;
}

PCCERT_CONTEXT WINAPI find_certificate_in_store(HCERTSTORE /*cert_store*/, DWORD /*encoding_type*/,
                                                DWORD /*find_flags*/, DWORD /*find_type*/, const void* /*find_para*/,
                                                PCCERT_CONTEXT /*prev_cert_context*/) {
    SetLastError(static_cast<DWORD>(CRYPT_E_NOT_FOUND));
    return nullptr;
}

BOOL WINAPI find_chain_in_store(HCERTSTORE /*cert_store*/, DWORD /*encoding_type*/, DWORD /*find_flags*/,
                                DWORD /*find_type*/, const void* /*find_para*/,
                                PCCERT_CHAIN_CONTEXT /*prev_chain_context*/,
                                PCCERT_CHAIN_CONTEXT* chain_context) {
    if (chain_context) *chain_context = nullptr;
    SetLastError(static_cast<DWORD>(CRYPT_E_NOT_FOUND));
    return FALSE;
}

BOOL WINAPI free_certificate_context(PCCERT_CONTEXT /*cert_context*/) { return TRUE; }

void WINAPI free_certificate_chain(PCCERT_CHAIN_CONTEXT /*chain_context*/) {}

BOOL WINAPI close_store(HCERTSTORE cert_store, DWORD /*flags*/) {
    if (!cert_store) {
        SetLastError(ERROR_INVALID_HANDLE);
        return FALSE;
    }
    return TRUE;
}

// Pure string-to-bytes decoding (base64/hex): load host crypt32.dll from System32 on first use so the runtime does
// not link crypt32.lib directly.
BOOL WINAPI string_to_binary_w(LPCWSTR psz_string, DWORD cch_string, DWORD dw_flags, BYTE* pb_binary,
                               DWORD* pcb_binary, DWORD* pdw_skip, DWORD* pdw_flags) {
    static auto host_fn = []() -> StringToBinaryWFn {
        const HMODULE m = LoadLibraryExW(L"crypt32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        return m ? reinterpret_cast<StringToBinaryWFn>(GetProcAddress(m, "CryptStringToBinaryW")) : nullptr;
    }();
    if (!host_fn) {
        SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
        return FALSE;
    }
    return host_fn(psz_string, cch_string, dw_flags, pb_binary, pcb_binary, pdw_skip, pdw_flags);
}

static_assert(std::is_same_v<decltype(&open_store), OpenStoreFn>);
static_assert(std::is_same_v<decltype(&open_system_store_w), OpenSystemStoreWFn>);
static_assert(std::is_same_v<decltype(&find_certificate_in_store), FindCertificateInStoreFn>);
static_assert(std::is_same_v<decltype(&find_chain_in_store), FindChainInStoreFn>);
static_assert(std::is_same_v<decltype(&free_certificate_context), FreeCertificateContextFn>);
static_assert(std::is_same_v<decltype(&free_certificate_chain), FreeCertificateChainFn>);
static_assert(std::is_same_v<decltype(&close_store), CloseStoreFn>);
static_assert(std::is_same_v<decltype(&string_to_binary_w), StringToBinaryWFn>);

}  // namespace
}  // namespace rt::crypt32

using namespace rt::crypt32;

SHIM_TABLE(crypt32_offline,
    {"crypt32.dll", "CertOpenStore", reinterpret_cast<void*>(&open_store)},
    {"crypt32.dll", "CertOpenSystemStoreW", reinterpret_cast<void*>(&open_system_store_w)},
    {"crypt32.dll", "CertFindCertificateInStore", reinterpret_cast<void*>(&find_certificate_in_store)},
    {"crypt32.dll", "CertFindChainInStore", reinterpret_cast<void*>(&find_chain_in_store)},
    {"crypt32.dll", "CertFreeCertificateContext", reinterpret_cast<void*>(&free_certificate_context)},
    {"crypt32.dll", "CertFreeCertificateChain", reinterpret_cast<void*>(&free_certificate_chain)},
    {"crypt32.dll", "CertCloseStore", reinterpret_cast<void*>(&close_store)},
    {"crypt32.dll", "CryptStringToBinaryW", reinterpret_cast<void*>(&string_to_binary_w)},
);
