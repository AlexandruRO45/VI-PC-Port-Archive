// Checks the offline crypt32.dll shims (runtime/shims/crypt32_offline.cpp).
//
// Every function is resolved through rt::shims::find under crypt32.dll, verifying that certificate-store opens and
// lookups fail offline without touching host stores, while CryptStringToBinaryW decodes base64 inputs.
#include <windows.h>
#include <wincrypt.h>

#include <cstdio>
#include <cstring>

#include "shims.h"

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

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find("crypt32.dll", name));
}

}  // namespace

int main() {
    auto open_store = resolve<OpenStoreFn>("CertOpenStore");
    auto open_system_store_w = resolve<OpenSystemStoreWFn>("CertOpenSystemStoreW");
    auto find_cert = resolve<FindCertificateInStoreFn>("CertFindCertificateInStore");
    auto find_chain = resolve<FindChainInStoreFn>("CertFindChainInStore");
    auto free_cert = resolve<FreeCertificateContextFn>("CertFreeCertificateContext");
    auto free_chain = resolve<FreeCertificateChainFn>("CertFreeCertificateChain");
    auto close_store = resolve<CloseStoreFn>("CertCloseStore");
    auto str_to_bin = resolve<StringToBinaryWFn>("CryptStringToBinaryW");

    check(open_store && open_system_store_w && find_cert && find_chain && free_cert && free_chain && close_store &&
              str_to_bin,
          "all 8 crypt32.dll exports are registered in the shim table");
    check(rt::shims::owns_dll("crypt32.dll"), "crypt32.dll is owned by the shims");

    SetLastError(0);
    HCERTSTORE store = open_store(CERT_STORE_PROV_SYSTEM_W, 0, 0, CERT_SYSTEM_STORE_CURRENT_USER, L"MY");
    check(store == nullptr && GetLastError() == ERROR_ACCESS_DENIED,
          "CertOpenStore fails offline with ERROR_ACCESS_DENIED");

    SetLastError(0);
    HCERTSTORE sys_store = open_system_store_w(0, L"ROOT");
    check(sys_store == nullptr && GetLastError() == ERROR_ACCESS_DENIED,
          "CertOpenSystemStoreW fails offline with ERROR_ACCESS_DENIED");

    SetLastError(0);
    PCCERT_CONTEXT cert = find_cert(nullptr, X509_ASN_ENCODING, 0, CERT_FIND_ANY, nullptr, nullptr);
    check(cert == nullptr && GetLastError() == static_cast<DWORD>(CRYPT_E_NOT_FOUND),
          "CertFindCertificateInStore fails offline with CRYPT_E_NOT_FOUND");

    PCCERT_CHAIN_CONTEXT chain = reinterpret_cast<PCCERT_CHAIN_CONTEXT>(0x1);
    SetLastError(0);
    BOOL chain_ok = find_chain(nullptr, X509_ASN_ENCODING, 0, CERT_CHAIN_FIND_BY_ISSUER, nullptr, nullptr, &chain);
    check(!chain_ok && chain == nullptr && GetLastError() == static_cast<DWORD>(CRYPT_E_NOT_FOUND),
          "CertFindChainInStore clears out-pointer and fails with CRYPT_E_NOT_FOUND");

    check(free_cert(nullptr) == TRUE, "CertFreeCertificateContext accepts null");
    free_chain(nullptr);
    check(true, "CertFreeCertificateChain accepts null");

    SetLastError(0);
    check(close_store(nullptr, 0) == FALSE && GetLastError() == ERROR_INVALID_HANDLE,
          "CertCloseStore rejects null with ERROR_INVALID_HANDLE");
    check(close_store(reinterpret_cast<HCERTSTORE>(0x1), 0) == TRUE,
          "CertCloseStore accepts non-null handle");

    // "hello" in base64 is "aGVsbG8="
    BYTE decoded[16] = {};
    DWORD decoded_len = sizeof(decoded);
    DWORD skip = 0;
    DWORD out_flags = 0;
    BOOL b64_ok = str_to_bin(L"aGVsbG8=", 0, CRYPT_STRING_BASE64, decoded, &decoded_len, &skip, &out_flags);
    check(b64_ok && decoded_len == 5 && std::memcmp(decoded, "hello", 5) == 0,
          "CryptStringToBinaryW decodes base64 string");

    std::printf("crypt32_offline_test: %d/%d passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
