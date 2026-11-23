// Tests for runtime/shims/mf_offline.cpp through rt::shims::find.
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mferror.h>
#include <mfreadwrite.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "file_system.h"
#include "shims.h"
#include "test_temp_dir.h"

namespace {

using MfStartupFn = HRESULT(STDAPICALLTYPE*)(ULONG, DWORD);
using MfShutdownFn = HRESULT(STDAPICALLTYPE*)();
using MfCreateAttributesFn = HRESULT(STDAPICALLTYPE*)(IMFAttributes**, UINT32);
using MfCreateMediaTypeFn = HRESULT(STDAPICALLTYPE*)(IMFMediaType**);
using MfCreateSourceResolverFn = HRESULT(STDAPICALLTYPE*)(IMFSourceResolver**);
using MfGetSystemTimeFn = MFTIME(STDAPICALLTYPE*)();
using MfCreateSourceReaderFromMediaSourceFn = HRESULT(STDAPICALLTYPE*)(IMFMediaSource*, IMFAttributes*,
                                                                       IMFSourceReader**);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename Fn>
Fn resolve(const char* dll, const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find(dll, name));
}

void write_bytes(const std::filesystem::path& path, const char* data, size_t size) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(data, static_cast<std::streamsize>(size));
}

void test_registry_lookup() {
    for (const char* name :
         {"MFStartup", "MFShutdown", "MFCreateAttributes", "MFCreateMediaType", "MFCreateSourceResolver",
          "MFGetSystemTime"}) {
        check(rt::shims::find("mfplat.dll", name) != nullptr, "mfplat.dll import resolves");
    }
    check(rt::shims::find("mfreadwrite.dll", "MFCreateSourceReaderFromMediaSource") != nullptr,
          "mfreadwrite.dll!MFCreateSourceReaderFromMediaSource resolves");
}

void test_startup_shutdown_balance() {
    auto startup = resolve<MfStartupFn>("mfplat.dll", "MFStartup");
    auto shutdown = resolve<MfShutdownFn>("mfplat.dll", "MFShutdown");

    check(shutdown() == MF_E_SHUTDOWN, "unbalanced initial MFShutdown returns MF_E_SHUTDOWN");
    check(startup(MF_VERSION, MFSTARTUP_LITE) == S_OK, "first MFStartup succeeds");
    check(startup(MF_VERSION, MFSTARTUP_LITE) == S_OK, "nested MFStartup succeeds");
    check(shutdown() == S_OK, "first MFShutdown balances nested startup");
    check(shutdown() == S_OK, "second MFShutdown balances initial startup");
    check(shutdown() == MF_E_SHUTDOWN, "extra MFShutdown after balance returns MF_E_SHUTDOWN");
}

constexpr GUID kTestAttrKey{0x12345678, 0x2345, 0x4567, {0x89, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd}};
constexpr GUID kMfMtMajorType{0x48eba18e, 0xf8c9, 0x4687, {0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f}};
constexpr GUID kMfMtSubtype{0xf7e34c9a, 0x42e8, 0x4714, {0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5}};
constexpr GUID kMfMtAllSamplesIndependent{
    0xc9173739, 0x5e56, 0x461c, {0xb7, 0x13, 0x46, 0xfb, 0x99, 0x5c, 0xb9, 0x5f}};
constexpr GUID kMfMediaTypeAudio{0x73647561, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
constexpr GUID kMfAudioFormatPcm{0x00000001, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};

void test_attributes_and_media_type() {
    auto startup = resolve<MfStartupFn>("mfplat.dll", "MFStartup");
    auto shutdown = resolve<MfShutdownFn>("mfplat.dll", "MFShutdown");
    auto create_attrs = resolve<MfCreateAttributesFn>("mfplat.dll", "MFCreateAttributes");
    auto create_mt = resolve<MfCreateMediaTypeFn>("mfplat.dll", "MFCreateMediaType");

    check(startup(MF_VERSION, MFSTARTUP_LITE) == S_OK, "MFStartup for attributes/media type succeeds");
    check(create_attrs(nullptr, 0) == E_POINTER, "MFCreateAttributes rejects null pointer");
    check(create_mt(nullptr) == E_POINTER, "MFCreateMediaType rejects null pointer");

    IMFAttributes* attrs = nullptr;
    check(create_attrs(&attrs, 4) == S_OK && attrs != nullptr, "MFCreateAttributes returns non-null object");
    if (attrs) {
        check(attrs->SetUINT32(kMfMtAllSamplesIndependent, 1) == S_OK, "IMFAttributes::SetUINT32 succeeds");
        UINT32 val = 0;
        check(attrs->GetUINT32(kMfMtAllSamplesIndependent, &val) == S_OK && val == 1,
              "IMFAttributes::GetUINT32 reads value");
        check(attrs->SetString(kTestAttrKey, L"test") == S_OK, "IMFAttributes::SetString succeeds");
        wchar_t buf[16]{};
        UINT32 len = 0;
        check(attrs->GetString(kTestAttrKey, buf, 16, &len) == S_OK && len == 4 &&
                  std::wcscmp(buf, L"test") == 0,
              "IMFAttributes::GetString reads value");
        IMFMediaType* as_mt = nullptr;
        check(attrs->QueryInterface(__uuidof(IMFMediaType), reinterpret_cast<void**>(&as_mt)) == E_NOINTERFACE,
              "plain IMFAttributes refuses QueryInterface(IMFMediaType)");
        attrs->Release();
    }

    IMFMediaType* mt1 = nullptr;
    IMFMediaType* mt2 = nullptr;
    check(create_mt(&mt1) == S_OK && mt1 != nullptr, "MFCreateMediaType creates first media type");
    check(create_mt(&mt2) == S_OK && mt2 != nullptr, "MFCreateMediaType creates second media type");
    if (mt1 && mt2) {
        check(mt1->SetGUID(kMfMtMajorType, kMfMediaTypeAudio) == S_OK, "IMFMediaType::SetGUID major type");
        check(mt1->SetGUID(kMfMtSubtype, kMfAudioFormatPcm) == S_OK, "IMFMediaType::SetGUID subtype");
        check(mt1->SetUINT32(kMfMtAllSamplesIndependent, 0) == S_OK, "IMFMediaType::SetUINT32 independent=0");

        GUID major{};
        check(mt1->GetMajorType(&major) == S_OK && IsEqualGUID(major, kMfMediaTypeAudio),
              "IMFMediaType::GetMajorType matches");
        BOOL compressed = FALSE;
        check(mt1->IsCompressedFormat(&compressed) == S_OK && compressed == TRUE,
              "IMFMediaType::IsCompressedFormat reflects independent=0");
        check(mt1->CopyAllItems(mt2) == S_OK, "IMFMediaType::CopyAllItems copies items");
        DWORD eq_flags = 0;
        check(mt1->IsEqual(mt2, &eq_flags) == S_OK &&
                  (eq_flags & (MF_MEDIATYPE_EQUAL_MAJOR_TYPES | MF_MEDIATYPE_EQUAL_FORMAT_TYPES)) != 0,
              "IMFMediaType::IsEqual matches copied type");
    }
    if (mt1) mt1->Release();
    if (mt2) mt2->Release();
    check(shutdown() == S_OK, "MFShutdown after attributes/media type succeeds");
}

void test_source_resolver_and_console_paths() {
    auto startup = resolve<MfStartupFn>("mfplat.dll", "MFStartup");
    auto shutdown = resolve<MfShutdownFn>("mfplat.dll", "MFShutdown");
    auto create_resolver = resolve<MfCreateSourceResolverFn>("mfplat.dll", "MFCreateSourceResolver");
    auto get_time = resolve<MfGetSystemTimeFn>("mfplat.dll", "MFGetSystemTime");
    auto create_reader =
        resolve<MfCreateSourceReaderFromMediaSourceFn>("mfreadwrite.dll", "MFCreateSourceReaderFromMediaSource");

    check(get_time() > 0, "MFGetSystemTime returns positive timestamp");
    check(create_resolver(nullptr) == E_POINTER, "MFCreateSourceResolver rejects null pointer");

    IMFSourceReader* reader = nullptr;
    check(create_reader(nullptr, nullptr, nullptr) == E_POINTER,
          "MFCreateSourceReaderFromMediaSource rejects null out pointer");
    check(create_reader(nullptr, nullptr, &reader) == E_INVALIDARG && reader == nullptr,
          "MFCreateSourceReaderFromMediaSource rejects null media source");

    const testfiles::ScratchDir temp("mf_offline_test");
    const std::filesystem::path game = temp.path() / L"game";
    const std::filesystem::path overlay = temp.path() / L"overlay_unused";
    std::filesystem::create_directories(game / L"movies");
    write_bytes(game / L"movies" / L"dummy.bin", "NOTAVIDEO", 9);

    rt::fs::configure(game, overlay);

    check(startup(MF_VERSION, MFSTARTUP_LITE) == S_OK, "MFStartup for resolver succeeds");
    IMFSourceResolver* resolver = nullptr;
    check(create_resolver(&resolver) == S_OK && resolver != nullptr, "MFCreateSourceResolver returns resolver proxy");
    if (resolver) {
        MF_OBJECT_TYPE obj_type = MF_OBJECT_INVALID;
        IUnknown* obj = nullptr;
        const HRESULT invalid_name = HRESULT_FROM_WIN32(ERROR_INVALID_NAME);
        const HRESULT not_found = HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);

        check(resolver->CreateObjectFromURL(nullptr, MF_RESOLUTION_MEDIASOURCE, nullptr, &obj_type, &obj) ==
                  E_INVALIDARG,
              "CreateObjectFromURL rejects nullptr URL");
        check(resolver->CreateObjectFromURL(L"GAME:\\..\\outside.bin", MF_RESOLUTION_MEDIASOURCE, nullptr, &obj_type,
                                            &obj) == invalid_name,
              "CreateObjectFromURL rejects GAME: traversal past root");
        check(resolver->CreateObjectFromURL(L"file:///G:/../outside.bin", MF_RESOLUTION_MEDIASOURCE, nullptr, &obj_type,
                                            &obj) == invalid_name,
              "CreateObjectFromURL rejects file:///G: traversal past root");
        check(resolver->CreateObjectFromURL(L"T:\\missing.bin", MF_RESOLUTION_MEDIASOURCE, nullptr, &obj_type, &obj) ==
                  not_found,
              "CreateObjectFromURL returns ERROR_FILE_NOT_FOUND on missing T: file");
        check(!std::filesystem::exists(overlay),
              "read-only T: / GAME: URL lookup does not create missing overlay directory");

        // Deleted package file must stay hidden even when present on disk in `game`.
        write_bytes(overlay / L".gta6rt-deleted" / L"movies" / L"deleted.bin", "", 0);
        write_bytes(game / L"movies" / L"deleted.bin", "NOTAVIDEO", 9);
        check(resolver->CreateObjectFromURL(L"GAME:/movies/deleted.bin", MF_RESOLUTION_MEDIASOURCE, nullptr, &obj_type,
                                            &obj) == not_found,
              "CreateObjectFromURL honors .gta6rt-deleted markers");

        // Valid bogus-media file reaches host parser or offline unsupported-bytestream error (never FILE_NOT_FOUND).
        const HRESULT bogus_hr =
            resolver->CreateObjectFromURL(L"GAME:\\movies\\dummy.bin", MF_RESOLUTION_MEDIASOURCE, nullptr, &obj_type, &obj);
        if (obj) obj->Release();
        check(FAILED(bogus_hr) && bogus_hr != not_found && bogus_hr != invalid_name,
              "CreateObjectFromURL resolves existing GAME: file past path lookup");

        check(resolver->BeginCreateObjectFromURL(L"T:\\..\\outside.bin", MF_RESOLUTION_MEDIASOURCE, nullptr, nullptr,
                                                 nullptr, nullptr) == invalid_name,
              "BeginCreateObjectFromURL rejects T: traversal past root");

        resolver->Release();
    }
    check(shutdown() == S_OK, "MFShutdown after resolver succeeds");
    rt::fs::configure({}, {});
}

}  // namespace

int main() {
    test_registry_lookup();
    test_startup_shutdown_balance();
    test_attributes_and_media_type();
    test_source_resolver_and_console_paths();
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
