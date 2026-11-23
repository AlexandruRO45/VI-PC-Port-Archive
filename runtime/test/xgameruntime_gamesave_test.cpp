// XGameSaveImpl through QueryApiImpl, called the way xgameruntime.lib's XGameSave* thunks call it: QueryApiImpl with
// the class id (or a later interface id), one vtable slot with `this` first. Slots follow xgameruntime-docs' layout;
// arguments, limits and error codes follow the public GDK reference. Saves go to a temporary LEONIDA_SAVE_DIR; no
// game data, nothing online.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "shims.h"

namespace rt::shims {
long xgameruntime_gamesave_open_handles();
}

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

using Handle = void*;
struct ContainerInfo {
    const char* name;
    const char* displayName;
    uint32_t blobCount;
    uint64_t totalSize;
    time_t lastModifiedTime;
    bool needsSync;
};
struct BlobInfo {
    const char* name;
    uint32_t size;
};
struct Blob {
    BlobInfo info;
    uint8_t* data;
};

constexpr HRESULT E_GS_INVALID_CONTAINER_NAME = static_cast<HRESULT>(0x80830001);
constexpr HRESULT E_GS_OUT_OF_LOCAL_STORAGE = static_cast<HRESULT>(0x80830003);
constexpr HRESULT E_GS_UPDATE_TOO_BIG = static_cast<HRESULT>(0x80830005);
constexpr HRESULT E_GS_PROVIDED_BUFFER_TOO_SMALL = static_cast<HRESULT>(0x80830007);
constexpr HRESULT E_GS_BLOB_NOT_FOUND = static_cast<HRESULT>(0x80830008);
constexpr HRESULT E_GS_HANDLE_EXPIRED = static_cast<HRESULT>(0x8083000D);

constexpr GUID kSaveClass = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}};
constexpr GUID kSaveImpl2 = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6e, 0xe2}};
constexpr GUID kSaveImpl3 = {0x1bfff3af, 0xf14a, 0x40a3, {0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9}};
constexpr GUID kOther = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};  // XGameImpl

using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
using Fn = void*;
Fn slot(void* obj, int index) { return (*static_cast<Fn* const*>(obj))[index]; }
template <class F> F at(void* obj, int index) { return reinterpret_cast<F>(slot(obj, index)); }

bool finished(XAsyncBlock* async) { return XAsyncGetStatus(async, true) != E_PENDING; }

using Init = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, bool, Handle*);
using InitAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, bool, XAsyncBlock*);
using ResultHandle = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, Handle*);
using Close = void(STDMETHODCALLTYPE*)(void*, Handle);
using Quota = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, int64_t*);
using WithAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, XAsyncBlock*);
using QuotaResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, int64_t*);
using ByName = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*);
using ByNameAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, XAsyncBlock*);
using AsyncOnly = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*);
using ContainerCb = bool(CALLBACK*)(const ContainerInfo*, void*);
using BlobCb = bool(CALLBACK*)(const BlobInfo*, void*);
using GetInfo = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, void*, ContainerCb);
using EnumContainers = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, void*, ContainerCb);
using EnumBlobs = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, void*, BlobCb);
using EnumBlobsByName = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, void*, BlobCb);
using Create = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, Handle*);
using Read = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char**, uint32_t*, size_t, Blob*);
using ReadAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char**, uint32_t, XAsyncBlock*);
using ReadResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, Blob*, uint32_t*);
using Write = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, const uint8_t*, size_t);
using Submit = HRESULT(STDMETHODCALLTYPE*)(void*, Handle);
using FolderAsync = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, XAsyncBlock*);
using FolderResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, char*);
using FilesQuota = HRESULT(STDMETHODCALLTYPE*)(void*, Handle, const char*, int64_t*);

struct Seen {
    std::vector<std::string> names, displays;
    std::vector<uint32_t> counts;
    std::vector<uint64_t> sizes;
    int stop_after = 1000;
};
bool CALLBACK on_container(const ContainerInfo* info, void* context) {
    auto* seen = static_cast<Seen*>(context);
    seen->names.emplace_back(info->name);
    seen->displays.emplace_back(info->displayName ? info->displayName : "");
    seen->counts.push_back(info->blobCount);
    seen->sizes.push_back(info->totalSize);
    return static_cast<int>(seen->names.size()) < seen->stop_after;
}
bool CALLBACK on_blob(const BlobInfo* info, void* context) {
    auto* seen = static_cast<Seen*>(context);
    seen->names.emplace_back(info->name);
    seen->sizes.push_back(info->size);
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a crash still shows the checks before it
    std::printf("xgameruntime_gamesave_test: XGameSaveImpl through QueryApiImpl\n");
    const std::filesystem::path root = std::filesystem::temp_directory_path() / L"leonida_gamesave_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    SetEnvironmentVariableW(L"LEONIDA_SAVE_DIR", root.c_str());

    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    void* save = nullptr;
    check(query(&kSaveClass, kSaveClass, &save) == S_OK && save, "QueryApiImpl hands out XGameSaveImpl for its class id");
    if (!save) return 1;
    void* save2 = nullptr;
    void* save3 = nullptr;
    check(query(&kSaveClass, kSaveImpl2, &save2) == S_OK && save2 == save &&
              query(&kSaveClass, kSaveImpl3, &save3) == S_OK && save3 == save,
          "IXGameSaveImpl2 and IXGameSaveImpl3 answer with the same object");
    void* none = reinterpret_cast<void*>(1);
    check(query(&kSaveClass, kOther, &none) == E_NOINTERFACE && none == nullptr, "another interface id is refused");

    Handle user = reinterpret_cast<Handle>(0x1000);   // XUserHandle: opaque here, only checked for null
    Handle provider = nullptr;
    check(at<Init>(save, 3)(save, nullptr, "SCID", false, &provider) == E_INVALIDARG && !provider,
          "InitializeProvider without a user is refused");
    check(at<Init>(save, 3)(save, user, "SCID", true, &provider) == S_OK && provider,
          "InitializeProvider opens the offline provider (syncOnDemand is accepted)");
    check(std::filesystem::is_directory(root / L"local-user"), "the provider's folder is <LEONIDA_SAVE_DIR>\\local-user");

    int64_t quota = 0;
    check(at<Quota>(save, 7)(save, provider, &quota) == S_OK && quota == (int64_t{256} << 20),
          "GetRemainingQuota of an empty provider is the whole documented default, 256 MB");

    // ---- containers and names ----
    Handle container = nullptr;
    check(at<Create>(save, 16)(save, provider, "bad name", &container) == E_GS_INVALID_CONTAINER_NAME && !container,
          "a container name with a space is refused");
    for (const char* bad : {"a..b", "dir/", "x.", "a/b-c/d", ""})
        check(at<Create>(save, 16)(save, provider, bad, &container) == E_GS_INVALID_CONTAINER_NAME,
              "a container name against the documented rule is refused");
    check(at<Create>(save, 16)(save, provider, std::string(257, 'a').c_str(), &container) == E_GS_INVALID_CONTAINER_NAME,
          "a container name over 256 characters is refused");
    check(at<Create>(save, 16)(save, provider, "slots/SAVE_00.sav-1", &container) == S_OK && container,
          "a name with a path part and . - in the file part is accepted");
    check(!std::filesystem::exists(root / L"local-user" / L"slots%2FSAVE_00.sav-1"),
          "a container's folder appears on its first update, not on create");

    // ---- an update ----
    Handle update = nullptr;
    check(at<Create>(save, 23)(save, container, "Story 1", &update) == S_OK && update, "CreateUpdate");
    const std::string story(5000, 'S'), profile = "profile bytes";
    check(at<Write>(save, 25)(save, update, "story", reinterpret_cast<const uint8_t*>(story.data()), story.size()) == S_OK,
          "SubmitBlobWrite buffers a blob");
    check(at<Write>(save, 25)(save, update, "story", reinterpret_cast<const uint8_t*>("x"), 1) == E_INVALIDARG,
          "a second write of the same blob in one update is refused");
    check(at<Write>(save, 25)(save, update, "profile/main", reinterpret_cast<const uint8_t*>(profile.data()),
                              profile.size()) == S_OK,
          "a blob name with a slash is accepted (it is %-encoded on disk)");
    std::vector<uint8_t> big(16u * 1024 * 1024, 1);
    check(at<Write>(save, 25)(save, update, "big", big.data(), big.size()) == E_GS_UPDATE_TOO_BIG,
          "an update past 16 MB is refused");
    check(at<Submit>(save, 27)(save, update) == S_OK, "SubmitUpdate commits");
    const auto dir = root / L"local-user" / L"slots%2FSAVE_00.sav-1";
    check(std::filesystem::file_size(dir / L"story") == story.size() && std::filesystem::exists(dir / L"profile%2Fmain"),
          "the blobs are files, in tools/saves.py's layout and encoding");
    at<Close>(save, 24)(save, update);

    // ---- enumeration ----
    Seen containers;
    check(at<EnumContainers>(save, 14)(save, provider, &containers, on_container) == S_OK &&
              containers.names.size() == 1 && containers.names[0] == "slots/SAVE_00.sav-1" &&
              containers.displays[0] == "Story 1" && containers.counts[0] == 2 &&
              containers.sizes[0] == story.size() + profile.size(),
          "EnumerateContainerInfo: name, display name, blob count and size");
    Seen info;
    check(at<GetInfo>(save, 13)(save, provider, "slots/SAVE_00.sav-1", &info, on_container) == S_OK &&
              info.names.size() == 1,
          "GetContainerInfo calls back for a container that exists");
    Seen missing;
    check(at<GetInfo>(save, 13)(save, provider, "nothing", &missing, on_container) == S_OK && missing.names.empty(),
          "GetContainerInfo of a missing container calls nothing");
    Seen blobs;
    check(at<EnumBlobs>(save, 18)(save, container, &blobs, on_blob) == S_OK && blobs.names.size() == 2 &&
              blobs.names[0] == "profile/main" && blobs.names[1] == "story" && blobs.sizes[1] == story.size(),
          "EnumerateBlobInfo lists both blobs with their decoded names and sizes");
    Seen prefixed;
    check(at<EnumBlobsByName>(save, 19)(save, container, "sto", &prefixed, on_blob) == S_OK &&
              prefixed.names.size() == 1 && prefixed.names[0] == "story",
          "EnumerateBlobInfoByName filters by prefix");

    // ---- reading ----
    const char* names[] = {"story", "profile/main"};
    uint32_t count = 2;
    std::vector<uint8_t> small(64);
    check(at<Read>(save, 20)(save, container, names, &count, small.size(), reinterpret_cast<Blob*>(small.data())) ==
              E_GS_PROVIDED_BUFFER_TOO_SMALL,
          "ReadBlobData into a buffer too small for the blobs is refused");
    std::vector<uint64_t> buffer(1024);
    count = 2;
    auto* out = reinterpret_cast<Blob*>(buffer.data());
    check(at<Read>(save, 20)(save, container, names, &count, buffer.size() * 8, out) == S_OK && count == 2 &&
              std::strcmp(out[0].info.name, "story") == 0 && out[0].info.size == story.size() &&
              std::memcmp(out[0].data, story.data(), story.size()) == 0 &&
              std::strcmp(out[1].info.name, "profile/main") == 0 &&
              std::memcmp(out[1].data, profile.data(), profile.size()) == 0,
          "ReadBlobData returns both blobs, byte for byte, pointing into the buffer");
    const char* absent[] = {"nope"};
    count = 1;
    check(at<Read>(save, 20)(save, container, absent, &count, buffer.size() * 8, out) == E_GS_BLOB_NOT_FOUND,
          "a blob that is not there is E_GS_BLOB_NOT_FOUND");
    count = 0;
    check(at<Read>(save, 20)(save, container, nullptr, &count, buffer.size() * 8, out) == S_OK && count == 2,
          "blobNames = nullptr reads every blob");

    XAsyncBlock async{};
    check(at<ReadAsync>(save, 21)(save, container, names, 1, &async) == S_OK && finished(&async),
          "ReadBlobDataAsync runs");
    size_t needed = 0;
    check(XAsyncGetResultSize(&async, &needed) == S_OK && needed > story.size(),
          "its result size is what the buffer needs (XAsyncGetResultSize)");
    std::vector<uint64_t> abuf((needed + 7) / 8);
    uint32_t acount = 0;
    auto* aout = reinterpret_cast<Blob*>(abuf.data());
    check(at<ReadResult>(save, 22)(save, &async, needed, aout, &acount) == S_OK && acount == 1 &&
              std::strcmp(aout[0].info.name, "story") == 0 && std::memcmp(aout[0].data, story.data(), story.size()) == 0,
          "ReadBlobDataResult fills the title's buffer with one blob");

    // ---- delete a blob, async submit ----
    check(at<Create>(save, 23)(save, container, nullptr, &update) == S_OK, "a second update");
    check(at<ByName>(save, 26)(save, update, "profile/main") == S_OK, "SubmitBlobDelete");
    async = XAsyncBlock{};
    check(at<WithAsync>(save, 28)(save, update, &async) == S_OK && finished(&async) &&
              at<AsyncOnly>(save, 29)(save, &async) == S_OK,
          "SubmitUpdateAsync commits the delete");
    check(!std::filesystem::exists(dir / L"profile%2Fmain") && std::filesystem::exists(dir / L"story"),
          "the deleted blob is gone, the other stays");
    Seen after;
    at<EnumContainers>(save, 14)(save, provider, &after, on_container);
    check(after.displays.size() == 1 && after.displays[0] == "Story 1", "an update without a display name keeps it");
    at<Close>(save, 24)(save, update);
    check(at<Submit>(save, 27)(save, update) == E_GS_HANDLE_EXPIRED, "a closed update is refused");

    // ---- quota ----
    async = XAsyncBlock{};
    int64_t aquota = 0;
    check(at<WithAsync>(save, 8)(save, provider, &async) == S_OK && finished(&async) &&
              at<QuotaResult>(save, 9)(save, &async, &aquota) == S_OK &&
              aquota == (int64_t{256} << 20) - static_cast<int64_t>(story.size()),
          "GetRemainingQuotaAsync subtracts what the saves hold");

    // ---- delete the container ----
    async = XAsyncBlock{};
    check(at<ByNameAsync>(save, 11)(save, provider, "slots/SAVE_00.sav-1", &async) == S_OK && finished(&async) &&
              at<AsyncOnly>(save, 12)(save, &async) == S_OK && !std::filesystem::exists(dir),
          "DeleteContainerAsync removes the container");
    at<Close>(save, 17)(save, container);

    // ---- an update is atomic per container (RT-375) ----
    {
        const std::filesystem::path user_dir = root / L"local-user";
        auto file_text = [](const std::filesystem::path& f) {
            std::ifstream in(f, std::ios::binary);
            return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        };
        Handle atom = nullptr, up = nullptr;
        check(at<Create>(save, 16)(save, provider, "atomic", &atom) == S_OK && atom, "CreateContainer atomic");
        at<Create>(save, 23)(save, atom, "Atomic", &up);
        at<Write>(save, 25)(save, up, "a", reinterpret_cast<const uint8_t*>("old-a"), 5);
        at<Write>(save, 25)(save, up, "b", reinterpret_cast<const uint8_t*>("old-b"), 5);
        check(at<Submit>(save, 27)(save, up) == S_OK, "a first update writes a and b");
        at<Close>(save, 24)(save, up);
        // Hold "b" open without FILE_SHARE_DELETE, so neither replacing it nor moving its folder can succeed. Replacing
        // blob by blob (the old commit) renamed "a" first and failed on "b", leaving a mixed container.
        const std::filesystem::path adir = user_dir / L"atomic";
        HANDLE held = CreateFileW((adir / L"b").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        at<Create>(save, 23)(save, atom, nullptr, &up);
        at<Write>(save, 25)(save, up, "a", reinterpret_cast<const uint8_t*>("new-a"), 5);
        at<Write>(save, 25)(save, up, "b", reinterpret_cast<const uint8_t*>("new-b"), 5);
        const HRESULT blocked = at<Submit>(save, 27)(save, up);
        if (held != INVALID_HANDLE_VALUE) CloseHandle(held);
        check(held != INVALID_HANDLE_VALUE && blocked == E_GS_OUT_OF_LOCAL_STORAGE && file_text(adir / L"a") == "old-a" &&
                  file_text(adir / L"b") == "old-b",
              "an update that can't be applied leaves every blob as it was (container-level atomic)");
        bool hidden = false;
        for (const auto& e : std::filesystem::directory_iterator(user_dir)) hidden |= e.path().filename().wstring()[0] == L'.';
        check(!hidden, "and leaves no staging or backup folder behind");
        check(at<Submit>(save, 27)(save, up) == S_OK && file_text(adir / L"a") == "new-a" && file_text(adir / L"b") == "new-b",
              "the same update succeeds once nothing holds the blob");
        at<Close>(save, 24)(save, up);
        at<Close>(save, 17)(save, atom);

        // A container whose name is near the 255-character limit of a folder name updates like any other: the
        // commit's own folders have short fixed names (a suffix on the container's name would not fit). The whole path
        // is past MAX_PATH, so this provider's LEONIDA_SAVE_DIR is an extended-length "\\?\" path.
        {
            const std::filesystem::path xroot = L"\\\\?\\" + std::filesystem::absolute(root).wstring();
            SetEnvironmentVariableW(L"LEONIDA_SAVE_DIR", xroot.c_str());
            Handle lp = nullptr, lc = nullptr, lu = nullptr;
            at<Init>(save, 3)(save, user, "SCID", false, &lp);
            SetEnvironmentVariableW(L"LEONIDA_SAVE_DIR", root.c_str());
            const std::filesystem::path ldir = xroot / L"local-user" / std::wstring(250, L'L');
            std::error_code ec;
            std::filesystem::create_directories(ldir, ec);
            { std::ofstream(ldir / L"story", std::ios::binary) << "old"; }
            const std::string long_name(250, 'L');
            at<Create>(save, 16)(save, lp, long_name.c_str(), &lc);
            at<Create>(save, 23)(save, lc, nullptr, &lu);
            at<Write>(save, 25)(save, lu, "story", reinterpret_cast<const uint8_t*>("new"), 3);
            check(lp && lc && lu && at<Submit>(save, 27)(save, lu) == S_OK && file_text(ldir / L"story") == "new",
                  "an existing container with a 250-character name takes an update");
            at<Close>(save, 24)(save, lu);
            at<Close>(save, 17)(save, lc);
            at<Close>(save, 6)(save, lp);
            std::filesystem::remove_all(ldir, ec);
        }

        // A commit that died between its two renames leaves the container only in ".txn\<pid>-<n>\old"; opening a
        // provider puts it back, by the name the commit recorded, whatever that name looks like ("slot.staging-1" and
        // "x.old-1" are valid container names). Every other left-over commit folder is removed: one whose container
        // is in place, one that never got to record a name.
        const std::filesystem::path txn = user_dir / L".txn";
        auto crashed = [&](const wchar_t* folder, const char* name, bool old, const char* blob) {
            const std::filesystem::path t = txn / folder;
            std::filesystem::create_directories(t / L"new");
            if (name) std::ofstream(t / L"name", std::ios::binary) << name;
            if (old) {
                std::filesystem::create_directories(t / L"old");
                std::ofstream(t / L"old" / L"x", std::ios::binary) << blob;
            }
        };
        crashed(L"1-1", "slot.staging-1", true, "kept-1");
        crashed(L"1-2", "x.old-1", true, "kept-2");
        crashed(L"1-3", "atomic", true, "stale");    // its container exists: the stale copy goes, the container stays
        crashed(L"1-4", nullptr, false, nullptr);    // died before recording the name: nothing was moved yet
        crashed(L"1-5", "..", true, "never");        // not a name a commit could have recorded
        Handle reopened = nullptr;
        check(at<Init>(save, 3)(save, user, "SCID", false, &reopened) == S_OK && reopened &&
                  file_text(user_dir / L"slot.staging-1" / L"x") == "kept-1" &&
                  file_text(user_dir / L"x.old-1" / L"x") == "kept-2" && file_text(adir / L"a") == "new-a" &&
                  !std::filesystem::exists(adir / L"x") && !std::filesystem::exists(user_dir / L"..x") &&
                  !std::filesystem::exists(txn),
              "opening a provider restores a container a crash left as its old copy, and removes stale commits");
        at<Close>(save, 6)(save, reopened);
        std::error_code rm;
        std::filesystem::remove_all(user_dir / L"slot.staging-1", rm);
        std::filesystem::remove_all(user_dir / L"x.old-1", rm);
        std::filesystem::remove_all(adir, rm);
    }

    // ---- async provider, XGameSaveFiles ----
    async = XAsyncBlock{};
    Handle provider2 = nullptr;
    check(at<InitAsync>(save, 4)(save, user, "SCID", false, &async) == S_OK && finished(&async) &&
              at<ResultHandle>(save, 5)(save, &async, &provider2) == S_OK && provider2 && provider2 != provider,
          "InitializeProviderAsync hands out a second provider");
    async = XAsyncBlock{};
    check(at<FolderAsync>(save, 30)(save, user, "SCID", &async) == S_OK && finished(&async), "FilesGetFolderWithUiAsync");
    size_t folder_size = 0;
    XAsyncGetResultSize(&async, &folder_size);
    std::string folder(folder_size, '\0');
    check(folder_size > 0 && at<FolderResult>(save, 31)(save, &async, folder.size(), folder.data()) == S_OK &&
              std::filesystem::is_directory(std::filesystem::u8path(folder.c_str())),
          "FilesGetFolderWithUiResult returns a folder that exists");
    int64_t fquota = 0;
    check(at<FilesQuota>(save, 32)(save, user, "SCID", &fquota) == S_OK && fquota == (int64_t{256} << 20),
          "FilesGetRemainingQuota of an empty folder");

    at<Close>(save, 6)(save, provider);
    at<Close>(save, 6)(save, provider2);
    at<Close>(save, 6)(save, provider);   // twice: ignored
    check(at<Quota>(save, 7)(save, provider, &quota) == E_GS_HANDLE_EXPIRED, "a closed provider is refused");
    check(rt::shims::xgameruntime_gamesave_open_handles() == 0, "every handle the test opened is closed");

    std::filesystem::remove_all(root, ec);
    std::printf("xgameruntime_gamesave_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
