// xgamesave_test: XGameSaveImpl through QueryApiImpl, the way xgameruntime.lib's thunks reach it (class id, interface
// id, then one vtable slot with `this` first). Synthetic saves in a temporary LEONIDA_SAVE_DIR; no game files.
#include <windows.h>

#include <XAsync.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../shims/shims.h"

namespace fs = std::filesystem;

namespace {

int g_checks = 0, g_failures = 0;
void check(bool ok, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

constexpr HRESULT E_GS_INVALID_CONTAINER_NAME = static_cast<HRESULT>(0x80830001);
constexpr HRESULT E_GS_PROVIDED_BUFFER_TOO_SMALL = static_cast<HRESULT>(0x80830007);
constexpr HRESULT E_GS_BLOB_NOT_FOUND = static_cast<HRESULT>(0x80830008);
constexpr HRESULT E_GAMERUNTIME_INVALID_HANDLE = static_cast<HRESULT>(0x8924010C);

struct BlobInfo {
    const char* name;
    uint32_t size;
};
struct Blob {
    BlobInfo info;
    uint8_t* data;
};
struct ContainerInfo {
    const char* name;
    const char* display_name;
    uint32_t blob_count;
    uint64_t total_size;
    int64_t last_modified_time;
    bool needs_sync;
};

using Fn = void (*)();
Fn slot(void* obj, int i) { return (*static_cast<Fn* const*>(obj))[i]; }
template <class F> F at(void* obj, int i) { return reinterpret_cast<F>(slot(obj, i)); }

struct Seen {
    std::vector<std::string> names, display;
    std::vector<uint32_t> blobs;
    std::vector<uint64_t> bytes;
    int stop_after = 1 << 30;
};
BOOLEAN CALLBACK on_container(const ContainerInfo* info, void* ctx) {
    auto* s = static_cast<Seen*>(ctx);
    s->names.push_back(info->name);
    s->display.push_back(info->display_name);
    s->blobs.push_back(info->blob_count);
    s->bytes.push_back(info->total_size);
    return static_cast<int>(s->names.size()) < s->stop_after;
}
BOOLEAN CALLBACK on_blob(const BlobInfo* info, void* ctx) {
    auto* s = static_cast<Seen*>(ctx);
    s->names.push_back(info->name);
    s->bytes.push_back(info->size);
    return true;
}

bool wait(XAsyncBlock& async) { return XAsyncGetStatus(&async, true) == S_OK; }

}  // namespace

int main() {
    using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    if (!query) {
        std::printf("FAIL QueryApiImpl is not in the shim table\n");
        return 1;
    }
    char tmp[MAX_PATH] = {};
    GetTempPathA(sizeof(tmp), tmp);
    const fs::path root = fs::path(tmp) / ("leonida_xgamesave_test_" + std::to_string(GetCurrentProcessId()));
    std::error_code ec;
    fs::remove_all(root, ec);
    SetEnvironmentVariableW(L"LEONIDA_SAVE_DIR", root.c_str());

    // The class and its three interface ids (xgamesave.idl): one object.
    const GUID cls = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}};
    const GUID iid2 = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6e, 0xe2}};
    const GUID iid3 = {0x1bfff3af, 0xf14a, 0x40a3, {0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9}};
    void *gs = nullptr, *gs2 = nullptr, *gs3 = nullptr;
    check(query(&cls, cls, &gs) == S_OK && gs && query(&cls, iid2, &gs2) == S_OK && query(&cls, iid3, &gs3) == S_OK &&
              gs2 == gs && gs3 == gs,
          "QueryApiImpl hands out XGameSaveImpl for its class id, IXGameSaveImpl2 and IXGameSaveImpl3: one object");
    if (!gs) return 1;

    using InitProvider = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, bool, void**);
    using InitAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, bool, XAsyncBlock*);
    using InitResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, void**);
    using Close = void(STDMETHODCALLTYPE*)(void*, void*);
    using Quota = HRESULT(STDMETHODCALLTYPE*)(void*, void*, int64_t*);
    using QuotaAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, XAsyncBlock*);
    using QuotaResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, int64_t*);
    using Delete = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*);
    using DeleteAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, XAsyncBlock*);
    using AsyncResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*);
    using Info = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, void*, decltype(&on_container));
    using EnumAll = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void*, decltype(&on_container));
    using EnumBlobs = HRESULT(STDMETHODCALLTYPE*)(void*, void*, void*, decltype(&on_blob));
    using Create = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, void**);
    using Read = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char**, uint32_t*, size_t, Blob*);
    using ReadAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char**, uint32_t, XAsyncBlock*);
    using ReadResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, Blob*, uint32_t*);
    using CreateUpdate = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, void**);
    using Write = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, const uint8_t*, size_t);
    using DeleteBlob = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*);
    using Submit = HRESULT(STDMETHODCALLTYPE*)(void*, void*);
    using SubmitAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, XAsyncBlock*);
    using FolderAsync = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, XAsyncBlock*);
    using FolderResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, char*);
    using FilesQuota = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, int64_t*);

    // 3/4/5 InitializeProvider, sync and async.
    void* provider = nullptr;
    check(at<InitProvider>(gs, 3)(gs, nullptr, nullptr, false, &provider) == E_INVALIDARG,
          "slot 3 (InitializeProvider) refuses a null configuration id");
    check(at<InitProvider>(gs, 3)(gs, nullptr, "scid-test", false, &provider) == S_OK && provider &&
              fs::is_directory(root / "local-user"),
          "slot 3 gives a provider and creates <LEONIDA_SAVE_DIR>\\local-user");
    {
        XAsyncBlock async{};
        void* p2 = nullptr;
        check(at<InitAsync>(gs, 4)(gs, nullptr, "scid-test", false, &async) == S_OK && wait(async) &&
                  at<InitResult>(gs, 5)(gs, &async, &p2) == S_OK && p2 && p2 != provider,
              "slots 4/5 (InitializeProviderAsync/Result) give a second provider");
        at<Close>(gs, 6)(gs, p2);
    }

    // 16 CreateContainer: the documented name rules.
    void* container = nullptr;
    const std::string long_name(257, 'a');
    bool refused = true;
    for (const char* bad : {"", "a..b", "ends.", "has space", "dir.x/file", "/", long_name.c_str()})
        refused = refused && at<Create>(gs, 16)(gs, provider, bad, &container) == E_GS_INVALID_CONTAINER_NAME;
    check(refused, "slot 16 refuses names outside the rules (empty, '..', trailing '.', bad characters, 257 long)");
    check(at<Create>(gs, 16)(gs, provider, std::string(256, 'b').c_str(), &container) == S_OK,
          "slot 16 takes a 256-character name");
    at<Close>(gs, 17)(gs, container);
    check(at<Create>(gs, 16)(gs, provider, "slot_1/main.sav", &container) == S_OK && container,
          "slot 16 creates slot_1/main.sav (path part, then a file part with '.')");

    // Nothing exists until an update is submitted.
    Seen before;
    check(at<Info>(gs, 13)(gs, provider, "slot_1/main.sav", &before, on_container) == S_OK && before.names.empty() &&
              at<EnumAll>(gs, 14)(gs, provider, &before, on_container) == S_OK && before.names.empty(),
          "slots 13/14: a container with no submitted update is not listed");

    // 23-27: an update with two blobs.
    const std::vector<uint8_t> data_a = {1, 2, 3, 4, 5, 6, 7, 8, 9}, meta = {'h', 'i'};
    void* update = nullptr;
    check(at<CreateUpdate>(gs, 23)(gs, container, "Slot One", &update) == S_OK && update &&
              at<Write>(gs, 25)(gs, update, "data", data_a.data(), data_a.size()) == S_OK &&
              at<Write>(gs, 25)(gs, update, "meta", meta.data(), meta.size()) == S_OK,
          "slots 23/25: an update takes two blob writes");
    check(at<Write>(gs, 25)(gs, update, "data", data_a.data(), data_a.size()) == E_INVALIDARG,
          "slot 25 refuses a second change to the same blob in one update");
    check(at<Submit>(gs, 27)(gs, update) == S_OK, "slot 27 (SubmitUpdate) commits it");
    at<Close>(gs, 24)(gs, update);
    const fs::path cdir = root / "local-user" / "slot_1%2Fmain.sav";
    {
        std::ifstream f(cdir / "data", std::ios::binary);
        const std::vector<uint8_t> disk{std::istreambuf_iterator<char>(f), {}};
        check(disk == data_a && fs::is_regular_file(cdir / "meta"),
              "the blobs are files in local-user\\<encoded container>, as tools/saves.py imports them");
    }

    Seen seen;
    check(at<EnumAll>(gs, 14)(gs, provider, &seen, on_container) == S_OK && seen.names.size() == 1 &&
              seen.names[0] == "slot_1/main.sav" && seen.display[0] == "Slot One" && seen.blobs[0] == 2 &&
              seen.bytes[0] == data_a.size() + meta.size(),
          "slot 14 lists the container: decoded name, display name, 2 blobs, their total size");
    Seen blobs;
    check(at<EnumBlobs>(gs, 18)(gs, container, &blobs, on_blob) == S_OK && blobs.names.size() == 2 &&
              blobs.names[0] == "data" && blobs.bytes[0] == data_a.size() && blobs.names[1] == "meta",
          "slot 18 (EnumerateBlobInfo) lists both blobs with their sizes, not the shim's '.display-name'");

    // 20 ReadBlobData: the buffer holds the Blob array, then each name and its data; the pointers point into it.
    const size_t need = 2 * sizeof(Blob) + (5 + data_a.size()) + (5 + meta.size());
    std::vector<uint8_t> buf(need);
    uint32_t count = 0;
    check(at<Read>(gs, 20)(gs, container, nullptr, &count, need - 1, reinterpret_cast<Blob*>(buf.data())) ==
              E_GS_PROVIDED_BUFFER_TOO_SMALL,
          "slot 20 (ReadBlobData) refuses a buffer one byte short: E_GS_PROVIDED_BUFFER_TOO_SMALL");
    auto* out = reinterpret_cast<Blob*>(buf.data());
    check(at<Read>(gs, 20)(gs, container, nullptr, &count, need, out) == S_OK && count == 2 &&
              std::strcmp(out[0].info.name, "data") == 0 && out[0].info.size == data_a.size() &&
              std::memcmp(out[0].data, data_a.data(), data_a.size()) == 0 && std::strcmp(out[1].info.name, "meta") == 0 &&
              std::memcmp(out[1].data, meta.data(), meta.size()) == 0 && out[0].data > buf.data() &&
              out[1].data + meta.size() == buf.data() + need,
          "slot 20 reads every blob (null names) into exactly the documented size, pointers inside the buffer");
    const char* missing[] = {"meta", "nope"};
    count = 2;
    check(at<Read>(gs, 20)(gs, container, missing, &count, need, out) == E_GS_BLOB_NOT_FOUND,
          "slot 20 with a name that doesn't exist: E_GS_BLOB_NOT_FOUND");

    // 21/22 ReadBlobDataAsync/Result: XAsyncGetResultSize gives the packed size.
    {
        const char* one[] = {"meta"};
        XAsyncBlock async{};
        size_t size = 0;
        std::vector<uint8_t> abuf(64, 0xCD);
        uint32_t n = 0;
        check(at<ReadAsync>(gs, 21)(gs, container, one, 1, &async) == S_OK && wait(async) &&
                  XAsyncGetResultSize(&async, &size) == S_OK && size == sizeof(Blob) + 5 + meta.size() &&
                  at<ReadResult>(gs, 22)(gs, &async, abuf.size(), reinterpret_cast<Blob*>(abuf.data()), &n) == S_OK &&
                  n == 1 && std::strcmp(reinterpret_cast<Blob*>(abuf.data())->info.name, "meta") == 0 &&
                  std::memcmp(reinterpret_cast<Blob*>(abuf.data())->data, meta.data(), meta.size()) == 0,
              "slots 21/22 read one named blob; XAsyncGetResultSize is its packed size, the count comes back");
    }

    // A second update: overwrite one blob, delete the other, through SubmitUpdateAsync.
    const std::vector<uint8_t> data_b = {42, 43};
    {
        XAsyncBlock async{};
        check(at<CreateUpdate>(gs, 23)(gs, container, nullptr, &update) == S_OK &&
                  at<Write>(gs, 25)(gs, update, "data", data_b.data(), data_b.size()) == S_OK &&
                  at<DeleteBlob>(gs, 26)(gs, update, "meta") == S_OK &&
                  at<SubmitAsync>(gs, 28)(gs, update, &async) == S_OK && wait(async) &&
                  at<AsyncResult>(gs, 29)(gs, &async) == S_OK,
              "slots 26/28/29: delete one blob and overwrite the other through SubmitUpdateAsync");
        at<Close>(gs, 24)(gs, update);
        Seen after;
        at<EnumBlobs>(gs, 18)(gs, container, &after, on_blob);
        Seen info;
        at<Info>(gs, 13)(gs, provider, "slot_1/main.sav", &info, on_container);
        check(after.names.size() == 1 && after.names[0] == "data" && after.bytes[0] == data_b.size() &&
                  info.display.size() == 1 && info.display[0] == "Slot One",
              "the container now holds the new data only, and keeps its display name");
    }

    // A failing update leaves the container as it was, and no staging folder behind.
    {
        check(at<CreateUpdate>(gs, 23)(gs, container, nullptr, &update) == S_OK &&
                  at<Write>(gs, 25)(gs, update, "extra", meta.data(), meta.size()) == S_OK &&
                  at<DeleteBlob>(gs, 26)(gs, update, "not-there") == S_OK &&
                  at<Submit>(gs, 27)(gs, update) == E_GS_BLOB_NOT_FOUND,
              "an update deleting a blob that doesn't exist fails with E_GS_BLOB_NOT_FOUND");
        at<Close>(gs, 24)(gs, update);
        Seen after;
        at<EnumBlobs>(gs, 18)(gs, container, &after, on_blob);
        bool hidden = false;
        for (const auto& e : fs::directory_iterator(root / "local-user")) hidden |= e.path().filename().string()[0] == '.';
        check(after.names.size() == 1 && after.names[0] == "data" && !hidden,
              "the failed update changed nothing (no 'extra' blob) and left no staging folder");
    }

    // Names on disk use tools/saves.py's xgs_encode: '.' kept except first, every other byte outside [A-Za-z0-9_-] as
    // %XX. ".hidden" is a valid container name (its file part may contain '.'), and must not become a '.' folder.
    {
        void* dotted = nullptr;
        const uint8_t one = 1;
        check(at<Create>(gs, 16)(gs, provider, ".hidden", &dotted) == S_OK &&
                  at<CreateUpdate>(gs, 23)(gs, dotted, nullptr, &update) == S_OK &&
                  at<Write>(gs, 25)(gs, update, "a b.c", &one, 1) == S_OK && at<Submit>(gs, 27)(gs, update) == S_OK &&
                  fs::is_regular_file(root / "local-user" / "%2Ehidden" / "a%20b.c"),
              "names are %-encoded as tools/saves.py does: .hidden -> %2Ehidden, 'a b.c' -> a%20b.c");
        Seen listed;
        at<Info>(gs, 13)(gs, provider, ".hidden", &listed, on_container);
        check(listed.names.size() == 1 && listed.names[0] == ".hidden", "and decode back to the names the title used");
        at<Close>(gs, 24)(gs, update);
        at<Delete>(gs, 10)(gs, provider, ".hidden");
        at<Close>(gs, 17)(gs, dotted);
    }

    // A save written in the import layout (tools/saves.py) reads back through the API.
    {
        fs::create_directories(root / "local-user" / "imported" , ec);
        std::ofstream(root / "local-user" / "imported" / "story%20save.bin", std::ios::binary) << "JSAV";
        void* imported = nullptr;
        std::vector<uint8_t> ib(sizeof(Blob) + 15 + 4);
        uint32_t n = 0;
        check(at<Create>(gs, 16)(gs, provider, "imported", &imported) == S_OK &&
                  at<Read>(gs, 20)(gs, imported, nullptr, &n, ib.size(), reinterpret_cast<Blob*>(ib.data())) == S_OK &&
                  n == 1 && std::strcmp(reinterpret_cast<Blob*>(ib.data())->info.name, "story save.bin") == 0 &&
                  std::memcmp(reinterpret_cast<Blob*>(ib.data())->data, "JSAV", 4) == 0,
              "a blob in tools/saves.py's layout reads back with its decoded name");
        at<Close>(gs, 17)(gs, imported);
    }

    // 7/8/9 quota: the documented 256 MB per user, less what is stored.
    {
        int64_t left = 0, left_async = 0;
        XAsyncBlock async{};
        check(at<Quota>(gs, 7)(gs, provider, &left) == S_OK && left > 0 && left <= (256ll << 20) - 6 &&
                  at<QuotaAsync>(gs, 8)(gs, provider, &async) == S_OK && wait(async) &&
                  at<QuotaResult>(gs, 9)(gs, &async, &left_async) == S_OK && left_async == left,
              "slots 7/8/9: the remaining quota is under 256 MB less the stored bytes, sync and async agree");
    }

    // Prefix enumeration and early stop.
    {
        void* second = nullptr;
        at<Create>(gs, 16)(gs, provider, "slot_2", &second);
        at<CreateUpdate>(gs, 23)(gs, second, "Two", &update);
        at<Write>(gs, 25)(gs, update, "x", meta.data(), meta.size());
        at<Submit>(gs, 27)(gs, update);
        at<Close>(gs, 24)(gs, update);
        using EnumPrefix = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const char*, void*, decltype(&on_container));
        Seen prefixed, stopped;
        stopped.stop_after = 1;
        check(at<EnumPrefix>(gs, 15)(gs, provider, "slot_", &prefixed, on_container) == S_OK &&
                  prefixed.names.size() == 2 && at<EnumAll>(gs, 14)(gs, provider, &stopped, on_container) == S_OK &&
                  stopped.names.size() == 1,
              "slot 15 filters by prefix; returning false from the callback stops the enumeration");
        at<Close>(gs, 17)(gs, second);
    }

    // 10/11/12 DeleteContainer.
    {
        XAsyncBlock async{};
        Seen left;
        check(at<DeleteAsync>(gs, 11)(gs, provider, "slot_2", &async) == S_OK && wait(async) &&
                  at<AsyncResult>(gs, 12)(gs, &async) == S_OK &&
                  at<Delete>(gs, 10)(gs, provider, "slot_1/main.sav") == S_OK &&
                  at<Delete>(gs, 10)(gs, provider, "never_made") == S_OK &&
                  at<EnumAll>(gs, 14)(gs, provider, &left, on_container) == S_OK && left.names.size() == 1 &&
                  left.names[0] == "imported",
              "slots 10-12 delete containers (only 'imported' is left); deleting one that doesn't exist succeeds");
    }

    // 30/31/32 XGameSaveFiles: a plain folder per configuration.
    {
        XAsyncBlock async{};
        char folder[MAX_PATH] = {};
        int64_t files_left = 0;
        check(at<FolderAsync>(gs, 30)(gs, nullptr, "scid-test", &async) == S_OK && wait(async) &&
                  at<FolderResult>(gs, 31)(gs, &async, sizeof(folder), folder) == S_OK && fs::is_directory(folder) &&
                  at<FilesQuota>(gs, 32)(gs, nullptr, "scid-test", &files_left) == S_OK && files_left > 0,
              "slots 30/31/32: XGameSaveFiles gives an existing folder and a quota");
    }

    // A closed handle is refused.
    at<Close>(gs, 17)(gs, container);
    void* after_close = nullptr;
    check(at<CreateUpdate>(gs, 23)(gs, container, nullptr, &after_close) == E_GAMERUNTIME_INVALID_HANDLE,
          "a closed container handle: E_GAMERUNTIME_INVALID_HANDLE");
    at<Close>(gs, 6)(gs, provider);
    int64_t q = 0;
    check(at<Quota>(gs, 7)(gs, provider, &q) == E_GAMERUNTIME_INVALID_HANDLE,
          "a closed provider handle: E_GAMERUNTIME_INVALID_HANDLE");

    fs::remove_all(root, ec);
    SetEnvironmentVariableW(L"LEONIDA_SAVE_DIR", nullptr);
    std::printf(g_failures ? "xgamesave_test: %d of %d checks failed\n" : "xgamesave_test: OK (%d checks)\n",
                g_failures ? g_failures : g_checks, g_checks);
    return g_failures ? 1 : 0;
}
