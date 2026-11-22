// XGameSaveImpl (asked for as IXGameSaveImpl3), offline and disk-backed.
//
// IXGameSaveImpl3 : IXGameSaveImpl2 : IXGameSaveImpl (xgameruntime include/xgamesave.idl, LGPL); the class id is
// IXGameSaveImpl's. 27 IXGameSaveImpl methods (slots 3-29), 3 IXGameSaveImpl2 methods (30-32, XGameSaveFiles),
// IXGameSaveImpl3 adds none. Behaviour and error codes follow Microsoft's XGameSave reference and its "Debugging Game
// Saves" error table.
//
// Storage is the layout docs/saves.md describes and tools/saves.py imports into:
//   <LEONIDA_SAVE_DIR>\local-user\<container>\<blob>      (default %LOCALAPPDATA%\Leonida\saves)
// with container and blob names %-encoded by encode_name() (the same rule as tools/saves.py's xgs_encode), so an
// encoded name never starts with '.'. Files and folders starting with '.' are the shim's own: a container's
// `.display-name`, and the staging and backup folders of an update being committed.
//
// A container exists once an update to it has been submitted (CreateContainer writes nothing). An update is applied
// atomically per container: the new contents are built in a staging folder next to it and swapped in by renaming, so
// a failure leaves the previous container as it was. The user handle is not checked: offline there is one local user.
// The quota is the documented default, 256 MB per user, capped by the free space of the save root's drive.

#include "xgameruntime_save.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include <XAsync.h>
#include <XAsyncProvider.h>

namespace fs = std::filesystem;

namespace {

// Microsoft's XGameSave error table ("Debugging Game Saves", Common errors).
constexpr HRESULT E_GS_INVALID_CONTAINER_NAME = static_cast<HRESULT>(0x80830001);
constexpr HRESULT E_GS_OUT_OF_LOCAL_STORAGE = static_cast<HRESULT>(0x80830003);
constexpr HRESULT E_GS_QUOTA_EXCEEDED = static_cast<HRESULT>(0x80830006);   // the table's E_QUOTA_EXCEEDED
constexpr HRESULT E_GS_PROVIDED_BUFFER_TOO_SMALL = static_cast<HRESULT>(0x80830007);
constexpr HRESULT E_GS_BLOB_NOT_FOUND = static_cast<HRESULT>(0x80830008);
constexpr HRESULT E_GAMERUNTIME_INVALID_HANDLE = static_cast<HRESULT>(0x8924010C);
constexpr uint64_t kUserQuota = 256ull << 20;   // "By default, this quota is 256 MB."

// The structures the XGameSave functions take (xgamesave.idl; x64 layout).
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
    int64_t last_modified_time;   // time_t, seconds since 1970
    bool needs_sync;
};
static_assert(sizeof(Blob) == 24 && sizeof(ContainerInfo) == 48, "XGameSave structure layouts");
using BlobInfoCallback = BOOLEAN(CALLBACK*)(const BlobInfo*, void*);
using ContainerInfoCallback = BOOLEAN(CALLBACK*)(const ContainerInfo*, void*);

// ---- names ----

// tools/saves.py's xgs_encode: keep [A-Za-z0-9_-], and '.' except as the first character; every other UTF-8 byte
// becomes %XX (uppercase hex).
std::string encode_name(const std::string& name) {
    static const char kHex[] = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : name) {
        if ((c < 0x80 && std::isalnum(c)) || c == '_' || c == '-' || (c == '.' && !out.empty())) {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += kHex[c >> 4];
            out += kHex[c & 15];
        }
    }
    return out;
}

// The inverse of encode_name. A '%' not followed by two hex digits is kept as it is.
std::string decode_name(const std::string& encoded) {
    const auto hex = [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; };
    std::string out;
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] == '%' && i + 2 < encoded.size() && hex(encoded[i + 1]) && hex(encoded[i + 2])) {
            out += static_cast<char>(std::stoi(encoded.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out += encoded[i];
        }
    }
    return out;
}

// XGameSaveCreateContainer's rules: at most 256 characters; the path part (up to the last '/', may be empty) of
// [A-Za-z0-9_/]; the file part not empty, of [A-Za-z0-9_.-], not ending in '.', without "..".
bool valid_container_name(const char* name) {
    if (!name) return false;
    const std::string s = name;
    if (s.empty() || s.size() > 256) return false;
    const size_t slash = s.rfind('/');
    const std::string path = slash == std::string::npos ? std::string() : s.substr(0, slash);
    const std::string file = slash == std::string::npos ? s : s.substr(slash + 1);
    for (const unsigned char c : path)
        if (!(c < 0x80 && (std::isalnum(c) || c == '_' || c == '/'))) return false;
    if (file.empty() || file.back() == '.' || file.find("..") != std::string::npos) return false;
    for (const unsigned char c : file)
        if (!(c < 0x80 && (std::isalnum(c) || c == '_' || c == '.' || c == '-'))) return false;
    return true;
}

// ---- the save root ----

fs::path save_root() {
    wchar_t buf[MAX_PATH] = {};
    DWORD n = GetEnvironmentVariableW(L"LEONIDA_SAVE_DIR", buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return fs::path(buf);
    n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return fs::path(buf) / L"Leonida" / L"saves";
    return fs::temp_directory_path() / L"Leonida" / L"saves";
}

HRESULT from_error(const std::error_code& ec) {
    if (!ec) return S_OK;
    if (ec.value() == ERROR_DISK_FULL || ec.value() == ERROR_HANDLE_DISK_FULL) return E_GS_OUT_OF_LOCAL_STORAGE;
    return HRESULT_FROM_WIN32(static_cast<DWORD>(ec.value()));
}

uint64_t folder_bytes(const fs::path& dir) {
    uint64_t total = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec)) total += it->file_size(ec);
    return total;
}

// What the user may still store: the quota less what the user's containers hold, and no more than the drive has.
int64_t remaining_quota(const fs::path& user_root) {
    const uint64_t used = folder_bytes(user_root);
    uint64_t left = used >= kUserQuota ? 0 : kUserQuota - used;
    std::error_code ec;
    const fs::space_info space = fs::space(user_root, ec);
    if (!ec) left = std::min<uint64_t>(left, space.available);
    return static_cast<int64_t>(left);
}

// ---- the handles ----

struct Provider {
    fs::path user_root;   // <save root>\local-user
};
struct Container {
    Provider* provider;
    std::string name;
    fs::path dir;
};
struct Update {
    Container* container;
    std::string display_name;
    std::map<std::string, std::optional<std::vector<uint8_t>>> blobs;   // nullopt: delete the blob
};

std::mutex g_handles_mutex;
std::set<const void*> g_handles;

template <class T> T* track(T* p) {
    std::lock_guard lock(g_handles_mutex);
    g_handles.insert(p);
    return p;
}
template <class T> bool live(const T* p) {
    std::lock_guard lock(g_handles_mutex);
    return p && g_handles.count(p) != 0;
}
template <class T> void untrack(T* p) {
    {
        std::lock_guard lock(g_handles_mutex);
        if (!g_handles.erase(p)) return;
    }
    delete p;
}

// One lock for the save folders: an update's swap and the reads that list or open blobs don't interleave.
std::mutex g_storage_mutex;

// ---- reading a container ----

struct StoredBlob {
    std::string name;
    fs::path file;
    uint64_t size;
};

// The blobs a finalized container holds, by name. Empty when the container doesn't exist yet.
std::vector<StoredBlob> list_blobs(const fs::path& dir) {
    std::vector<StoredBlob> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string leaf = it->path().filename().string();
        if (leaf.empty() || leaf[0] == '.' || !it->is_regular_file(ec)) continue;
        out.push_back({decode_name(leaf), it->path(), it->file_size(ec)});
    }
    std::sort(out.begin(), out.end(), [](const StoredBlob& a, const StoredBlob& b) { return a.name < b.name; });
    return out;
}

bool read_file(const fs::path& file, std::vector<uint8_t>& out) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

// The blobs a read names (or all of them), with their data.
struct ReadSet {
    std::vector<std::string> names;
    std::vector<std::vector<uint8_t>> data;
    size_t packed_size() const {
        size_t n = names.size() * sizeof(Blob);
        for (size_t i = 0; i < names.size(); ++i) n += names[i].size() + 1 + data[i].size();
        return n;
    }
};

HRESULT load_blobs(const Container& c, const std::vector<std::string>* wanted, ReadSet& out) {
    std::lock_guard lock(g_storage_mutex);
    const std::vector<StoredBlob> stored = list_blobs(c.dir);
    auto load = [&](const StoredBlob& b) {
        std::vector<uint8_t> bytes;
        if (!read_file(b.file, bytes)) return false;
        out.names.push_back(b.name);
        out.data.push_back(std::move(bytes));
        return true;
    };
    if (!wanted) {
        for (const StoredBlob& b : stored)
            if (!load(b)) return E_FAIL;
        return S_OK;
    }
    for (const std::string& name : *wanted) {
        const auto it = std::find_if(stored.begin(), stored.end(), [&](const StoredBlob& b) { return b.name == name; });
        if (it == stored.end()) return E_GS_BLOB_NOT_FOUND;
        if (!load(*it)) return E_FAIL;
    }
    return S_OK;
}

// The documented buffer: one Blob per entry first, then each entry's name and data; the pointers point into it.
void pack_blobs(const ReadSet& set, uint8_t* buffer) {
    auto* blobs = reinterpret_cast<Blob*>(buffer);
    uint8_t* at = buffer + set.names.size() * sizeof(Blob);
    for (size_t i = 0; i < set.names.size(); ++i) {
        std::memcpy(at, set.names[i].c_str(), set.names[i].size() + 1);
        blobs[i].info.name = reinterpret_cast<const char*>(at);
        at += set.names[i].size() + 1;
        if (!set.data[i].empty()) std::memcpy(at, set.data[i].data(), set.data[i].size());
        blobs[i].info.size = static_cast<uint32_t>(set.data[i].size());
        blobs[i].data = at;
        at += set.data[i].size();
    }
}

std::optional<std::vector<std::string>> names_of(const char** blob_names, uint32_t count) {
    if (!blob_names) return std::nullopt;
    std::vector<std::string> out;
    for (uint32_t i = 0; i < count; ++i) out.emplace_back(blob_names[i] ? blob_names[i] : "");
    return out;
}

// ---- container info ----

std::string read_display_name(const fs::path& dir, const std::string& fallback) {
    std::vector<uint8_t> bytes;
    if (!read_file(dir / L".display-name", bytes) || bytes.empty()) return fallback;
    return std::string(bytes.begin(), bytes.end());
}

int64_t modified_time(const fs::path& dir) {
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (!GetFileAttributesExW(dir.c_str(), GetFileExInfoStandard, &a)) return 0;
    const uint64_t ft = (uint64_t(a.ftLastWriteTime.dwHighDateTime) << 32) | a.ftLastWriteTime.dwLowDateTime;
    return static_cast<int64_t>(ft / 10000000ull) - 11644473600ll;   // FILETIME (1601, 100 ns) to Unix seconds
}

// Calls `callback` for each finalized container whose name starts with `prefix` (all of them for null), in name
// order, until it returns false.
HRESULT enumerate_containers(const Provider& p, const char* prefix, void* context, ContainerInfoCallback callback,
                             const std::string* exact = nullptr) {
    struct Entry {
        std::string name, display;
        uint32_t blobs;
        uint64_t bytes;
        int64_t time;
    };
    std::vector<Entry> entries;
    {
        std::lock_guard lock(g_storage_mutex);
        std::error_code ec;
        for (fs::directory_iterator it(p.user_root, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string leaf = it->path().filename().string();
            if (leaf.empty() || leaf[0] == '.' || !it->is_directory(ec)) continue;
            const std::string name = decode_name(leaf);
            if (exact && name != *exact) continue;
            if (prefix && name.compare(0, std::strlen(prefix), prefix) != 0) continue;
            uint64_t bytes = 0;
            const std::vector<StoredBlob> blobs = list_blobs(it->path());
            for (const StoredBlob& b : blobs) bytes += b.size;
            entries.push_back({name, read_display_name(it->path(), name), static_cast<uint32_t>(blobs.size()), bytes,
                               modified_time(it->path())});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.name < b.name; });
    for (const Entry& e : entries) {
        const ContainerInfo info{e.name.c_str(), e.display.c_str(), e.blobs, e.bytes, e.time, false};
        if (!callback(&info, context)) break;
    }
    return S_OK;
}

// ---- committing an update ----

std::atomic<uint64_t> g_staging_serial{0};

bool write_file(const fs::path& file, const uint8_t* data, size_t size, std::error_code& ec) {
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    if (!f) {
        ec = std::error_code(static_cast<int>(GetLastError() ? GetLastError() : ERROR_WRITE_FAULT), std::system_category());
        return false;
    }
    if (size) f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    f.close();
    if (f.fail()) {
        ec = std::error_code(ERROR_WRITE_FAULT, std::system_category());
        return false;
    }
    return true;
}

HRESULT commit(Update& u) {
    Container& c = *u.container;
    std::lock_guard lock(g_storage_mutex);
    std::error_code ec;
    fs::create_directories(c.provider->user_root, ec);
    if (ec) return from_error(ec);
    const std::string leaf = c.dir.filename().string();
    const std::string serial = std::to_string(GetCurrentProcessId()) + "-" + std::to_string(++g_staging_serial);
    const fs::path staging = c.provider->user_root / ("." + leaf + ".staging-" + serial);
    const fs::path backup = c.provider->user_root / ("." + leaf + ".old-" + serial);
    auto fail = [&](HRESULT hr) {
        std::error_code ignore;
        fs::remove_all(staging, ignore);
        return hr;
    };

    // The new contents: the current blobs, then the update's writes and deletes.
    fs::create_directory(staging, ec);
    if (ec) return fail(from_error(ec));
    const bool existed = fs::is_directory(c.dir, ec);
    if (existed) {
        fs::copy(c.dir, staging, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
        if (ec) return fail(from_error(ec));
    }
    for (const auto& [name, data] : u.blobs) {
        const fs::path file = staging / encode_name(name);
        if (!data) {
            if (!fs::remove(file, ec) && !ec) return fail(E_GS_BLOB_NOT_FOUND);
            if (ec) return fail(from_error(ec));
        } else if (!write_file(file, data->data(), data->size(), ec)) {
            return fail(from_error(ec));
        }
    }
    const std::string display = u.display_name.empty() ? read_display_name(c.dir, c.name) : u.display_name;
    if (!write_file(staging / L".display-name", reinterpret_cast<const uint8_t*>(display.data()), display.size(), ec))
        return fail(from_error(ec));

    // The user's whole store after the update must stay within the quota. The staging folder is inside the user's
    // folder, so the total already counts the new contents; the old container's go.
    const uint64_t after = folder_bytes(c.provider->user_root) - (existed ? folder_bytes(c.dir) : 0);
    if (after > kUserQuota) return fail(E_GS_QUOTA_EXCEEDED);

    // Swap: the old container aside, the new one in, then drop the old one. A failed second rename puts it back.
    if (existed) {
        fs::rename(c.dir, backup, ec);
        if (ec) return fail(from_error(ec));
    }
    fs::rename(staging, c.dir, ec);
    if (ec) {
        std::error_code restore;
        if (existed) fs::rename(backup, c.dir, restore);
        return fail(from_error(ec));
    }
    if (existed) fs::remove_all(backup, ec);   // a leftover backup is hidden ('.') and only costs space
    u.blobs.clear();
    return S_OK;
}

// ---- XAsync helpers ----

// Runs `work` on the async block's queue and completes it with work's HRESULT and result bytes, which the matching
// *Result function reads back with XAsyncGetResult (so XAsyncGetResultSize reports their size first).
struct AsyncJob {
    std::function<HRESULT(std::vector<uint8_t>&)> work;
    std::vector<uint8_t> result;
};

HRESULT CALLBACK job_provider(XAsyncOp op, const XAsyncProviderData* data) {
    auto* job = static_cast<AsyncJob*>(data->context);
    switch (op) {
        case XAsyncOp::Begin:
            return XAsyncSchedule(data->async, 0);
        case XAsyncOp::DoWork: {
            const HRESULT hr = job->work(job->result);
            XAsyncComplete(data->async, hr, SUCCEEDED(hr) ? job->result.size() : 0);
            return S_OK;
        }
        case XAsyncOp::GetResult:
            if (data->bufferSize < job->result.size()) return E_NOT_SUFFICIENT_BUFFER;
            if (!job->result.empty()) std::memcpy(data->buffer, job->result.data(), job->result.size());
            return S_OK;
        case XAsyncOp::Cleanup:
            delete job;
            return S_OK;
        default:
            return S_OK;
    }
}

HRESULT begin_job(XAsyncBlock* async, const void* identity, const char* name,
                  std::function<HRESULT(std::vector<uint8_t>&)> work) {
    if (!async) return E_INVALIDARG;
    auto* job = new AsyncJob{std::move(work), {}};
    const HRESULT hr = XAsyncBegin(async, job, identity, name, job_provider);
    if (FAILED(hr)) delete job;
    return hr;
}

template <class T> std::vector<uint8_t> bytes_of(const T& v) {
    std::vector<uint8_t> out(sizeof(T));
    std::memcpy(out.data(), &v, sizeof(T));
    return out;
}

template <class T> HRESULT job_value(XAsyncBlock* async, const void* identity, T* out) {
    if (!async || !out) return E_INVALIDARG;
    T value{};
    size_t used = 0;
    const HRESULT hr = XAsyncGetResult(async, identity, sizeof(T), &value, &used);
    if (SUCCEEDED(hr)) *out = value;
    return hr;
}

// One identity per async function, so a *Result call with another function's block is refused. Writable objects, so
// the linker can't fold them into one address.
char kInitIdentity, kQuotaIdentity, kDeleteIdentity, kReadIdentity, kSubmitIdentity, kFolderIdentity;

// ---- the methods (slot order of xgamesave.idl) ----

HRESULT STDMETHODCALLTYPE initialize_provider(void*, void*, const char* configuration_id, bool, Provider** provider) {
    if (!configuration_id || !provider) return E_INVALIDARG;
    *provider = nullptr;
    const fs::path user_root = save_root() / L"local-user";
    std::error_code ec;
    fs::create_directories(user_root, ec);
    if (ec) return from_error(ec);
    *provider = track(new Provider{user_root});
    return S_OK;
}
HRESULT STDMETHODCALLTYPE initialize_provider_async(void* self, void* user, const char* configuration_id,
                                                    bool sync_on_demand, XAsyncBlock* async) {
    if (!configuration_id) return E_INVALIDARG;
    const std::string id = configuration_id;
    return begin_job(async, &kInitIdentity, "XGameSaveInitializeProviderAsync",
                     [self, user, id, sync_on_demand](std::vector<uint8_t>& out) {
                         Provider* p = nullptr;
                         const HRESULT hr = initialize_provider(self, user, id.c_str(), sync_on_demand, &p);
                         if (SUCCEEDED(hr)) out = bytes_of(p);
                         return hr;
                     });
}
HRESULT STDMETHODCALLTYPE initialize_provider_result(void*, XAsyncBlock* async, Provider** provider) {
    return job_value(async, &kInitIdentity, provider);
}
void STDMETHODCALLTYPE close_provider(void*, Provider* provider) { untrack(provider); }

HRESULT STDMETHODCALLTYPE get_remaining_quota(void*, Provider* provider, int64_t* remaining) {
    if (!live(provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!remaining) return E_INVALIDARG;
    std::lock_guard lock(g_storage_mutex);
    *remaining = remaining_quota(provider->user_root);
    return S_OK;
}
HRESULT STDMETHODCALLTYPE get_remaining_quota_async(void* self, Provider* provider, XAsyncBlock* async) {
    if (!live(provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    return begin_job(async, &kQuotaIdentity, "XGameSaveGetRemainingQuotaAsync", [self, provider](std::vector<uint8_t>& out) {
        int64_t left = 0;
        const HRESULT hr = get_remaining_quota(self, provider, &left);
        if (SUCCEEDED(hr)) out = bytes_of(left);
        return hr;
    });
}
HRESULT STDMETHODCALLTYPE get_remaining_quota_result(void*, XAsyncBlock* async, int64_t* remaining) {
    return job_value(async, &kQuotaIdentity, remaining);
}

// Deleting a container that doesn't exist succeeds (the reference names no error for it).
HRESULT STDMETHODCALLTYPE delete_container(void*, Provider* provider, const char* name) {
    if (!live(provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!valid_container_name(name)) return E_GS_INVALID_CONTAINER_NAME;
    std::lock_guard lock(g_storage_mutex);
    std::error_code ec;
    fs::remove_all(provider->user_root / encode_name(name), ec);
    return from_error(ec);
}
HRESULT STDMETHODCALLTYPE delete_container_async(void* self, Provider* provider, const char* name, XAsyncBlock* async) {
    if (!live(provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!valid_container_name(name)) return E_GS_INVALID_CONTAINER_NAME;
    const std::string n = name;
    return begin_job(async, &kDeleteIdentity, "XGameSaveDeleteContainerAsync",
                     [self, provider, n](std::vector<uint8_t>&) { return delete_container(self, provider, n.c_str()); });
}
HRESULT STDMETHODCALLTYPE delete_container_result(void*, XAsyncBlock* async) {
    size_t used = 0;
    return async ? XAsyncGetResult(async, &kDeleteIdentity, 0, nullptr, &used) : E_INVALIDARG;
}

// A container that doesn't exist yet gives no callback ("will return only one result if found").
HRESULT STDMETHODCALLTYPE get_container_info(void*, Provider* provider, const char* name, void* context,
                                             ContainerInfoCallback callback) {
    if (!live(provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!valid_container_name(name)) return E_GS_INVALID_CONTAINER_NAME;
    if (!callback) return E_INVALIDARG;
    const std::string exact = name;
    return enumerate_containers(*provider, nullptr, context, callback, &exact);
}
HRESULT STDMETHODCALLTYPE enumerate_container_info(void*, Provider* provider, void* context,
                                                   ContainerInfoCallback callback) {
    if (!live(provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!callback) return E_INVALIDARG;
    return enumerate_containers(*provider, nullptr, context, callback);
}
HRESULT STDMETHODCALLTYPE enumerate_container_info_by_name(void*, Provider* provider, const char* prefix, void* context,
                                                           ContainerInfoCallback callback) {
    if (!live(provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!callback) return E_INVALIDARG;
    return enumerate_containers(*provider, prefix, context, callback);
}

HRESULT STDMETHODCALLTYPE create_container(void*, Provider* provider, const char* name, Container** container) {
    if (!live(provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!container) return E_INVALIDARG;
    *container = nullptr;
    if (!valid_container_name(name)) return E_GS_INVALID_CONTAINER_NAME;
    *container = track(new Container{provider, name, provider->user_root / encode_name(name)});
    return S_OK;
}
void STDMETHODCALLTYPE close_container(void*, Container* container) { untrack(container); }

HRESULT enumerate_blobs(Container* container, const char* prefix, void* context, BlobInfoCallback callback) {
    if (!live(container)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!callback) return E_INVALIDARG;
    std::vector<StoredBlob> blobs;
    {
        std::lock_guard lock(g_storage_mutex);
        blobs = list_blobs(container->dir);
    }
    for (const StoredBlob& b : blobs) {
        if (prefix && b.name.compare(0, std::strlen(prefix), prefix) != 0) continue;
        const BlobInfo info{b.name.c_str(), static_cast<uint32_t>(b.size)};
        if (!callback(&info, context)) break;
    }
    return S_OK;
}
HRESULT STDMETHODCALLTYPE enumerate_blob_info(void*, Container* container, void* context, BlobInfoCallback callback) {
    return enumerate_blobs(container, nullptr, context, callback);
}
HRESULT STDMETHODCALLTYPE enumerate_blob_info_by_name(void*, Container* container, const char* prefix, void* context,
                                                      BlobInfoCallback callback) {
    return enumerate_blobs(container, prefix, context, callback);
}

// blobNames null reads every blob. countOfBlobs is the number of names in, the number of blobs read out.
HRESULT STDMETHODCALLTYPE read_blob_data(void*, Container* container, const char** blob_names, uint32_t* count,
                                         size_t blobs_size, Blob* blob_data) {
    if (!live(container)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!count || (!blob_data && blobs_size)) return E_INVALIDARG;
    const std::optional<std::vector<std::string>> names = names_of(blob_names, *count);
    ReadSet set;
    const HRESULT hr = load_blobs(*container, names ? &*names : nullptr, set);
    if (FAILED(hr)) return hr;
    if (blobs_size < set.packed_size()) return E_GS_PROVIDED_BUFFER_TOO_SMALL;
    if (!set.names.empty()) pack_blobs(set, reinterpret_cast<uint8_t*>(blob_data));
    *count = static_cast<uint32_t>(set.names.size());
    return S_OK;
}

// The async read loads the blobs in DoWork and packs them straight into the caller's buffer in XAsyncGetResult, so
// the pointers in the result point into that buffer. The result size XAsyncGetResultSize reports is the packed size.
struct ReadJob {
    Container* container;
    std::optional<std::vector<std::string>> names;
    ReadSet set;
};
HRESULT CALLBACK read_provider(XAsyncOp op, const XAsyncProviderData* data) {
    auto* job = static_cast<ReadJob*>(data->context);
    switch (op) {
        case XAsyncOp::Begin:
            return XAsyncSchedule(data->async, 0);
        case XAsyncOp::DoWork: {
            const HRESULT hr = live(job->container)
                                   ? load_blobs(*job->container, job->names ? &*job->names : nullptr, job->set)
                                   : E_GAMERUNTIME_INVALID_HANDLE;
            XAsyncComplete(data->async, hr, SUCCEEDED(hr) ? job->set.packed_size() : 0);
            return S_OK;
        }
        case XAsyncOp::GetResult:
            if (data->bufferSize < job->set.packed_size()) return E_GS_PROVIDED_BUFFER_TOO_SMALL;
            if (!job->set.names.empty()) pack_blobs(job->set, static_cast<uint8_t*>(data->buffer));
            return S_OK;
        case XAsyncOp::Cleanup:
            delete job;
            return S_OK;
        default:
            return S_OK;
    }
}
HRESULT STDMETHODCALLTYPE read_blob_data_async(void*, Container* container, const char** blob_names, uint32_t count,
                                               XAsyncBlock* async) {
    if (!live(container)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!async) return E_INVALIDARG;
    auto* job = new ReadJob{container, names_of(blob_names, count), {}};
    const HRESULT hr = XAsyncBegin(async, job, &kReadIdentity, "XGameSaveReadBlobDataAsync", read_provider);
    if (FAILED(hr)) delete job;
    return hr;
}
HRESULT STDMETHODCALLTYPE read_blob_data_result(void*, XAsyncBlock* async, size_t blobs_size, Blob* blob_data,
                                                uint32_t* count) {
    if (!async || !count) return E_INVALIDARG;
    size_t needed = 0;
    HRESULT hr = XAsyncGetResultSize(async, &needed);
    if (FAILED(hr)) return hr;
    if (blobs_size < needed) return E_GS_PROVIDED_BUFFER_TOO_SMALL;
    size_t used = 0;
    hr = XAsyncGetResult(async, &kReadIdentity, blobs_size, blob_data, &used);
    if (FAILED(hr)) return hr;
    // The Blob array comes first and the first name follows it, so the first name's offset gives the count.
    *count = used == 0 ? 0
                       : static_cast<uint32_t>((reinterpret_cast<const uint8_t*>(blob_data[0].info.name) -
                                                reinterpret_cast<const uint8_t*>(blob_data)) / sizeof(Blob));
    return S_OK;
}

HRESULT STDMETHODCALLTYPE create_update(void*, Container* container, const char* display_name, Update** update) {
    if (!live(container)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!update) return E_INVALIDARG;
    *update = track(new Update{container, display_name ? display_name : "", {}});
    return S_OK;
}
void STDMETHODCALLTYPE close_update(void*, Update* update) { untrack(update); }

// "You can only write one modification/deletion per blob in an update context."
HRESULT stage_blob(Update* update, const char* name, std::optional<std::vector<uint8_t>> data) {
    if (!live(update)) return E_GAMERUNTIME_INVALID_HANDLE;
    if (!name || !*name) return E_INVALIDARG;
    if (!update->blobs.emplace(name, std::move(data)).second) return E_INVALIDARG;
    return S_OK;
}
HRESULT STDMETHODCALLTYPE submit_blob_write(void*, Update* update, const char* name, const uint8_t* data, size_t size) {
    if (size && !data) return E_INVALIDARG;
    if (size > UINT32_MAX) return E_INVALIDARG;   // XGameSaveBlobInfo::size is 32-bit
    return stage_blob(update, name, std::vector<uint8_t>(data, data + size));
}
HRESULT STDMETHODCALLTYPE submit_blob_delete(void*, Update* update, const char* name) {
    return stage_blob(update, name, std::nullopt);
}
HRESULT STDMETHODCALLTYPE submit_update(void*, Update* update) {
    if (!live(update) || !live(update->container) || !live(update->container->provider)) return E_GAMERUNTIME_INVALID_HANDLE;
    return commit(*update);
}
HRESULT STDMETHODCALLTYPE submit_update_async(void* self, Update* update, XAsyncBlock* async) {
    if (!live(update)) return E_GAMERUNTIME_INVALID_HANDLE;
    return begin_job(async, &kSubmitIdentity, "XGameSaveSubmitUpdateAsync",
                     [self, update](std::vector<uint8_t>&) { return submit_update(self, update); });
}
HRESULT STDMETHODCALLTYPE submit_update_result(void*, XAsyncBlock* async) {
    size_t used = 0;
    return async ? XAsyncGetResult(async, &kSubmitIdentity, 0, nullptr, &used) : E_INVALIDARG;
}

// IXGameSaveImpl2, XGameSaveFiles: a plain folder per configuration, <save root>\local-user-files\<configuration id>,
// with no UI to show offline.
fs::path files_folder(const char* configuration_id) {
    return save_root() / L"local-user-files" / encode_name(configuration_id);
}
HRESULT STDMETHODCALLTYPE files_get_folder_async(void*, void*, const char* configuration_id, XAsyncBlock* async) {
    if (!configuration_id) return E_INVALIDARG;
    const fs::path folder = files_folder(configuration_id);
    return begin_job(async, &kFolderIdentity, "XGameSaveFilesGetFolderWithUiAsync", [folder](std::vector<uint8_t>& out) {
        std::error_code ec;
        fs::create_directories(folder, ec);
        if (ec) return from_error(ec);
        const std::u8string utf8 = folder.u8string();
        out.assign(utf8.begin(), utf8.end());
        out.push_back(0);
        return S_OK;
    });
}
HRESULT STDMETHODCALLTYPE files_get_folder_result(void*, XAsyncBlock* async, size_t size, char* folder) {
    if (!async || !folder) return E_INVALIDARG;
    size_t needed = 0;
    HRESULT hr = XAsyncGetResultSize(async, &needed);
    if (FAILED(hr)) return hr;
    if (size < needed) return E_GS_PROVIDED_BUFFER_TOO_SMALL;
    size_t used = 0;
    return XAsyncGetResult(async, &kFolderIdentity, size, folder, &used);
}
HRESULT STDMETHODCALLTYPE files_get_remaining_quota(void*, void*, const char* configuration_id, int64_t* remaining) {
    if (!configuration_id || !remaining) return E_INVALIDARG;
    const fs::path folder = files_folder(configuration_id);
    std::error_code ec;
    fs::create_directories(folder, ec);
    if (ec) return from_error(ec);
    *remaining = remaining_quota(folder);
    return S_OK;
}

// ---- the object ----

constexpr GUID kSaveClass = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}};
constexpr GUID kSave2 = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6e, 0xe2}};
constexpr GUID kSave3 = {0x1bfff3af, 0xf14a, 0x40a3, {0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9}};
constexpr GUID kUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

struct SaveObject {
    const void* const* vtable;
    std::atomic<long> refs{1};
};

HRESULT STDMETHODCALLTYPE save_query_interface(void* self, REFIID iid, void** out) {
    if (!out) return E_POINTER;
    if (iid == kUnknown || iid == kSaveClass || iid == kSave2 || iid == kSave3) {
        static_cast<SaveObject*>(self)->refs.fetch_add(1);
        *out = self;
        return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
}
// A process-wide object: the count moves but never reaches zero.
ULONG STDMETHODCALLTYPE save_add_ref(void* self) {
    return static_cast<ULONG>(static_cast<SaveObject*>(self)->refs.fetch_add(1) + 1);
}
ULONG STDMETHODCALLTYPE save_release(void* self) {
    auto& refs = static_cast<SaveObject*>(self)->refs;
    long prev = refs.load();
    while (prev > 1 && !refs.compare_exchange_weak(prev, prev - 1)) {}
    return static_cast<ULONG>(prev > 1 ? prev - 1 : 1);
}

#define S(f) reinterpret_cast<const void*>(&f)
const void* const kSaveVtable[] = {
    S(save_query_interface), S(save_add_ref), S(save_release),
    S(initialize_provider),               // 3  XGameSaveInitializeProvider
    S(initialize_provider_async),         // 4  XGameSaveInitializeProviderAsync
    S(initialize_provider_result),        // 5  XGameSaveInitializeProviderResult
    S(close_provider),                    // 6  XGameSaveCloseProvider (void)
    S(get_remaining_quota),               // 7  XGameSaveGetRemainingQuota
    S(get_remaining_quota_async),         // 8  XGameSaveGetRemainingQuotaAsync
    S(get_remaining_quota_result),        // 9  XGameSaveGetRemainingQuotaResult
    S(delete_container),                  // 10 XGameSaveDeleteContainer
    S(delete_container_async),            // 11 XGameSaveDeleteContainerAsync
    S(delete_container_result),           // 12 XGameSaveDeleteContainerResult
    S(get_container_info),                // 13 XGameSaveGetContainerInfo
    S(enumerate_container_info),          // 14 XGameSaveEnumerateContainerInfo
    S(enumerate_container_info_by_name),  // 15 XGameSaveEnumerateContainerInfoByName
    S(create_container),                  // 16 XGameSaveCreateContainer
    S(close_container),                   // 17 XGameSaveCloseContainer (void)
    S(enumerate_blob_info),               // 18 XGameSaveEnumerateBlobInfo
    S(enumerate_blob_info_by_name),       // 19 XGameSaveEnumerateBlobInfoByName
    S(read_blob_data),                    // 20 XGameSaveReadBlobData
    S(read_blob_data_async),              // 21 XGameSaveReadBlobDataAsync
    S(read_blob_data_result),             // 22 XGameSaveReadBlobDataResult
    S(create_update),                     // 23 XGameSaveCreateUpdate
    S(close_update),                      // 24 XGameSaveCloseUpdate (void)
    S(submit_blob_write),                 // 25 XGameSaveSubmitBlobWrite
    S(submit_blob_delete),                // 26 XGameSaveSubmitBlobDelete
    S(submit_update),                     // 27 XGameSaveSubmitUpdate
    S(submit_update_async),               // 28 XGameSaveSubmitUpdateAsync
    S(submit_update_result),              // 29 XGameSaveSubmitUpdateResult
    S(files_get_folder_async),            // 30 XGameSaveFilesGetFolderWithUiAsync (IXGameSaveImpl2)
    S(files_get_folder_result),           // 31 XGameSaveFilesGetFolderWithUiResult
    S(files_get_remaining_quota),         // 32 XGameSaveFilesGetRemainingQuota
};
#undef S
static_assert(std::size(kSaveVtable) == 33, "IUnknown + IXGameSaveImpl's 27 + IXGameSaveImpl2's 3 methods");

SaveObject g_save{kSaveVtable};

}  // namespace

namespace rt::xgr {

bool query_save(const GUID* class_id, REFIID interface_id, void** out, HRESULT* result) {
    if (!class_id || *class_id != kSaveClass) return false;
    *result = out ? save_query_interface(&g_save, interface_id, out) : E_POINTER;
    return true;
}

}  // namespace rt::xgr
