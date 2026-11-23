// The kernel32 file calls behind MSVC's std::filesystem and fstream: attributes, directories, copies, seeks and
// handle queries, on POSIX calls. Paths go through to_host_path (GAME:, backslashes).
#include "host.h"

#include <copyfile.h>
#include <dirent.h>
#include <fnmatch.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace host {
namespace {

using HANDLE = void*;

std::string path_of(const char16_t* w) { return to_host_path(narrow(w).c_str()); }
int fd_of(HANDLE h) { return static_cast<Object*>(h)->fd; }
uint32_t win_error() {  // errno -> the closest Windows error
    switch (errno) {
        case ENOENT: return 2;   // ERROR_FILE_NOT_FOUND
        case ENOTDIR: return 3;  // ERROR_PATH_NOT_FOUND
        case EACCES: case EPERM: return 5;
        case EEXIST: return 183; // ERROR_ALREADY_EXISTS
        case ENOSPC: return 112;
        case ENOTEMPTY: return 145;  // ERROR_DIR_NOT_EMPTY: MSVC remove_all only lists a directory after this
        default: return 31;      // ERROR_GEN_FAILURE
    }
}
int failed() { return set_last_error(win_error()), 0; }

uint64_t filetime(const timespec& t) {
    return static_cast<uint64_t>(t.tv_sec) * 10000000ull + t.tv_nsec / 100 + 116444736000000000ull;
}
uint32_t attributes(const struct stat& st) {
    uint32_t a = S_ISDIR(st.st_mode) ? 0x10 : 0x80;  // DIRECTORY : NORMAL
    if (!(st.st_mode & S_IWUSR)) a |= 0x1;            // READONLY
    return a;
}
template <typename T> void put(uint8_t* p, size_t off, T v) { std::memcpy(p + off, &v, sizeof v); }

uint32_t GetFileAttributesW(const char16_t* name) {
    struct stat st;
    if (stat(path_of(name).c_str(), &st) != 0) return failed(), 0xFFFFFFFFu;
    return attributes(st);
}
int GetFileAttributesExW(const char16_t* name, int32_t, uint8_t* out) {  // WIN32_FILE_ATTRIBUTE_DATA
    struct stat st;
    if (stat(path_of(name).c_str(), &st) != 0) return failed();
    put<uint32_t>(out, 0, attributes(st));
    put<uint64_t>(out, 4, filetime(st.st_birthtimespec));
    put<uint64_t>(out, 12, filetime(st.st_atimespec));
    put<uint64_t>(out, 20, filetime(st.st_mtimespec));
    put<uint32_t>(out, 28, static_cast<uint32_t>(st.st_size >> 32));
    put<uint32_t>(out, 32, static_cast<uint32_t>(st.st_size));
    return 1;
}
int CreateDirectoryW(const char16_t* name, void*) { return mkdir(path_of(name).c_str(), 0755) == 0 ? 1 : failed(); }
int CreateDirectoryExW(const char16_t*, const char16_t* name, void* sa) { return CreateDirectoryW(name, sa); }
int CopyFileW(const char16_t* from, const char16_t* to, int32_t fail_if_exists) {
    std::string dst = path_of(to);
    if (fail_if_exists && access(dst.c_str(), F_OK) == 0) return set_last_error(80 /* ERROR_FILE_EXISTS */), 0;
    return copyfile(path_of(from).c_str(), dst.c_str(), nullptr, COPYFILE_DATA) == 0 ? 1 : failed();
}
int MoveFileExW(const char16_t* from, const char16_t* to, uint32_t) {
    return rename(path_of(from).c_str(), path_of(to).c_str()) == 0 ? 1 : failed();
}
int SetFilePointerEx(HANDLE h, int64_t distance, int64_t* pos, uint32_t method) {
    off_t r = lseek(fd_of(h), distance, method == 0 ? SEEK_SET : method == 1 ? SEEK_CUR : SEEK_END);
    if (r < 0) return failed();
    if (pos) *pos = r;
    return 1;
}
int GetFileSizeEx(HANDLE h, int64_t* size) {
    struct stat st;
    if (fstat(fd_of(h), &st) != 0) return failed();
    *size = st.st_size;
    return 1;
}
uint32_t GetFileSize(HANDLE h, uint32_t* high) {
    int64_t size = 0;
    if (!GetFileSizeEx(h, &size)) return 0xFFFFFFFFu;
    if (high) *high = static_cast<uint32_t>(size >> 32);
    return static_cast<uint32_t>(size);
}
int GetFileTime(HANDLE h, uint64_t* created, uint64_t* accessed, uint64_t* written) {
    struct stat st;
    if (fstat(fd_of(h), &st) != 0) return failed();
    if (created) *created = filetime(st.st_birthtimespec);
    if (accessed) *accessed = filetime(st.st_atimespec);
    if (written) *written = filetime(st.st_mtimespec);
    return 1;
}
int SetEndOfFile(HANDLE h) {
    off_t at = lseek(fd_of(h), 0, SEEK_CUR);
    return at >= 0 && ftruncate(fd_of(h), at) == 0 ? 1 : failed();
}
int FlushFileBuffers(HANDLE h) { return fsync(fd_of(h)) == 0 ? 1 : failed(); }
int GetFileInformationByHandle(HANDLE h, uint8_t* out) {  // BY_HANDLE_FILE_INFORMATION
    struct stat st;
    if (fstat(fd_of(h), &st) != 0) return failed();
    std::memset(out, 0, 52);
    put<uint32_t>(out, 0, attributes(st));
    put<uint64_t>(out, 4, filetime(st.st_birthtimespec));
    put<uint64_t>(out, 12, filetime(st.st_atimespec));
    put<uint64_t>(out, 20, filetime(st.st_mtimespec));
    put<uint32_t>(out, 28, static_cast<uint32_t>(st.st_dev));
    put<uint32_t>(out, 32, static_cast<uint32_t>(st.st_size >> 32));
    put<uint32_t>(out, 36, static_cast<uint32_t>(st.st_size));
    put<uint32_t>(out, 40, static_cast<uint32_t>(st.st_nlink));
    put<uint32_t>(out, 44, static_cast<uint32_t>(st.st_ino >> 32));
    put<uint32_t>(out, 48, static_cast<uint32_t>(st.st_ino));
    return 1;
}
int GetFileInformationByHandleEx(HANDLE h, int32_t cls, uint8_t* out, uint32_t size) {
    struct stat st;
    if (fstat(fd_of(h), &st) != 0) return failed();
    std::memset(out, 0, size);
    switch (cls) {
        case 0:  // FileBasicInfo
            put<uint64_t>(out, 0, filetime(st.st_birthtimespec));
            put<uint64_t>(out, 8, filetime(st.st_atimespec));
            put<uint64_t>(out, 16, filetime(st.st_mtimespec));
            put<uint64_t>(out, 24, filetime(st.st_ctimespec));
            put<uint32_t>(out, 32, attributes(st));
            return 1;
        case 1:  // FileStandardInfo
            put<int64_t>(out, 0, st.st_blocks * 512);
            put<int64_t>(out, 8, st.st_size);
            put<uint32_t>(out, 16, static_cast<uint32_t>(st.st_nlink));
            out[21] = S_ISDIR(st.st_mode);
            return 1;
        case 9:  // FileAttributeTagInfo
            put<uint32_t>(out, 0, attributes(st));
            return 1;
        case 18:  // FileIdInfo
            put<uint64_t>(out, 0, static_cast<uint64_t>(st.st_dev));
            put<uint64_t>(out, 8, static_cast<uint64_t>(st.st_ino));
            return 1;
        default:
            fail("GetFileInformationByHandleEx: class " + std::to_string(cls));
    }
}
// MSVC's std::filesystem::remove marks a handle for deletion; POSIX can unlink at once and keep the fd usable.
int SetFileInformationByHandle(HANDLE h, int32_t cls, const uint8_t* info, uint32_t) {
    bool del;
    if (cls == 4) del = info[0] != 0;              // FileDispositionInfo
    else if (cls == 21) del = info[0] & 1;         // FileDispositionInfoEx, FILE_DISPOSITION_FLAG_DELETE
    else if (cls == 0) return 1;                   // FileBasicInfo: ponytail, times are not set
    else fail("SetFileInformationByHandle: class " + std::to_string(cls));
    if (!del) return 1;
    char p[PATH_MAX];
    struct stat st;
    if (fcntl(fd_of(h), F_GETPATH, p) != 0 || fstat(fd_of(h), &st) != 0) return failed();
    return (S_ISDIR(st.st_mode) ? rmdir(p) : unlink(p)) == 0 ? 1 : failed();
}
uint32_t copy_path(const std::string& s, uint32_t n, char16_t* buf) {
    std::u16string w = widen(s.c_str());
    if (n <= w.size()) return static_cast<uint32_t>(w.size() + 1);
    std::memcpy(buf, w.c_str(), (w.size() + 1) * 2);
    return static_cast<uint32_t>(w.size());
}
uint32_t GetCurrentDirectoryW(uint32_t n, char16_t* buf) {
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof cwd)) return failed();
    return copy_path(cwd, n, buf);
}
int SetCurrentDirectoryW(const char16_t* name) { return chdir(path_of(name).c_str()) == 0 ? 1 : failed(); }
uint32_t GetFullPathNameW(const char16_t* name, uint32_t n, char16_t* buf, char16_t** file_part) {
    std::string p = path_of(name);
    if (p.empty() || p[0] != '/') {
        char cwd[PATH_MAX];
        if (!getcwd(cwd, sizeof cwd)) return failed();
        p = std::string(cwd) + "/" + p;
    }
    uint32_t r = copy_path(p, n, buf);
    if (file_part && r < n) {
        char16_t* slash = nullptr;
        for (char16_t* c = buf; *c; ++c)
            if (*c == u'/') slash = c;
        *file_part = slash && slash[1] ? slash + 1 : nullptr;
    }
    return r;
}
uint32_t GetFinalPathNameByHandleW(HANDLE h, char16_t* buf, uint32_t n, uint32_t) {
    char p[PATH_MAX];
    if (fcntl(fd_of(h), F_GETPATH, p) != 0) return failed();
    return copy_path(p, n, buf);
}

// FindFirstFile: "dir\\pattern" listed with opendir, names matched with fnmatch. WIN32_FIND_DATAW: attributes 0,
// times 4/12/20, size high 28 low 32, cFileName[260] at 44.
struct Find {
    DIR* dir;
    std::string path, pattern;
};
int next_match(Find* f, uint8_t* out) {
    while (dirent* e = readdir(f->dir)) {
        if (fnmatch(f->pattern.c_str(), e->d_name, 0) != 0) continue;
        struct stat st;
        if (stat((f->path + "/" + e->d_name).c_str(), &st) != 0) continue;
        std::memset(out, 0, 592);
        put<uint32_t>(out, 0, attributes(st));
        put<uint64_t>(out, 4, filetime(st.st_birthtimespec));
        put<uint64_t>(out, 12, filetime(st.st_atimespec));
        put<uint64_t>(out, 20, filetime(st.st_mtimespec));
        put<uint32_t>(out, 28, static_cast<uint32_t>(st.st_size >> 32));
        put<uint32_t>(out, 32, static_cast<uint32_t>(st.st_size));
        std::u16string w = widen(e->d_name).substr(0, 259);
        std::memcpy(out + 44, w.c_str(), (w.size() + 1) * 2);
        return 1;
    }
    return 0;
}
HANDLE FindFirstFileExW(const char16_t* name, int32_t, uint8_t* out, int32_t, void*, uint32_t) {
    std::string full = path_of(name);
    size_t slash = full.rfind('/');
    Find* f = new Find{nullptr, slash == std::string::npos ? "." : full.substr(0, slash ? slash : 1),
                       slash == std::string::npos ? full : full.substr(slash + 1)};
    f->dir = opendir(f->path.c_str());
    if (f->dir && next_match(f, out)) return f;
    set_last_error(f->dir ? 2 : win_error());
    if (f->dir) closedir(f->dir);
    delete f;
    return reinterpret_cast<HANDLE>(-1);
}
HANDLE FindFirstFileW(const char16_t* name, uint8_t* out) { return FindFirstFileExW(name, 0, out, 0, nullptr, 0); }
int FindNextFileW(HANDLE h, uint8_t* out) {
    return next_match(static_cast<Find*>(h), out) ? 1 : (set_last_error(18 /* ERROR_NO_MORE_FILES */), 0);
}
int FindClose(HANDLE h) {
    closedir(static_cast<Find*>(h)->dir);
    delete static_cast<Find*>(h);
    return 1;
}

}  // namespace

#define K(name) EXPORT("kernel32.dll", name)
Exports kernel32_file_exports() {
    return {
        K(GetFileAttributesW), K(GetFileAttributesExW), K(CreateDirectoryW), K(CreateDirectoryExW), K(CopyFileW),
        K(MoveFileExW), K(SetFilePointerEx), K(GetFileSizeEx), K(GetFileSize), K(GetFileTime), K(SetEndOfFile), K(FlushFileBuffers), K(GetFileInformationByHandle),
        K(GetFileInformationByHandleEx), K(GetCurrentDirectoryW), K(SetCurrentDirectoryW), K(GetFullPathNameW),
        K(GetFinalPathNameByHandleW), K(SetFileInformationByHandle), K(FindFirstFileExW), K(FindFirstFileW),
        K(FindNextFileW), K(FindClose),
    };
}

}  // namespace host
