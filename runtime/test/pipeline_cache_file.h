// Publish a checkpoint only after every byte has reached a closed, same-directory staging file.
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#include <thread>
#include <windows.h>
#endif

namespace pipeline_cache_file_detail {

inline std::FILE* create_exclusive(const std::filesystem::path& path) {
#ifdef _WIN32
    // Some static UCRT builds reject fopen's "x" mode; low-level creation preserves its no-overwrite contract.
    int descriptor = -1;
    const errno_t error = _wsopen_s(&descriptor, path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
                                   _SH_DENYNO, _S_IREAD | _S_IWRITE);
    if (error) {
        errno = error;
        return nullptr;
    }
    std::FILE* file = _wfdopen(descriptor, L"wb");
    if (!file) {
        const int stream_error = errno ? errno : EIO;
        _close(descriptor);
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        errno = stream_error;
    }
    return file;
#else
    return std::fopen(path.c_str(), "wbx");
#endif
}

}  // namespace pipeline_cache_file_detail

inline std::error_code write_pipeline_cache_file(const std::filesystem::path& destination,
                                                const char* data, size_t size) {
    namespace fs = std::filesystem;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path staged;
    std::FILE* file = nullptr;
    for (int attempt = 0; attempt < 8; ++attempt) {
        staged = destination;
        staged += ".tmp." + std::to_string(stamp) + "." + std::to_string(attempt);
        file = pipeline_cache_file_detail::create_exclusive(staged);
        if (file || errno != EEXIST) break;
    }
    if (!file) return {errno, std::generic_category()};

    std::error_code error;
    // Keep individual CRT writes small even when the cache is several GiB.
    while (size > 0) {
        const size_t chunk = (std::min)(size, size_t{16} << 20);
        if (std::fwrite(data, 1, chunk, file) != chunk || std::ferror(file)) {
            error = {errno ? errno : EIO, std::generic_category()};
            break;
        }
        data += chunk;
        size -= chunk;
    }
    if (std::fflush(file) != 0 && !error) error = {errno ? errno : EIO, std::generic_category()};
    if (std::fclose(file) != 0 && !error) error = {errno ? errno : EIO, std::generic_category()};
    if (!error) {
        fs::rename(staged, destination, error);
#ifdef _WIN32
        // A checker reading the previous cache can briefly block replacement if it did not share delete access.
        for (int retry = 0; retry < 8; ++retry) {
            if (error != std::errc::permission_denied && error.value() != ERROR_SHARING_VIOLATION) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            fs::rename(staged, destination, error);
        }
#endif
    }
    if (error) {
        std::error_code ignored;
        fs::remove(staged, ignored);
    }
    return error;
}
