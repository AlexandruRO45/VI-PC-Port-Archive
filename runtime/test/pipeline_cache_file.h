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

inline std::error_code write_pipeline_cache_file(const std::filesystem::path& destination,
                                                const char* data, size_t size) {
    namespace fs = std::filesystem;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path staged;
    std::FILE* file = nullptr;
    int open_error = 0;
    for (int attempt = 0; attempt < 8; ++attempt) {
        staged = destination;
        staged += ".tmp." + std::to_string(stamp) + "." + std::to_string(attempt);
#ifdef _WIN32
        // The checker's static Windows CRT can terminate on the fopen "x" mode.
        // Open exclusively through the CRT descriptor API, then attach a binary stream.
        int fd = -1;
        open_error = _wsopen_s(&fd, staged.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
                               _SH_DENYNO, _S_IREAD | _S_IWRITE);
        if (!open_error) {
            file = _wfdopen(fd, L"wb");
            if (!file) {
                const std::error_code error{errno ? errno : EIO, std::generic_category()};
                _close(fd);
                std::error_code ignored;
                fs::remove(staged, ignored);
                return error;
            }
        }
#else
        file = std::fopen(staged.c_str(), "wbx");
        open_error = file ? 0 : errno;
#endif
        if (file || open_error != EEXIST) break;
    }
    if (!file) return {open_error, std::generic_category()};

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
