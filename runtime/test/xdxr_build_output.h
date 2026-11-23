#pragma once

#include <filesystem>

inline std::error_code prepare_xdxr_build_output(const std::filesystem::path& dir, bool timing_only) {
    if (timing_only) return {};

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return ec;

    // A failed run must not leave the previous run's cases for the checker to pass.
    const std::filesystem::directory_iterator end;
    for (std::filesystem::directory_iterator entry(dir, ec); !ec && entry != end; entry.increment(ec)) {
        const auto ext = entry->path().extension();
        if (ext == ".bin" || ext == ".json") {
            std::filesystem::remove(entry->path(), ec);
            if (ec) return ec;
        }
    }
    return ec;
}
