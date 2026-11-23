#pragma once

#include <filesystem>

inline std::error_code prepare_xdxr_build_output(const std::filesystem::path& dir, bool timing_only) {
    if (timing_only) return {};

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return ec;

    // A failed run must not leave the previous run's cases for the checker to pass.
    using std::filesystem::directory_iterator;
    for (directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const auto ext = it->path().extension();
        if (ext != ".bin" && ext != ".json") continue;
        std::filesystem::remove(it->path(), ec);
        if (ec) return ec;
    }
    return ec;
}
