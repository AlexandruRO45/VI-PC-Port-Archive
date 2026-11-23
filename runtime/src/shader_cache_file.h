#pragma once

#include <filesystem>
#include <fstream>
#include <system_error>

#include <windows.h>

namespace rt::d3d12 {

template <typename OutputFile = std::ofstream, typename Serialize>
bool publish_shader_cache_file(const std::filesystem::path& staged, const std::filesystem::path& destination,
                               Serialize serialize) {
    OutputFile out(staged, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    serialize(out);
    // Closing the stream reports failures from its final buffered write.
    out.close();
    if (out && MoveFileExW(staged.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING)) return true;
    std::error_code ignored;
    std::filesystem::remove(staged, ignored);
    return false;
}

}  // namespace rt::d3d12
