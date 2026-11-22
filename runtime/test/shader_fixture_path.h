// Synthetic GPU test shaders are packaged under generated/ in runtime-windows artifacts.
// Run downloaded tests from the artifact directory, as CTest does for a local build.
#pragma once

#include <filesystem>
#include <system_error>

namespace vktest {
inline std::filesystem::path shader_fixture_path(const std::filesystem::path& compiled_path) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(compiled_path, ec)) return compiled_path;
    const auto packaged = std::filesystem::path("generated") / compiled_path.filename();
    ec.clear();
    if (std::filesystem::is_regular_file(packaged, ec)) return packaged;
    // Preserve the original path in the caller's missing-file diagnostic.
    return compiled_path;
}
}  // namespace vktest
