// Synthetic GPU test shaders are packaged under generated/ in runtime-windows artifacts.
// Run downloaded tests from the artifact directory, as CTest does for a local build.
#pragma once

#include <filesystem>

namespace vktest {
inline std::filesystem::path shader_fixture_path(const std::filesystem::path& compiled_path) {
    if (std::filesystem::is_regular_file(compiled_path)) return compiled_path;
    const auto packaged = std::filesystem::path("generated") / compiled_path.filename();
    if (std::filesystem::is_regular_file(packaged)) return packaged;
    // Preserve the original path in the caller's missing-file diagnostic.
    return compiled_path;
}

// A directory build.py wrote, such as generated/m2_shaders. The artifact keeps the same name under generated/.
inline std::filesystem::path shader_fixture_dir(const std::filesystem::path& compiled_dir) {
    if (std::filesystem::is_directory(compiled_dir)) return compiled_dir;
    auto name = compiled_dir.filename();
    if (name.empty()) name = compiled_dir.parent_path().filename();  // a path with a trailing separator
    const auto packaged = std::filesystem::path("generated") / name;
    if (std::filesystem::is_directory(packaged)) return packaged;
    // Preserve the original path in the caller's missing-directory diagnostic.
    return compiled_dir;
}
}  // namespace vktest
