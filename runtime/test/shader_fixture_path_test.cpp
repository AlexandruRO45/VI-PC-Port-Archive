#include "shader_fixture_path.h"
#include <cstdio>
#include <chrono>
#include <fstream>

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("leonida_fixture_path_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root / "generated");
    const auto previous = fs::current_path();
    int result = 0;
    try {
        fs::current_path(root);
        std::ofstream("generated/test.spv") << "packaged";
        const auto original = root / "build" / "test.spv";
        if (vktest::shader_fixture_path(original) != fs::path("generated/test.spv")) result = 1;
        fs::create_directory("build");
        std::ofstream(original) << "local";
        if (vktest::shader_fixture_path(original) != original) result = 1;
        const auto missing = root / "build" / "missing.spv";
        if (vktest::shader_fixture_path(missing) != missing) result = 1;
        fs::remove(original);
        fs::remove("generated/test.spv");
        fs::create_directory("generated/test.spv");
        if (vktest::shader_fixture_path(original) != original) result = 1;

        // Directories: the packaged copy under generated/ is used only when the build tree's folder is gone.
        fs::create_directories("generated/m2_shaders/fssm");
        const auto dir = root / "build" / "generated" / "m2_shaders";
        if (vktest::shader_fixture_dir(dir) != fs::path("generated/m2_shaders")) result = 1;
        fs::create_directories(dir / "fssm");
        if (vktest::shader_fixture_dir(dir) != dir) result = 1;
        const auto missing_dir = root / "build" / "generated" / "missing";
        if (vktest::shader_fixture_dir(missing_dir) != missing_dir) result = 1;
        fs::remove_all(dir);
        fs::remove_all("generated/m2_shaders");
        std::ofstream("generated/m2_shaders") << "not a directory";
        if (vktest::shader_fixture_dir(dir) != dir) result = 1;  // a file is not a directory
    } catch (...) { result = 1; }
    fs::current_path(previous);
    fs::remove_all(root);
    std::puts(result ? "shader fixture lookup failed" : "shader fixture lookup passed");
    return result;
}
