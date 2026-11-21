// End-to-end PE loader test: rt::Loader maps leonida_test_harness.exe, binds its d3d12.dll import to the shim in
// pe_hello_shims.cpp, and calling the image's entry point renders the triangle through Vulkan into hello_triangle.bmp.
//
// The harness EXE is built next to this test by CMake (custom entry, no CRT, one import). No game data.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "pe_loader.h"

namespace {

int failures = 0;

#define EXPECT(condition)                                                \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::vector<std::uint8_t> data;
    std::FILE* f = std::fopen(path.string().c_str(), "rb");
    if (!f) return data;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    data.resize(std::size_t(size));
    if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
    std::fclose(f);
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path harness =
        argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("leonida_test_harness.exe");
    if (!std::filesystem::exists(harness)) {
        std::fprintf(stderr, "pe_hello_test: harness not found: %s\n", harness.string().c_str());
        return 1;
    }
    std::error_code ec;
    std::filesystem::remove("hello_triangle.bmp", ec);  // stale output must not pass the test

    try {
        rt::Loader loader(harness.parent_path());
        rt::Module& module = loader.load(harness);
        EXPECT(module.base != nullptr);
        EXPECT(module.entry_rva != 0);
        loader.initialise();  // an EXE's DllMain is not called; its entry is the caller's to run

        auto* entry = reinterpret_cast<void (*)()>(module.base + module.entry_rva);
        entry();

        const std::vector<std::uint8_t> bmp = read_file("hello_triangle.bmp");
        EXPECT(bmp.size() > 54);
        if (bmp.size() > 54) {
            EXPECT(bmp[0] == 'B' && bmp[1] == 'M');
            const auto* pixels = bmp.data() + 54;
            std::size_t lit = 0;
            for (std::size_t i = 0; i + 2 < bmp.size() - 54; i += 3) {
                if (pixels[i] || pixels[i + 1] || pixels[i + 2]) ++lit;
            }
            std::printf("pe_hello_test: %zu lit pixels written by the harness\n", lit);
            EXPECT(lit > 100);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pe_hello_test: exception: %s\n", e.what());
        ++failures;
    }

    if (failures) {
        std::fprintf(stderr, "pe_hello_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("pe_hello_test: all tests passed\n");
    return 0;
}
