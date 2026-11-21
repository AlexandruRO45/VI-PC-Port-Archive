// Engine-simulation PE test: threads, clocks, file I/O, a missing import and WSI frames through the shim DLL.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

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

std::string read_text(const char* path) {
    std::string text;
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return text;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    text.resize(std::size_t(size));
    if (std::fread(text.data(), 1, text.size(), f) != text.size()) text.clear();
    std::fclose(f);
    return text;
}
}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path harness =
        argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("test_engine_simulation.exe");
    EXPECT(std::filesystem::exists(harness));
    if (!std::filesystem::exists(harness)) return 1;

    {
        std::FILE* input = std::fopen("engine_input.txt", "wb");
        if (input) {
            std::fwrite("LEONIDA", 1, 7, input);
            std::fclose(input);
        }
    }
    std::error_code ec;
    std::filesystem::remove("missing_imports.log", ec);
    std::filesystem::remove("engine_sim.log", ec);
    std::filesystem::remove("leonida_window.log", ec);

    HMODULE shim_dll = LoadLibraryA("leonida_shims.dll");
    EXPECT(shim_dll != nullptr);

    try {
        rt::Loader loader(harness.parent_path());
        rt::Module& module = loader.load(harness);
        loader.initialise();
        auto* entry = reinterpret_cast<void (*)()>(module.base + module.entry_rva);
        entry();

        const std::string missing = read_text("missing_imports.log");
        EXPECT(missing.find("XInputGetStateForTests") != std::string::npos);
        const std::string sim = read_text("engine_sim.log");
        EXPECT(sim.find("thread=1000") != std::string::npos);
        EXPECT(sim.find("bytes=7") != std::string::npos);
        EXPECT(sim.find("text=LEONIDA") != std::string::npos);
        EXPECT(sim.find("qpc=") != std::string::npos);
        const std::string window = read_text("leonida_window.log");
        const auto at = window.find("frames=");
        int frames = 0;
        if (at != std::string::npos) std::sscanf(window.c_str() + at, "frames=%d", &frames);
        std::printf("engine_sim_test: frames=%d missing=%s\n", frames, missing.c_str());
        EXPECT(frames > 60);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "engine_sim_test: exception: %s\n", e.what());
        ++failures;
    }

    if (failures) {
        std::fprintf(stderr, "engine_sim_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("engine_sim_test: all tests passed\n");
    return 0;
}
