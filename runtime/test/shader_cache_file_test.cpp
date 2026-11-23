#include "../src/shader_cache_file.h"
#include "test_temp_dir.h"

#include <cstdio>
#include <iterator>
#include <string>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

// Use a real staging file and inject the stream state of a failed final flush at close.
struct FailsOnClose : std::ofstream {
    using std::ofstream::ofstream;

    void close() {
        std::ofstream::close();
        setstate(std::ios::failbit);
    }
};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

int main() {
    using rt::d3d12::publish_shader_cache_file;
    const testfiles::ScratchDir scratch("leonida_shader_cache_publication");
    const auto cache = scratch.path() / "cache.bin";
    const auto staged = scratch.path() / "cache.bin.tmp";
    const auto write_new_cache = [](std::ofstream& out) { out << "new cache"; };

    EXPECT(publish_shader_cache_file(staged, cache, [](std::ofstream& out) { out << "previous cache"; }));
    EXPECT(read_file(cache) == "previous cache" && !std::filesystem::exists(staged));

    EXPECT(!publish_shader_cache_file<FailsOnClose>(staged, cache, write_new_cache));
    EXPECT(read_file(cache) == "previous cache" && !std::filesystem::exists(staged));

    EXPECT(!publish_shader_cache_file(staged, cache, [](std::ofstream& out) {
        out << "partial cache";
        out.setstate(std::ios::badbit);
    }));
    EXPECT(read_file(cache) == "previous cache" && !std::filesystem::exists(staged));

    // A reader can block the replacement by opening the previous cache without FILE_SHARE_DELETE.
    const HANDLE reader = CreateFileW(cache.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL, nullptr);
    EXPECT(reader != INVALID_HANDLE_VALUE);
    const bool replaced_while_locked = publish_shader_cache_file(staged, cache, write_new_cache);
    EXPECT(CloseHandle(reader));
    EXPECT(!replaced_while_locked);
    EXPECT(read_file(cache) == "previous cache" && !std::filesystem::exists(staged));

    EXPECT(publish_shader_cache_file(staged, cache, write_new_cache));
    EXPECT(read_file(cache) == "new cache" && !std::filesystem::exists(staged));
    std::puts("shader cache publication: write and close failures, locked destination and retry passed");
    return 0;
}
