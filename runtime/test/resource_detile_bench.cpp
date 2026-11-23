// Throughput measurement for the tiling and detiling loops (src/resource_detile.h), without a GPU.
//
//   resource_detile_bench              times tile and detile over a few surface shapes, best of N
//   resource_detile_bench --self-test  checks the round trips and exits 0 (what ctest runs)
//
// The loops walk every texel of a console-tiled 64 KiB surface, so they are the cost of a texture upload that
// starts in that layout. Run it before and after a change to the inner loops; the numbers are per element size,
// which is where the difference is (a 1-byte element pays for the per-element overhead, a 16-byte one is already
// a whole move).

#include "../src/resource_detile.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace {

using namespace leonida;

struct Case {
    std::uint32_t width, height, element_bytes, block_dim;
    const char* note;
};

// The shapes a game uploads: colour targets, a 4K albedo, an 8-bit mask, a half-float pair, an RGBA32F target,
// a 128 MB volume-sized one (past the caches), and a block-compressed texture.
constexpr Case kCases[] = {
    {1920, 1080, 4, 1, "1080p RGBA8"},
    {3840, 2160, 4, 1, "4K RGBA8"},
    {1024, 1024, 1, 1, "R8"},
    {1024, 1024, 2, 1, "RG8 / R16F"},
    {1024, 1024, 16, 1, "RGBA32F"},
    {4096, 4096, 8, 1, "RGBA16F, 128 MB"},
    {2048, 2048, 4, 4, "BC1 2048x2048"},
};

template <class Fn>
double best_of(Fn&& fn, int reps) {
    double best = 1e9;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

bool roundtrip(const Case& c) {
    const std::size_t linear_size = linear_surface_size(c.width, c.height, c.element_bytes, c.block_dim);
    const std::size_t tiled_size = tiled_surface_size_64k(c.width, c.height, c.element_bytes, c.block_dim);
    CHECK(linear_size && tiled_size);
    std::vector<std::byte> original(linear_size), tiled(tiled_size, std::byte{0xCC}), recovered(linear_size);
    for (std::size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<std::byte>((i * 37u + 13u) & 0xFFu);
    }
    CHECK(tile_surface_64k(original, tiled, c.width, c.height, c.element_bytes, c.block_dim));
    CHECK(detile_surface_64k(tiled, recovered, c.width, c.height, c.element_bytes, c.block_dim));
    return std::memcmp(original.data(), recovered.data(), linear_size) == 0;
}

int self_test() {
    for (const Case& c : kCases) {
        if (!roundtrip(c)) {
            std::fprintf(stderr, "round trip failed for %ux%u eb=%u blk=%u (%s)\n",
                         c.width, c.height, c.element_bytes, c.block_dim, c.note);
            return 1;
        }
    }
    std::printf("resource detile bench: round trips ok\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--self-test") return self_test();

    std::printf("%-24s %8s %12s %12s %12s\n", "surface", "MB", "tile ms", "detile ms", "detile GB/s");
    for (const Case& c : kCases) {
        const std::size_t linear_size = linear_surface_size(c.width, c.height, c.element_bytes, c.block_dim);
        const std::size_t tiled_size = tiled_surface_size_64k(c.width, c.height, c.element_bytes, c.block_dim);
        std::vector<std::byte> linear(linear_size), tiled(tiled_size, std::byte{0xCC}), recovered(linear_size);
        for (std::size_t i = 0; i < linear.size(); ++i) {
            linear[i] = static_cast<std::byte>((i * 37u + 13u) & 0xFFu);
        }
        const double t_tile = best_of([&] {
            tile_surface_64k(linear, tiled, c.width, c.height, c.element_bytes, c.block_dim);
        }, 6);
        const double t_detile = best_of([&] {
            detile_surface_64k(tiled, recovered, c.width, c.height, c.element_bytes, c.block_dim);
        }, 6);
        const double mb = double(linear_size) / (1024.0 * 1024.0);
        const double gbps = t_detile > 0.0 ? mb / 1024.0 / (t_detile / 1000.0) : 0.0;
        std::printf("%-24s %8.1f %12.3f %12.3f %12.2f\n", c.note, mb, t_tile, t_detile, gbps);
    }
    return 0;
}
