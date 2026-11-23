#pragma once
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <optional>

namespace leonida::conservative_raster {
// Private, per-emitted-primitive storage. Guest PrimitiveId is not necessarily unique after a GS.
// The synthetic shader contract uses 32-bit bit indices and uint32 atomic OR operations.
struct InnerCoverageMask {
    std::uint32_t width, height, primitives, instances;
    std::uint64_t bits, bytes;

    static constexpr std::optional<InnerCoverageMask> make(
        std::uint32_t width, std::uint32_t height, std::uint32_t primitives,
        std::uint32_t instances, std::uint64_t budget) {
        if (!width || !height || !primitives || !instances) return std::nullopt;
        std::uint64_t bits = width;
        for (auto factor : {height, primitives, instances}) {
            if (bits > std::numeric_limits<std::uint32_t>::max() / factor) return std::nullopt;
            bits *= factor;
        }
        const auto bytes = ((bits + 31) / 32) * 4;
        if (bytes > budget) return std::nullopt;
        return InnerCoverageMask{width, height, primitives, instances, bits, bytes};
    }

    struct Address { std::uint64_t byte_offset; std::uint32_t bit; };
    constexpr std::optional<Address> address(std::uint32_t x, std::uint32_t y,
                                           std::uint32_t primitive, std::uint32_t instance) const {
        if (x >= width || y >= height || primitive >= primitives || instance >= instances) return std::nullopt;
        const auto index = ((std::uint64_t(instance) * primitives + primitive) * height + y) * width + x;
        return Address{(index / 32) * 4, std::uint32_t(1u << (index % 32))};
    }
};
} // namespace leonida::conservative_raster
