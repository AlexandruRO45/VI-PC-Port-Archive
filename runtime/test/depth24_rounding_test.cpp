// A D24 code must have the same stored float in every host rounding mode.
// The independent oracle is the nearest-mode double quotient, plus literal edges.
// No Vulkan loader, GPU, shader fixture or game data is needed.
#include "../src/depth24.h"
#include <array>
#include <bit>
#include <cfenv>
#include <cstdio>
#include <vector>

namespace {
struct FpEnvironment {
    std::fenv_t saved{};
    bool valid = std::fegetenv(&saved) == 0;
    ~FpEnvironment() { if (valid) std::fesetenv(&saved); }
};
struct Sample { std::uint32_t code, bits; };
}

int main() {
    FpEnvironment environment;
    if (!environment.valid || std::fesetround(FE_TONEAREST)) {
        std::fprintf(stderr, "cannot save/set the floating-point environment\n");
        return 1;
    }
    const Sample literals[] = {
        {0x000000u, 0x00000000u}, {0x000001u, 0x33800001u},
        {0x000002u, 0x34000001u}, {0x7fffffu, 0x3effffffu},
        {0x800000u, 0x3f000001u}, {0x9fffffu, 0x3f200000u},
        {0xa00000u, 0x3f200001u}, {0xfffffeu, 0x3f7fffffu},
        {0xffffffu, 0x3f800000u},
    };
    std::vector<Sample> samples(std::begin(literals), std::end(literals));
    // Both sides of each exponent transition, including zero and the maximum.
    for (unsigned exponent = 0; exponent < 24; ++exponent) {
        const std::uint32_t edge = 1u << exponent;
        for (const std::uint32_t code : {edge - 1, edge, edge + 1}) {
            const float nearest = static_cast<float>(double(code) / double(leonida::depth24_max));
            samples.push_back({code, std::bit_cast<std::uint32_t>(nearest)});
        }
    }
    const std::size_t unmasked_size = samples.size();
    for (std::size_t i = 0; i < unmasked_size; ++i)
        samples.push_back({samples[i].code | 0xab000000u, samples[i].bits});

    const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    struct Clear { std::uint32_t input; std::array<std::uint32_t, 4> codes; };
    // Literal WARP clear/readback measurements: the quantizer still follows the
    // submission mode. Only converting the chosen code to float is mode independent.
    const Clear clears[] = {
        {0x3f200000u, {0x9fffffu, 0x9fffffu, 0xa00000u, 0x9fffffu}},
        {0x3e800000u, {0x400000u, 0x3fffffu, 0x400000u, 0x3fffffu}},
    };
    unsigned checks = 0, failures = 0;
    for (unsigned m = 0; m < 4; ++m) {
        if (std::fesetround(modes[m])) return 1;
        std::vector<float> stored;
        for (const auto& sample : samples) {
            const float value = leonida::depth24_to_float(sample.code);
            stored.push_back(value);
            const auto actual = std::bit_cast<std::uint32_t>(value);
            if (actual != sample.bits) {
                if (failures < 8)
                    std::fprintf(stderr, "mode %d code %08x: bits %08x expected %08x\n",
                                 modes[m], sample.code, actual, sample.bits);
                ++failures;
            }
            ++checks;
        }
        std::array<float, 2> clear_values{};
        for (unsigned c = 0; c < 2; ++c) {
            volatile float input = std::bit_cast<float>(clears[c].input);
            const auto code = leonida::depth24_quantize(input);
            clear_values[c] = leonida::depth24_clear_value(input);
            if (code != clears[c].codes[m]) {
                std::fprintf(stderr, "mode %d input %08x: code %06x expected %06x\n",
                             modes[m], clears[c].input, code, clears[c].codes[m]);
                ++failures;
            }
            ++checks;
        }
        if (std::fegetround() != modes[m]) {
            std::fprintf(stderr, "conversion changed the rounding mode\n");
            return 1;
        }
        // Readback uses the shader's nearest-mode float product and roundEven.
        // Restore that mode before checking the values produced in the other modes.
        if (std::fesetround(FE_TONEAREST)) return 1;
        for (std::size_t i = 0; i < stored.size(); ++i) {
            const auto code = leonida::depth24_quantize(stored[i]);
            if (code != (samples[i].code & leonida::depth24_max)) {
                if (failures < 8)
                    std::fprintf(stderr, "mode %d roundtrip %08x became %06x\n",
                                 modes[m], samples[i].code, code);
                ++failures;
            }
            ++checks;
        }
        for (unsigned c = 0; c < 2; ++c) {
            const auto code = leonida::depth24_quantize(clear_values[c]);
            if (code != clears[c].codes[m]) {
                std::fprintf(stderr, "mode %d clear %08x read back %06x expected %06x\n",
                             modes[m], clears[c].input, code, clears[c].codes[m]);
                ++failures;
            }
            ++checks;
        }
    }
    std::printf("depth24_rounding_test: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
