#pragma once

// MoltenVK's SPIRV-Cross turns every NoContraction OpFAdd, OpFSub and OpFMul into a call to a [[clang::optnone]]
// helper (spvFAdd, spvFSub, spvFMul), which made a translated shader 7 to 15 times slower on an Apple M3 Max
// (docs/design.md, "NoContraction on MoltenVK"). When MoltenVK compiles a shader with Metal's safe math, Metal
// rounds separate statements separately, so the decoration is only needed where Metal still fuses: inside one
// expression, `a * b + c` becomes llvm.fmuladd even in safe math, and SPIRV-Cross writes a multiply (OpFMul, a
// vector or matrix product, OpDot or OpOuterProduct) whose result only an add or subtract uses into that add's
// expression. Such a multiply keeps NoContraction, and so does the add or subtract unless the multiply is an OpFMul
// with a decoration of its own: SPIRV-Cross (MoltenVK 1.4.2) writes only OpFMul as spvFMul, and the other
// multiplies as a plain `*` even when decorated, so the add's helper is then all that keeps them apart.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <span>
#include <vector>

namespace leonida {

// When MoltenVK compiles a shader with MTLMathModeSafe (MVKDevice::getMTLCompileOptions).
enum class MetalSafeMath {
    never,        // not MoltenVK, its fast math is always on, or its settings are not known
    if_declared,  // fast math on demand: safe when the entry point declares SignedZeroInfNanPreserve
    always,       // MVK_CONFIG_FAST_MATH_ENABLED=0
};

// Where the MoltenVK settings of a VkInstance come from. MoltenVK reads MVK_CONFIG_FAST_MATH_ENABLED from the
// environment, but an instance created with VK_EXT_layer_settings overrides it (MVKInstance::initMVKConfig).
enum class MoltenVKSettings {
    unknown,      // an instance somebody else created: its fast math may be anything
    environment,  // created here without VK_EXT_layer_settings: MoltenVK's defaults and the environment
};

// `fast_math_env` is MVK_CONFIG_FAST_MATH_ENABLED (null when unset). MoltenVK 1.4.0 made "on demand" (2) the
// default; before that it was "always" (1). Any other value, and an instance with unknown settings, keeps every
// decoration.
inline MetalSafeMath moltenvk_safe_math(VkDriverId driver, std::uint32_t driver_version, MoltenVKSettings settings,
                                        const char* fast_math_env) {
    if (driver != VK_DRIVER_ID_MOLTENVK || settings != MoltenVKSettings::environment) return MetalSafeMath::never;
    constexpr std::uint32_t kOnDemandDefault = 10400;  // MVK_MAKE_VERSION(1, 4, 0), MoltenVK's driverVersion
    if (!fast_math_env || !*fast_math_env)
        return driver_version >= kOnDemandDefault ? MetalSafeMath::if_declared : MetalSafeMath::never;
    if (!std::strcmp(fast_math_env, "0")) return MetalSafeMath::always;
    if (!std::strcmp(fast_math_env, "2")) return MetalSafeMath::if_declared;
    return MetalSafeMath::never;
}

inline MetalSafeMath moltenvk_safe_math(VkDriverId driver, std::uint32_t driver_version, MoltenVKSettings settings) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once per device
#endif
    const char* env = std::getenv("MVK_CONFIG_FAST_MATH_ENABLED");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    return moltenvk_safe_math(driver, driver_version, settings, env);
}

// The same for a physical device, read with vkGetPhysicalDeviceProperties2 (null: never).
inline MetalSafeMath moltenvk_safe_math(PFN_vkGetPhysicalDeviceProperties2 get_properties2, VkPhysicalDevice physical,
                                        MoltenVKSettings settings) {
    if (!get_properties2 || !physical) return MetalSafeMath::never;
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &driver;
    get_properties2(physical, &properties);
    return moltenvk_safe_math(driver.driverID, properties.properties.driverVersion, settings);
}

// The word of an instruction that holds its result id (1, or 2 after a result type), or 0 for none: runs of opcodes
// from SPIR-V 1.6 revision 7's spirv.core.grammar.json, extensions included. An opcode the grammar does not define
// counts as its neighbours when both sides of it agree, so at worst it makes a module look malformed.
inline std::uint32_t spirv_result_word(std::uint32_t op) {
    struct Run {
        std::uint16_t first, last;
        std::uint8_t word;
    };
    static constexpr Run kRuns[] = {
        {1, 1, 2}, {7, 7, 1}, {11, 11, 1}, {12, 12, 2}, {19, 38, 1}, {41, 55, 2}, {57, 61, 2}, {65, 70, 2},
        {73, 73, 1}, {77, 98, 2}, {100, 215, 2}, {227, 227, 2}, {229, 245, 2}, {248, 248, 1}, {259, 259, 2},
        {261, 279, 2}, {282, 286, 2}, {291, 296, 2}, {299, 300, 2}, {303, 316, 2}, {318, 318, 2}, {320, 321, 2},
        {322, 322, 1}, {323, 326, 2}, {327, 327, 1}, {328, 328, 2}, {333, 4162, 2}, {4163, 4163, 1}, {4164, 4164, 2},
        {4166, 4181, 2}, {4183, 4184, 2}, {4190, 4190, 1}, {4195, 4195, 2}, {4417, 4417, 1}, {4418, 4425, 2},
        {4427, 4434, 2}, {4447, 4447, 2}, {4450, 4455, 2}, {4456, 4456, 1}, {4457, 4457, 2}, {4459, 4463, 2},
        {4472, 4472, 1}, {4477, 5074, 2}, {5076, 5076, 1}, {5078, 5101, 2}, {5103, 5104, 1}, {5110, 5111, 2},
        {5115, 5115, 1}, {5119, 5119, 2}, {5126, 5126, 2}, {5129, 5159, 2}, {5252, 5255, 2}, {5257, 5258, 2},
        {5265, 5265, 2}, {5267, 5278, 2}, {5281, 5281, 1}, {5283, 5283, 2}, {5288, 5288, 1}, {5289, 5289, 2},
        {5292, 5293, 2}, {5296, 5296, 2}, {5300, 5302, 2}, {5307, 5308, 2}, {5313, 5313, 1}, {5320, 5320, 2},
        {5322, 5334, 2}, {5340, 5340, 2}, {5341, 5341, 1}, {5345, 5352, 2}, {5358, 5358, 1}, {5359, 5359, 2},
        {5361, 5363, 2}, {5366, 5367, 2}, {5369, 5369, 2}, {5370, 5371, 1}, {5372, 5379, 2}, {5381, 5396, 2},
        {5398, 5575, 2}, {5577, 5577, 2}, {5580, 5580, 2}, {5585, 5601, 2}, {5609, 5609, 1}, {5610, 5615, 2},
        {5631, 5631, 2}, {5699, 5699, 2}, {5700, 5712, 1}, {5713, 5819, 2}, {5840, 5882, 2}, {5911, 5913, 1},
        {5923, 5938, 2}, {5949, 6035, 2}, {6086, 6086, 1}, {6116, 6117, 2}, {6145, 6163, 2}, {6165, 6165, 2},
        {6199, 6199, 1}, {6237, 6244, 2}, {6251, 6258, 2}, {6401, 6428, 2}, {6529, 6918, 2},
    };
    const Run* run = std::upper_bound(std::begin(kRuns), std::end(kRuns), op,
                                      [](std::uint32_t o, const Run& r) { return o < r.first; });
    return run != std::begin(kRuns) && op <= (run - 1)->last ? (run - 1)->word : 0;
}

// The module MoltenVK should get: without the NoContraction decorations safe math makes redundant, or unchanged
// when MoltenVK would not compile it with safe math or the words are not a module this can read.
inline std::vector<std::uint32_t> drop_no_contraction(std::span<const std::uint32_t> words, MetalSafeMath math) {
    const auto unchanged = [&] { return std::vector<std::uint32_t>(words.begin(), words.end()); };
    constexpr std::uint32_t kMagic = 0x07230203, kHeaderWords = 5;
    if (math == MetalSafeMath::never || words.size() <= kHeaderWords || words[0] != kMagic) return unchanged();
    // Version 0x00MMmm00, SPIR-V 1.0 to 1.6; the schema word is reserved and 0.
    const std::uint32_t version = words[1], minor = version >> 8 & 0xff;
    if ((version & 0xff0000ff) != 0 || (version >> 16) != 1 || minor > 6 || words[4] != 0) return unchanged();
    // Every result id takes an instruction of at least two words, so the bound of a module numbered densely, as
    // the translator's are, is well below its word count. A larger bound is malformed or numbered sparsely; the
    // tables are not sized by it (0xffffffff would take 16 GiB) and the module stays as it is.
    const std::uint32_t bound = words[3];
    if (bound < 2 || bound > words.size()) return unchanged();

    constexpr std::uint32_t kEntryPoint = 15, kExecutionMode = 16, kDecorate = 71, kDecorationGroup = 73,
                            kGroupDecorate = 74, kCopyObject = 83, kCompositeExtract = 81, kBitcast = 124,
                            kFNegate = 127, kFAdd = 129, kFSub = 131;
    constexpr std::uint32_t kNoContraction = 42, kSignedZeroInfNanPreserve = 4461;
    // What Metal can fuse into an add: OpFMul, OpVectorTimesScalar (142), OpMatrixTimesScalar,
    // OpVectorTimesMatrix, OpMatrixTimesVector, OpMatrixTimesMatrix, OpOuterProduct and OpDot (148).
    constexpr std::uint32_t kFMul = 133;
    const auto is_multiply = [](std::uint32_t op) { return op == kFMul || (op >= 142 && op <= 148); };
    // The multiply each value is, through negates, copies, bitcasts and composite extracts, or 0. Each of those
    // reads a value defined before it, so a value's multiply is known when it is defined, and no chain can loop.
    std::vector<std::uint32_t> multiply(bound, 0);
    std::vector<bool> defined(bound, false), decorated(bound, false), group(bound, false), preserves(bound, false),
        fmul(bound, false);
    struct Sum {
        std::uint32_t result, products[2];
    };
    std::vector<Sum> sums;
    std::vector<std::uint32_t> entries;
    std::vector<std::size_t> group_decorations;  // where each OpGroupDecorate starts
    // What entry points, execution modes and decorations name: ids defined later in the module, so each is checked
    // once the whole module has been read. A decoration of an id nothing defines must not be dropped, since that
    // would turn an invalid module into a valid one.
    std::vector<std::uint32_t> targets;
    bool any_decorated = false, malformed = false;
    for (std::size_t i = kHeaderWords; i < words.size() && !malformed;) {
        const std::uint32_t op = words[i] & 0xffff, count = words[i] >> 16;
        if (!count || count > words.size() - i) return unchanged();
        // Operand `k`, which must be an id; anything else makes the module malformed.
        const auto id = [&](std::uint32_t k) {
            if (k < count && words[i + k] && words[i + k] < bound) return words[i + k];
            malformed = true;
            return std::uint32_t(0);
        };
        // Operand `k`, an id that some instruction of the module must define.
        const auto target = [&](std::uint32_t k) {
            const std::uint32_t t = id(k);
            targets.push_back(t);
            return t;
        };
        // Operand `k`, an id that an earlier instruction must define.
        const auto value = [&](std::uint32_t k) {
            const std::uint32_t v = id(k);
            if (!defined[v]) malformed = true;
            return v;
        };
        // Every result id is below the bound, not 0, and defined once.
        std::uint32_t result = 0;
        if (const std::uint32_t at = spirv_result_word(op)) {
            result = id(at);
            if (defined[result]) malformed = true;
        }
        switch (op) {
        case kEntryPoint:
            if (count < 4) malformed = true;
            entries.push_back(target(2));
            break;
        case kExecutionMode:
            if (count < 3) malformed = true;
            else if (words[i + 2] == kSignedZeroInfNanPreserve) {
                // The mode takes one operand, the bit width.
                const std::uint32_t width = count == 4 ? words[i + 3] : 0;
                if (width != 16 && width != 32 && width != 64) malformed = true;
                preserves[target(1)] = true;
            } else target(1);
            break;
        case kDecorate:
            if (count < 3) malformed = true;
            else if (words[i + 2] == kNoContraction) {
                if (count != 3) malformed = true;  // NoContraction takes no operands
                decorated[target(1)] = true;
                any_decorated = true;
            } else target(1);
            break;
        case kDecorationGroup:
            if (count != 2) malformed = true;
            group[result] = true;
            break;
        case kGroupDecorate:
            for (std::uint32_t k = 1; k < count; ++k) target(k);
            if (count < 2) malformed = true;
            group_decorations.push_back(i);
            break;
        case kFNegate:
        case kCopyObject:
        case kBitcast:
            if (count != 4) malformed = true;
            id(1);
            multiply[result] = multiply[value(3)];
            break;
        case kCompositeExtract:  // the composite and at least one index
            if (count < 5) malformed = true;
            id(1);
            multiply[result] = multiply[value(3)];
            break;
        case kFAdd:
        case kFSub:
            if (count != 5) malformed = true;
            id(1);
            sums.push_back({result, {multiply[value(3)], multiply[value(4)]}});
            break;
        default:
            if (is_multiply(op)) {
                if (count != 5) malformed = true;
                id(1);
                value(3);
                value(4);
                multiply[result] = result;
                fmul[result] = op == kFMul;
            }
            break;
        }
        if (result) defined[result] = true;
        i += count;
    }
    if (malformed || !any_decorated || entries.empty()) return unchanged();
    for (const std::uint32_t t : targets)
        if (!defined[t]) return unchanged();
    if (math == MetalSafeMath::if_declared)
        for (const std::uint32_t entry : entries)
            if (!preserves[entry]) return unchanged();
    // A group's decorations are left as they are; its members count as decorated for the adds below.
    for (const std::size_t at : group_decorations)
        if (decorated[words[at + 1]])
            for (std::uint32_t k = 2; k < (words[at] >> 16); ++k) decorated[words[at + k]] = true;

    std::vector<bool> keep(bound, false);
    for (const Sum& sum : sums)
        for (const std::uint32_t product : sum.products)
            if (product) {
                keep[product] = true;  // spvFMul keeps the product out of the add's expression
                // Only a decorated OpFMul becomes spvFMul; otherwise spvFAdd / spvFSub has to.
                if (!decorated[product] || !fmul[product]) keep[sum.result] = true;
            }

    std::vector<std::uint32_t> out(words.begin(), words.begin() + kHeaderWords);
    out.reserve(words.size());
    for (std::size_t i = kHeaderWords; i < words.size();) {
        const std::uint32_t count = words[i] >> 16;
        const bool drop = (words[i] & 0xffff) == kDecorate && words[i + 2] == kNoContraction &&
                          !group[words[i + 1]] && !keep[words[i + 1]];
        if (!drop) out.insert(out.end(), words.begin() + i, words.begin() + i + count);
        i += count;
    }
    return out;
}

}  // namespace leonida
