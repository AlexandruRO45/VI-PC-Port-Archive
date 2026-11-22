#pragma once

// MoltenVK's SPIRV-Cross turns every NoContraction OpFAdd, OpFSub and OpFMul into a call to a [[clang::optnone]]
// helper (spvFAdd, spvFSub, spvFMul), which made a translated shader 7 to 15 times slower on an Apple M3 Max
// (docs/design.md, "NoContraction on MoltenVK"). When MoltenVK compiles a shader with Metal's safe math, Metal
// rounds separate statements separately, so the decoration is only needed where Metal still fuses: inside one
// expression, `a * b + c` becomes llvm.fmuladd even in safe math, and SPIRV-Cross writes a multiply whose result
// only an add or subtract uses into that add's expression. Such a multiply keeps NoContraction, and so does the
// add or subtract when the multiply has none of its own, since the add's helper is then all that keeps them apart.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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

// The module MoltenVK should get: without the NoContraction decorations safe math makes redundant, or unchanged
// when MoltenVK would not compile it with safe math or the words are not a module this can read.
inline std::vector<std::uint32_t> drop_no_contraction(std::span<const std::uint32_t> words, MetalSafeMath math) {
    const auto unchanged = [&] { return std::vector<std::uint32_t>(words.begin(), words.end()); };
    constexpr std::uint32_t kMagic = 0x07230203, kHeaderWords = 5;
    if (math == MetalSafeMath::never || words.size() < kHeaderWords || words[0] != kMagic) return unchanged();
    // Every result id takes an instruction of at least two words, so the bound of a module numbered densely, as
    // the translator's are, is well below its word count. A larger bound is malformed or numbered sparsely; the
    // tables are not sized by it (0xffffffff would take 16 GiB) and the module stays as it is.
    const std::uint32_t bound = words[3];
    if (bound < 2 || bound > words.size()) return unchanged();

    constexpr std::uint32_t kEntryPoint = 15, kExecutionMode = 16, kDecorate = 71, kDecorationGroup = 73,
                            kGroupDecorate = 74, kCopyObject = 83, kBitcast = 124, kFNegate = 127, kFAdd = 129,
                            kFSub = 131, kFMul = 133;
    constexpr std::uint32_t kNoContraction = 42, kSignedZeroInfNanPreserve = 4461;
    std::vector<std::uint32_t> source(bound, 0);  // the operand of a negate, copy or bitcast
    std::vector<bool> multiply(bound, false), decorated(bound, false), group(bound, false), preserves(bound, false);
    struct Sum {
        std::uint32_t result, operands[2];
    };
    std::vector<Sum> sums;
    std::vector<std::uint32_t> entries;
    std::vector<std::size_t> group_decorations;  // where each OpGroupDecorate starts
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
        const auto ids = [&](std::initializer_list<std::uint32_t> ks) {
            for (const std::uint32_t k : ks) id(k);
        };
        switch (op) {
        case kEntryPoint:
            if (count < 4) malformed = true;
            entries.push_back(id(2));
            break;
        case kExecutionMode:
            if (count < 3) malformed = true;
            else if (words[i + 2] == kSignedZeroInfNanPreserve) {
                if (count < 4) malformed = true;  // the mode's bit-width operand
                preserves[id(1)] = true;
            } else id(1);
            break;
        case kDecorate:
            if (count < 3) malformed = true;
            else if (words[i + 2] == kNoContraction) {
                if (count != 3) malformed = true;  // NoContraction takes no operands
                decorated[id(1)] = true;
                any_decorated = true;
            } else id(1);
            break;
        case kDecorationGroup:
            if (count != 2) malformed = true;
            group[id(1)] = true;
            break;
        case kGroupDecorate:
            for (std::uint32_t k = 1; k < count; ++k) id(k);
            if (count < 2) malformed = true;
            group_decorations.push_back(i);
            break;
        case kFMul:
            if (count != 5) malformed = true;
            ids({1, 3, 4});
            multiply[id(2)] = true;
            break;
        case kFNegate:
        case kCopyObject:
        case kBitcast:
            if (count != 4) malformed = true;
            id(1);
            source[id(2)] = id(3);
            break;
        case kFAdd:
        case kFSub:
            if (count != 5) malformed = true;
            id(1);
            sums.push_back({id(2), {id(3), id(4)}});
            break;
        default:
            break;
        }
        i += count;
    }
    if (malformed || !any_decorated || entries.empty()) return unchanged();
    if (math == MetalSafeMath::if_declared)
        for (const std::uint32_t entry : entries)
            if (!preserves[entry]) return unchanged();
    // A group's decorations are left as they are; its members count as decorated for the adds below.
    for (const std::size_t at : group_decorations)
        if (decorated[words[at + 1]])
            for (std::uint32_t k = 2; k < (words[at] >> 16); ++k) decorated[words[at + k]] = true;

    // The multiply a value is, through negates, copies and bitcasts, or 0. Every value on a chain is resolved
    // once; a cycle, which only an invalid module has, ends its chain without a multiply.
    constexpr std::uint32_t kUnresolved = ~0u;
    std::vector<std::uint32_t> origin(bound, kUnresolved), chain;
    const auto multiply_of = [&](std::uint32_t value) {
        std::uint32_t found = 0;
        chain.clear();
        for (std::uint32_t v = value; v; v = source[v]) {
            if (origin[v] != kUnresolved) {
                found = origin[v];
                break;
            }
            if (multiply[v]) {
                found = v;
                break;
            }
            origin[v] = 0;  // on this chain: meeting it again is a cycle
            chain.push_back(v);
        }
        for (const std::uint32_t v : chain) origin[v] = found;
        return found;
    };
    std::vector<bool> keep(bound, false);
    for (const Sum& sum : sums)
        for (const std::uint32_t operand : sum.operands)
            if (const std::uint32_t product = multiply_of(operand)) {
                keep[product] = true;  // spvFMul keeps the product out of the add's expression
                if (!decorated[product]) keep[sum.result] = true;  // or else spvFAdd / spvFSub has to
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
