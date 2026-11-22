// CPU-only tests for moltenvk_no_contraction.h: which devices get NoContraction dropped, and which decorations a
// module keeps. The modules are built by hand; no Vulkan instance or GPU is needed.
//
//   moltenvk_no_contraction_test [--spirv-val <spirv-val> <scratch dir>]
//
// With --spirv-val, every module a test builds as valid is written to the scratch directory, before and after the
// rewrite, and must pass `spirv-val --target-env vulkan1.2`.
#include "../src/moltenvk_no_contraction.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;
const char* spirv_val = nullptr;
std::string scratch;
int validated = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

using leonida::MetalSafeMath;
using leonida::MoltenVKSettings;
using Words = std::vector<std::uint32_t>;

constexpr std::uint32_t kNoContraction = 42, kSzinp = 4461, kSzinpCapability = 4466;
constexpr std::uint32_t kDecorate = 71, kDecorationGroup = 73, kGroupDecorate = 74, kCopyObject = 83,
                        kFNegate = 127, kFAdd = 129, kFSub = 131, kFMul = 133;

// The ids every module has; a test's own values start at kFirstValue.
enum : std::uint32_t { kVoid = 1, kFloat, kFunctionType, kA, kB, kC, kMain, kLabel, kFirstValue };

std::uint32_t opcode(std::uint32_t code, std::size_t operands) { return std::uint32_t(operands + 1) << 16 | code; }

// A GLCompute module, SPIR-V 1.4, whose function computes `body` on the constants a = 1, b = 2 and c = 3.
struct Module {
    bool signed_zero_inf_nan_preserve = true;
    std::set<std::uint32_t> decorated;     // OpDecorate %id NoContraction
    std::vector<Words> annotations;        // further annotation instructions, after the decorations
    std::vector<Words> body;               // instructions between OpLabel and OpReturn
    std::uint32_t next = kFirstValue;

    std::uint32_t value(std::uint32_t code, std::initializer_list<std::uint32_t> operands) {
        Words op{opcode(code, operands.size() + 2), kFloat, next};
        op.insert(op.end(), operands);
        body.push_back(op);
        return next++;
    }
    std::uint32_t mul(std::uint32_t a, std::uint32_t b) { return value(kFMul, {a, b}); }
    std::uint32_t add(std::uint32_t a, std::uint32_t b) { return value(kFAdd, {a, b}); }
    std::uint32_t sub(std::uint32_t a, std::uint32_t b) { return value(kFSub, {a, b}); }
    std::uint32_t neg(std::uint32_t a) { return value(kFNegate, {a}); }
    std::uint32_t copy(std::uint32_t a) { return value(kCopyObject, {a}); }
    std::uint32_t group() { return next++; }

    Words words() const {
        Words w{0x07230203, 0x00010400, 0, next, 0};
        const auto op = [&](std::uint32_t code, std::initializer_list<std::uint32_t> operands) {
            w.push_back(opcode(code, operands.size()));
            w.insert(w.end(), operands);
        };
        op(17, {1});  // OpCapability Shader
        if (signed_zero_inf_nan_preserve) op(17, {kSzinpCapability});
        op(14, {0, 1});                            // OpMemoryModel Logical GLSL450
        op(15, {5, kMain, 0x6e69616d, 0});         // OpEntryPoint GLCompute %main "main"
        op(16, {kMain, 17, 1, 1, 1});              // OpExecutionMode %main LocalSize 1 1 1
        if (signed_zero_inf_nan_preserve) op(16, {kMain, kSzinp, 32});
        for (const std::uint32_t id : decorated) op(kDecorate, {id, kNoContraction});
        for (const Words& a : annotations) w.insert(w.end(), a.begin(), a.end());
        op(19, {kVoid});                           // OpTypeVoid
        op(22, {kFloat, 32});                      // OpTypeFloat 32
        op(33, {kFunctionType, kVoid});            // OpTypeFunction
        op(43, {kFloat, kA, 0x3f800000});          // OpConstant 1.0
        op(43, {kFloat, kB, 0x40000000});          // 2.0
        op(43, {kFloat, kC, 0x40400000});          // 3.0
        op(54, {kVoid, kMain, 0, kFunctionType});  // OpFunction
        op(248, {kLabel});                         // OpLabel
        for (const Words& b : body) w.insert(w.end(), b.begin(), b.end());
        op(253, {});  // OpReturn
        op(56, {});   // OpFunctionEnd
        return w;
    }
    Module with_decorated(std::set<std::uint32_t> ids) const {
        Module m = *this;
        m.decorated = std::move(ids);
        return m;
    }
};

bool write(const std::string& path, const Words& words) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    f.close();
    return !f.fail();
}

// A module the test calls valid must pass spirv-val, when the test has it.
void validate(const Words& words, const char* what) {
    if (!spirv_val) return;
    const std::string path = scratch + "/moltenvk_no_contraction_" + std::to_string(validated++) + ".spv";
    if (!write(path, words)) {
        expect(false, ("cannot write " + path).c_str());
        return;
    }
    const std::string command =
        std::string("\"") + spirv_val + "\" --target-env vulkan1.2 \"" + path + "\"";
    if (std::system(command.c_str()) != 0) {
        std::fprintf(stderr, "spirv-val rejects %s (%s)\n", path.c_str(), what);
        ++failures;
    }
}

// Rewrites `before` for `math` and checks the result is `after`; both must be valid.
void expect_rewrite(const Module& before, MetalSafeMath math, const Module& after, const char* what) {
    const Words in = before.words(), out = after.words();
    validate(in, what);
    validate(out, what);
    expect(leonida::drop_no_contraction(in, math) == out, what);
}

void expect_unchanged(const Words& words, const char* what) {
    expect(leonida::drop_no_contraction(words, MetalSafeMath::always) == words, what);
}

void test_policy() {
    using leonida::moltenvk_safe_math;
    constexpr std::uint32_t v142 = 10402, v133 = 10303;
    constexpr auto env = MoltenVKSettings::environment;
    expect(moltenvk_safe_math(VK_DRIVER_ID_MESA_RADV, v142, env, nullptr) == MetalSafeMath::never, "RADV");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MESA_RADV, v142, env, "0") == MetalSafeMath::never, "RADV, MVK env set");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v142, env, nullptr) == MetalSafeMath::if_declared, "1.4.2 default");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v142, env, "") == MetalSafeMath::if_declared, "1.4.2, empty env");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v142, env, "0") == MetalSafeMath::always, "1.4.2 fast math off");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v142, env, "1") == MetalSafeMath::never, "1.4.2 always fast");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v142, env, "2") == MetalSafeMath::if_declared, "1.4.2 on demand");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v142, env, "yes") == MetalSafeMath::never, "1.4.2 unknown value");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v133, env, nullptr) == MetalSafeMath::never, "1.3.3 default");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v133, env, "0") == MetalSafeMath::always, "1.3.3 fast math off");
    expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v133, env, "2") == MetalSafeMath::if_declared, "1.3.3 on demand");

    // A caller's VkInstance may have set fast math through VK_EXT_layer_settings, which overrides the environment:
    // whatever the environment says, every decoration stays.
    constexpr auto unknown = MoltenVKSettings::unknown;
    for (const char* value : {static_cast<const char*>(nullptr), "", "0", "1", "2"}) {
        expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v142, unknown, value) == MetalSafeMath::never,
               "caller's instance, 1.4.2");
        expect(moltenvk_safe_math(VK_DRIVER_ID_MOLTENVK, v133, unknown, value) == MetalSafeMath::never,
               "caller's instance, 1.3.3");
    }
    expect(moltenvk_safe_math(nullptr, VK_NULL_HANDLE, env) == MetalSafeMath::never, "no vkGetPhysicalDeviceProperties2");
}

// a * b stored on its own, a + b, (a * c) + b, and a - -(b * c), every operation decorated.
struct Basic {
    Module m;
    std::uint32_t product, sum, fused_product, fused_sum, neg_product, neg, neg_sum;
    Basic() {
        product = m.mul(kA, kB);
        sum = m.add(kA, kB);
        fused_product = m.mul(kA, kC);
        fused_sum = m.add(fused_product, kB);
        neg_product = m.mul(kB, kC);
        neg = m.neg(neg_product);
        neg_sum = m.sub(kA, neg);
        m.decorated = {product, sum, fused_product, fused_sum, neg_product, neg, neg_sum};
    }
};

void test_drop() {
    const Basic basic;
    const Module& declared = basic.m;
    // The multiplies an add or subtract uses keep theirs; their own decorations keep the adds apart.
    const Module kept = declared.with_decorated({basic.fused_product, basic.neg_product});
    expect_rewrite(declared, MetalSafeMath::if_declared, kept, "declared, on demand: keep fused muls");
    expect_rewrite(declared, MetalSafeMath::always, kept, "declared, fast math off");
    expect_rewrite(declared, MetalSafeMath::never, declared, "declared, not MoltenVK");

    Module undeclared = declared;
    undeclared.signed_zero_inf_nan_preserve = false;
    Module undeclared_kept = undeclared.with_decorated(kept.decorated);
    expect_rewrite(undeclared, MetalSafeMath::if_declared, undeclared, "undeclared, on demand: unchanged");
    expect_rewrite(undeclared, MetalSafeMath::always, undeclared_kept, "undeclared, fast math off");

    expect_rewrite(declared.with_decorated({}), MetalSafeMath::always, declared.with_decorated({}), "no decorations");
}

// A multiply reaches its add through nine copies: more than a fixed hop limit of eight would follow.
void test_copy_chain() {
    Module m;
    const std::uint32_t product = m.mul(kA, kB);
    std::uint32_t value = product;
    for (int i = 0; i < 9; ++i) value = m.copy(value);
    const std::uint32_t sum = m.add(value, kC);
    const std::uint32_t lone = m.mul(kB, kC);
    m.decorated = {product, sum, lone};
    expect_rewrite(m, MetalSafeMath::always, m.with_decorated({product}), "nine copies: the multiply keeps its own");

    // With the multiply undecorated, the add's decoration is all that stops fusion.
    Module consumer = m.with_decorated({sum, lone});
    expect_rewrite(consumer, MetalSafeMath::always, consumer.with_decorated({sum}), "nine copies: the add keeps its own");
}

// NoContraction through a decoration group: the group's decorations are left alone, whatever its members are.
void test_group() {
    Module m;
    const std::uint32_t group = m.group();
    const std::uint32_t product = m.mul(kA, kB);
    const std::uint32_t sum = m.add(product, kC);
    const std::uint32_t lone = m.mul(kA, kC);        // in the group, used by nothing
    const std::uint32_t direct = m.mul(kB, kC);      // decorated directly, used by nothing
    const std::uint32_t direct_sum = m.add(kA, kB);  // decorated directly, no multiply
    (void)sum;
    // OpDecorate %group NoContraction comes before OpDecorationGroup %group, and OpGroupDecorate after it.
    m.decorated = {group, direct, direct_sum};
    m.annotations = {{opcode(kDecorationGroup, 1), group}, {opcode(kGroupDecorate, 3), group, product, lone}};
    expect_rewrite(m, MetalSafeMath::always, m.with_decorated({group}), "group: grouped decorations stay");

    // The add of a grouped multiply can drop its own decoration: spvFMul keeps the product out of it.
    Module with_sum = m;
    with_sum.decorated.insert(sum);
    expect_rewrite(with_sum, MetalSafeMath::always, m.with_decorated({group}), "group: its multiply's add");
}

// Only the add or subtract is decorated: dropping it would let Metal fuse the multiply in.
void test_decorated_consumer() {
    Module m;
    const std::uint32_t product = m.mul(kA, kB);
    const std::uint32_t sum = m.add(product, kC);
    const std::uint32_t negated = m.mul(kA, kC);
    const std::uint32_t neg = m.neg(negated);
    const std::uint32_t difference = m.sub(kB, neg);
    const std::uint32_t plain = m.add(kA, kB);
    (void)neg;
    m.decorated = {sum, difference, plain};
    expect_rewrite(m, MetalSafeMath::always, m.with_decorated({sum, difference}), "decorated adds of multiplies stay");
    expect_rewrite(m, MetalSafeMath::if_declared, m.with_decorated({sum, difference}), "on demand, same");
}

void test_malformed() {
    const Words valid = Basic().m.words();

    std::vector<std::uint32_t> truncated = valid;
    truncated.resize(truncated.size() - 3);  // OpFunctionEnd, OpReturn and the last word of the OpFSub
    expect_unchanged(truncated, "truncated module");
    expect_unchanged({1, 2, 3, 4, 5, 6}, "not SPIR-V");
    expect_unchanged({0x07230203, 0x00010400, 0, 16}, "a header of four words");
    expect_unchanged({}, "empty");

    // A bound of 0xffffffff must not size the tables: 16 GiB for four words of header.
    const auto start = std::chrono::steady_clock::now();
    expect_unchanged({0x07230203, 0x00010400, 0, 0xffffffff, 0}, "header only, bound 0xffffffff");
    Words huge = valid;
    huge[3] = 0xffffffff;
    expect_unchanged(huge, "module with bound 0xffffffff");
    expect(std::chrono::steady_clock::now() - start < std::chrono::seconds(1), "bound 0xffffffff is not allocated");

    Words zero_count = valid;
    zero_count.insert(zero_count.begin() + 5, 0);  // an instruction of zero words
    expect_unchanged(zero_count, "instruction word count 0");

    // OpExecutionMode %main SignedZeroInfNanPreserve without its bit width.
    Module module = Basic().m;
    module.signed_zero_inf_nan_preserve = false;
    Words no_width = module.words();
    for (std::size_t i = 5; i < no_width.size(); i += no_width[i] >> 16)
        if ((no_width[i] & 0xffff) == 14) {  // after OpMemoryModel's three words, OpEntryPoint's five
            const std::size_t at = i + 3 + 5;
            no_width.insert(no_width.begin() + at, {opcode(16, 2), kMain, kSzinp});
            break;
        }
    expect_unchanged(no_width, "SignedZeroInfNanPreserve without its width");
    expect(leonida::drop_no_contraction(no_width, MetalSafeMath::if_declared) == no_width, "same, on demand");

    Words out_of_bound = valid;
    for (std::size_t i = 5; i < out_of_bound.size(); i += out_of_bound[i] >> 16)
        if ((out_of_bound[i] & 0xffff) == kFAdd) {
            out_of_bound[i + 3] = out_of_bound[3];  // an operand at the bound
            break;
        }
    expect_unchanged(out_of_bound, "operand id at the bound");

    Words decorate_operand = valid;
    for (std::size_t i = 5; i < decorate_operand.size(); i += decorate_operand[i] >> 16)
        if ((decorate_operand[i] & 0xffff) == kDecorate) {
            decorate_operand[i] += 1u << 16;
            decorate_operand.insert(decorate_operand.begin() + i + 3, 7);  // NoContraction 7
            break;
        }
    expect_unchanged(decorate_operand, "NoContraction with an operand");

    // A copy that is its own operand: invalid, and must not hang.
    Module cycle;
    const std::uint32_t self = cycle.next;
    cycle.body.push_back({opcode(kCopyObject, 3), kFloat, self, self});
    cycle.next++;
    const std::uint32_t sum = cycle.add(self, kA);
    cycle.decorated = {sum};
    const Words cyclic = cycle.words();
    expect(leonida::drop_no_contraction(cyclic, MetalSafeMath::always) == cycle.with_decorated({}).words(),
           "a cycle of copies ends without a multiply");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--spirv-val") {
        spirv_val = argv[2];
        scratch = argv[3];
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--spirv-val <spirv-val> <scratch dir>]\n", argv[0]);
        return 2;
    }
    test_policy();
    test_drop();
    test_copy_chain();
    test_group();
    test_decorated_consumer();
    test_malformed();
    if (failures) return 1;
    if (spirv_val) std::printf("moltenvk_no_contraction: %d modules pass spirv-val\n", validated);
    std::printf("moltenvk_no_contraction: all tests passed\n");
    return 0;
}
