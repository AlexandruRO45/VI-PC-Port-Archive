// Portable CPU unit tests for runtime/src/depth24_fragment.h (D24-as-D32 fragment SPIR-V rewriter).
// Pure C++20 standard library; no Vulkan, Windows SDK, or game files required.

#include "depth24_fragment.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace leonida;
using namespace leonida::spirv24;

int failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

struct ParsedView {
    bool ok = false;
    std::uint32_t bound = 0;
    std::vector<Inst> insts;
};

ParsedView parse_words(const std::vector<std::uint32_t>& words) {
    ParsedView v;
    if (words.size() < 5 || words[0] != 0x07230203u) return v;
    v.bound = words[3];
    for (std::size_t w = 5; w < words.size();) {
        const std::uint32_t count = words[w] >> 16;
        if (count == 0 || w + count > words.size()) return v;
        v.insts.emplace_back(words.begin() + std::ptrdiff_t(w), words.begin() + std::ptrdiff_t(w + count));
        w += count;
    }
    v.ok = true;
    return v;
}

std::size_t count_op(const ParsedView& v, std::uint32_t op) {
    std::size_t n = 0;
    for (const Inst& i : v.insts)
        if (opcode(i) == op) ++n;
    return n;
}

bool has_mode(const ParsedView& v, std::uint32_t entry, std::uint32_t mode) {
    for (const Inst& i : v.insts)
        if (opcode(i) == op_execution_mode && i.size() >= 3 && i[1] == entry && i[2] == mode) return true;
    return false;
}

std::uint32_t find_builtin(const ParsedView& v, std::uint32_t builtin) {
    for (const Inst& i : v.insts)
        if (opcode(i) == op_decorate && i.size() >= 4 && i[2] == decoration_builtin && i[3] == builtin) return i[1];
    return 0;
}

bool entry_lists_id(const ParsedView& v, std::uint32_t entry, std::uint32_t id) {
    for (const Inst& i : v.insts) {
        if (opcode(i) == op_entry_point && i.size() > 2 && i[2] == entry) {
            const std::size_t start = 3 + string_words(i, 3);
            for (std::size_t k = start; k < i.size(); ++k)
                if (i[k] == id) return true;
        }
    }
    return false;
}

bool has_glsl_ext_op(const ParsedView& v, std::uint32_t ext_op) {
    for (const Inst& i : v.insts)
        if (opcode(i) == op_ext_inst && i.size() >= 5 && i[4] == ext_op) return true;
    return false;
}

std::size_t count_stores_to(const ParsedView& v, std::uint32_t target_id) {
    std::size_t n = 0;
    for (const Inst& i : v.insts)
        if (opcode(i) == op_store && i.size() >= 3 && i[1] == target_id) ++n;
    return n;
}

// Builds a minimal fragment SPIR-V module with configurable pieces for positive and malformed tests.
struct Builder {
    std::vector<std::uint32_t> w;
    explicit Builder(std::uint32_t bound) : w{0x07230203u, 0x00010000u, 0, bound, 0} {}
    void put(const Inst& i) { w.insert(w.end(), i.begin(), i.end()); }
    void entry(std::uint32_t model, std::uint32_t id, const char* name,
               std::initializer_list<std::uint32_t> iface = {}) {
        Inst ep = make(op_entry_point, {model, id});
        append_string(ep, name);
        for (std::uint32_t x : iface) ep.push_back(x);
        fix_length(ep);
        put(ep);
    }
};

std::vector<std::uint32_t> make_basic_fragment(bool with_conservative_mode, bool with_early_tests) {
    enum : std::uint32_t { id_main = 1, id_void = 2, id_fn = 3, id_label = 4, bound = 5 };
    Builder b(bound);
    b.put(make(op_capability, {1}));
    b.put(make(op_memory_model, {0, 1}));
    b.entry(model_fragment, id_main, "main");
    b.put(make(op_execution_mode, {id_main, mode_origin_upper_left}));
    if (with_conservative_mode) b.put(make(op_execution_mode, {id_main, mode_depth_greater}));
    if (with_early_tests) b.put(make(op_execution_mode, {id_main, mode_early_fragment_tests}));
    b.put(make(op_type_void, {id_void}));
    b.put(make(op_type_function, {id_fn, id_void}));
    b.put(make(op_function, {id_void, id_main, 0, id_fn}));
    b.put(make(op_label, {id_label}));
    b.put(make(op_return, {}));
    b.put(make(op_function_end, {}));
    return b.w;
}

std::vector<std::uint32_t> make_depth_writing_fragment(bool with_return, bool with_helper) {
    enum : std::uint32_t {
        id_main = 1, id_void = 2, id_fn = 3, id_f32 = 4, id_ptr_out_f32 = 5, id_frag_depth = 6,
        id_half = 7, id_label_entry = 8, id_helper = 9, id_label_helper = 10, id_label_ret2 = 11,
        id_bool = 12, id_true = 13, bound = 14
    };
    Builder b(bound);
    b.put(make(op_capability, {1}));
    b.put(make(op_memory_model, {0, 1}));
    b.entry(model_fragment, id_main, "main", {id_frag_depth});
    b.put(make(op_execution_mode, {id_main, mode_origin_upper_left}));
    b.put(make(op_execution_mode, {id_main, mode_depth_replacing}));
    b.put(make(op_decorate, {id_frag_depth, decoration_builtin, builtin_frag_depth}));
    b.put(make(op_type_void, {id_void}));
    b.put(make(op_type_function, {id_fn, id_void}));
    b.put(make(op_type_float, {id_f32, 32}));
    b.put(make(op_type_pointer, {id_ptr_out_f32, storage_output, id_f32}));
    b.put(make(op_type_bool, {id_bool}));
    b.put(make(41 /* OpConstantTrue */, {id_bool, id_true}));
    b.put(make(op_constant, {id_f32, id_half, bits(0.5f)}));
    b.put(make(op_variable, {id_ptr_out_f32, id_frag_depth, storage_output}));

    if (with_helper) {
        b.put(make(op_function, {id_void, id_helper, 0, id_fn}));
        b.put(make(op_label, {id_label_helper}));
        b.put(make(op_return, {}));
        b.put(make(op_function_end, {}));
    }

    b.put(make(op_function, {id_void, id_main, 0, id_fn}));
    b.put(make(op_label, {id_label_entry}));
    b.put(make(op_store, {id_frag_depth, id_half}));
    if (with_return) {
        b.put(make(250 /* OpBranchConditional */, {id_true, id_label_ret2, id_label_ret2}));
        b.put(make(op_label, {id_label_ret2}));
        b.put(make(op_return, {}));
    } else {
        b.put(make(252 /* OpKill */, {}));
    }
    b.put(make(op_function_end, {}));
    return b.w;
}

void test_bias_constant() {
    check(depth24_bias_constant(0) == 0.0f, "zero bias constant is 0");
    check(std::fabs(depth24_bias_constant(1) - float(1.0 / 16777215.0)) == 0.0f, "unit bias constant");
    check(depth24_bias_constant(-4) == -depth24_bias_constant(4), "negative bias constant is symmetric");
}

void test_depth_only_and_bias_modes() {
    const auto plain = depth24_depth_only_fragment({});
    const ParsedView vp = parse_words(plain);
    check(vp.ok, "depth-only plain module parses");
    check(has_mode(vp, 1, mode_depth_replacing), "depth-only sets DepthReplacing");
    const std::uint32_t fd = find_builtin(vp, builtin_frag_depth);
    const std::uint32_t fc = find_builtin(vp, builtin_frag_coord);
    check(fd != 0 && fc != 0, "depth-only declares FragDepth and FragCoord");
    check(entry_lists_id(vp, 1, fd) && entry_lists_id(vp, 1, fc), "depth-only lists builtins in entry interface");
    check(count_stores_to(vp, fd) == 1, "depth-only emits one FragDepth store");
    check(count_op(vp, op_dpdx) == 0 && count_op(vp, op_dpdy) == 0, "no derivatives without bias");

    const auto pos_clamp = depth24_depth_only_fragment(Depth24Bias{2, 1.5f, 0.01f});
    const ParsedView vpos = parse_words(pos_clamp);
    check(vpos.ok, "depth-only with positive bias/clamp parses");
    check(count_op(vpos, op_dpdx) == 1 && count_op(vpos, op_dpdy) == 1, "bias emits dpdx/dpdy");
    check(has_glsl_ext_op(vpos, glsl_fmin), "positive DepthBiasClamp uses FMin");

    const auto neg_clamp = depth24_depth_only_fragment(Depth24Bias{-2, -1.0f, -0.01f});
    const ParsedView vneg = parse_words(neg_clamp);
    check(vneg.ok, "depth-only with negative bias/clamp parses");
    check(has_glsl_ext_op(vneg, glsl_fmax), "negative DepthBiasClamp uses FMax");
}

void test_color_and_depth_writing_shaders() {
    std::string why;
    const auto patched_color = depth24_patch_fragment(make_basic_fragment(true, false), {}, why);
    check(!patched_color.empty() && why.empty(), "color fragment patches cleanly");
    const ParsedView vc = parse_words(patched_color);
    check(vc.ok, "patched color fragment parses");
    check(has_mode(vc, 1, mode_depth_replacing), "patched color sets DepthReplacing");
    check(!has_mode(vc, 1, mode_depth_greater), "conservative DepthGreater mode stripped");
    const std::uint32_t fd = find_builtin(vc, builtin_frag_depth);
    check(fd != 0 && count_stores_to(vc, fd) == 1, "patched color stores quantised FragDepth");

    // Depth-writing shader: quantises before entry OpReturn, ignores rasterizer bias, leaves helper untouched.
    why.clear();
    const auto patched_dw = depth24_patch_fragment(make_depth_writing_fragment(true, true),
                                                   Depth24Bias{4, 2.0f, 0.05f}, why);
    check(!patched_dw.empty() && why.empty(), "depth-writing fragment patches cleanly");
    const ParsedView vdw = parse_words(patched_dw);
    check(vdw.ok, "patched depth-writing fragment parses");
    check(find_builtin(vdw, builtin_frag_depth) == 6, "reuses existing FragDepth variable");
    check(find_builtin(vdw, builtin_frag_coord) == 0, "depth-writing shader does not add FragCoord");
    check(count_op(vdw, op_dpdx) == 0 && count_op(vdw, op_dpdy) == 0, "depth-writing shader applies no bias");
    // Original store + 1 quantised store before main's OpReturn; helper's OpReturn must NOT get a store.
    check(count_stores_to(vdw, 6) == 2, "only main's OpReturn receives quantised FragDepth store");
}

void test_existing_rejections() {
    std::string why;
    check(depth24_patch_fragment({1, 2, 3}, {}, why).empty() && !why.empty(), "rejects non-SPIR-V");

    why.clear();
    check(depth24_patch_fragment({0x07230203u, 0x00010000u, 0, 5, 0, (4u << 16) | op_capability}, {}, why).empty() &&
              !why.empty(),
          "rejects truncated instruction");

    why.clear();
    check(depth24_patch_fragment(make_basic_fragment(false, true), {}, why).empty() && !why.empty(),
          "rejects EarlyFragmentTests");

    // No fragment entry point (compute only).
    {
        Builder b(5);
        b.put(make(op_capability, {1}));
        b.put(make(op_memory_model, {0, 1}));
        b.entry(5 /* GLCompute */, 1, "cs");
        why.clear();
        check(depth24_patch_fragment(b.w, {}, why).empty() && !why.empty(), "rejects non-fragment shader");
    }

    // Two fragment entry points.
    {
        Builder b(6);
        b.put(make(op_capability, {1}));
        b.put(make(op_memory_model, {0, 1}));
        b.entry(model_fragment, 1, "fs1");
        b.entry(model_fragment, 2, "fs2");
        why.clear();
        check(depth24_patch_fragment(b.w, {}, why).empty() && !why.empty(), "rejects two fragment entry points");
    }
}

void test_malformed_layout_rejections() {
    std::string why;

    // 1. Fragment OpEntryPoint with no OpFunction at all.
    {
        Builder b(5);
        b.put(make(op_capability, {1}));
        b.put(make(op_memory_model, {0, 1}));
        b.entry(model_fragment, 1, "main");
        why.clear();
        const auto out = depth24_patch_fragment(b.w, {}, why);
        check(out.empty() && !why.empty(), "rejects module with no OpFunction");
    }

    // 2. Fragment OpEntryPoint %1, but module only defines helper OpFunction %9.
    {
        Builder b(12);
        b.put(make(op_capability, {1}));
        b.put(make(op_memory_model, {0, 1}));
        b.entry(model_fragment, 1, "main");
        b.put(make(op_type_void, {2}));
        b.put(make(op_type_function, {3, 2}));
        b.put(make(op_function, {2, 9, 0, 3}));
        b.put(make(op_label, {10}));
        b.put(make(op_return, {}));
        b.put(make(op_function_end, {}));
        why.clear();
        const auto out = depth24_patch_fragment(b.w, {}, why);
        check(out.empty() && !why.empty(), "rejects missing entry OpFunction when only helper exists");
    }

    // 3. Fragment entry OpFunction exists, but has no body instruction (only OpLabel + OpFunctionEnd).
    {
        Builder b(6);
        b.put(make(op_capability, {1}));
        b.put(make(op_memory_model, {0, 1}));
        b.entry(model_fragment, 1, "main");
        b.put(make(op_type_void, {2}));
        b.put(make(op_type_function, {3, 2}));
        b.put(make(op_function, {2, 1, 0, 3}));
        b.put(make(op_label, {4}));
        b.put(make(op_function_end, {}));
        why.clear();
        const auto out = depth24_patch_fragment(b.w, {}, why);
        check(out.empty() && !why.empty(), "rejects entry function with no body instruction");
    }

    // 4. Module without OpMemoryModel (GLSL.std.450 OpExtInstImport would never be spliced).
    {
        Builder b(6);
        b.put(make(op_capability, {1}));
        b.entry(model_fragment, 1, "main");
        b.put(make(op_type_void, {2}));
        b.put(make(op_type_function, {3, 2}));
        b.put(make(op_function, {2, 1, 0, 3}));
        b.put(make(op_label, {4}));
        b.put(make(op_return, {}));
        b.put(make(op_function_end, {}));
        why.clear();
        const auto out = depth24_patch_fragment(b.w, {}, why);
        check(out.empty() && !why.empty(), "rejects module without OpMemoryModel");
    }

    // 5. Depth-writing fragment shader whose entry function has no OpReturn (only OpKill).
    {
        why.clear();
        const auto out = depth24_patch_fragment(make_depth_writing_fragment(false, false), {}, why);
        check(out.empty() && !why.empty(), "rejects depth-writing entry without OpReturn");
    }
}

}  // namespace

int main() {
    test_bias_constant();
    test_depth_only_and_bias_modes();
    test_color_and_depth_writing_shaders();
    test_existing_rejections();
    test_malformed_layout_rejections();
    if (failures) {
        std::fprintf(stderr, "depth24_fragment_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("depth24_fragment_test: ok\n");
    return 0;
}
