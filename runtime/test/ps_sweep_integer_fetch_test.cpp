// CPU-only unit tests for ps_sweep_integer_fetch.h over hand-built SPIR-V word streams.
// No Vulkan instance, GPU or game files are needed.
#include "ps_sweep_integer_fetch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

int failures = 0;

void expect_kind(const char* name, const std::vector<std::uint32_t>& spv, const char* want) {
    const char* got = integer_fetch_type(spv);
    const bool ok = (got == nullptr && want == nullptr) || (got && want && std::strcmp(got, want) == 0);
    if (ok) return;
    std::fprintf(stderr, "FAIL %s: got %s, expected %s\n",
                 name, got ? got : "nullptr", want ? want : "nullptr");
    ++failures;
}

void push_inst(std::vector<std::uint32_t>& out, std::uint32_t op, std::initializer_list<std::uint32_t> args) {
    out.push_back(static_cast<std::uint32_t>((args.size() + 1) << 16) | op);
    out.insert(out.end(), args.begin(), args.end());
}

enum Ids : std::uint32_t {
    kMain = 1,
    kVoid = 2,
    kFn = 3,
    kInt = 4,
    kU32 = 5,
    kV2Int = 6,
    kV2U32 = 7,
    kV4Int = 8,
    kF32 = 9,
    kV2F32 = 10,
    kV4F32 = 11,
    kImg = 12,
    kArrLen = 13,
    kArr = 14,
    kPtrImg = 15,
    kPtrArr = 16,
    kPtrOut = 17,
    kTex = 18,
    kColor = 19,
    kZero = 20,
    kLod = 21,
    kArrIdx = 22,
    kCx = 23,
    kCy = 24,
    kCoord = 25,
    kExtraNull = 26,
    kEntry = 27,
    kZeroCopy = 28,
    kCoordCopy = 29,
    kElem = 30,
    kPtrCopy = 31,
    kLoaded = 32,
    kLoadedCopy = 33,
    kTexel = 34,
    kOut = 35,
    kOutCopy = 36,
    kIntArrLen = 37,
    kIntArr2 = 38,
    kMain2 = 39,
    kEntry2 = 40,
    kBound = 44,
};

struct BuildOpts {
    std::uint32_t signedness = 1;       // 1 = sint, 0 = uint
    bool unsigned_coord_on_sint = false;
    bool use_array = false;
    bool in_bounds_chain = false;
    std::uint32_t access_result_type = 0; // 0 = kPtrImg
    std::uint32_t array_len = 1;
    std::uint32_t arr_index = 0;
    std::uint32_t cx = 2;
    std::uint32_t cy = 3;
    std::uint32_t lod = 0;
    std::uint32_t fetch_mask = 2;       // ImageOperands Lod
    bool null_scalar_zero = false;
    bool null_cx = false;
    bool null_cy = false;
    bool null_vec_coord = false;
    bool extra_v4f_null = false;
    bool copy_objects = false;
    std::uint32_t coord_composite_type = 0; // 0 = default integer vec2
    bool add_int_arr2_type = false;
    bool emit_shader_cap = true;
    std::uint32_t capability = 1;       // Shader
    bool emit_memory_model = true;
    bool duplicate_memory_model = false;
    std::uint32_t addressing_model = 0; // Logical
    std::uint32_t entry_point_count = 1;
    std::uint32_t exec_model = 4;       // Fragment
    bool emit_set = true;
    std::uint32_t descriptor_set = 0;
    bool emit_binding = true;
    std::uint32_t binding = 0;
};

std::vector<std::uint32_t> build_module(const BuildOpts& o) {
    std::vector<std::uint32_t> m{0x07230203u, 0x00010000u, 0u, kBound, 0u};

    if (o.emit_shader_cap) push_inst(m, 17, {o.capability});
    if (o.emit_memory_model) push_inst(m, 14, {o.addressing_model, 1u});
    if (o.duplicate_memory_model) push_inst(m, 14, {o.addressing_model, 1u});

    for (std::uint32_t ep = 0; ep < o.entry_point_count; ++ep) {
        const std::uint32_t fn_id = ep == 0 ? kMain : kMain2;
        const std::uint32_t name_word = ep == 0 ? 0x6e69616du : 0x746c616du; // "main" / "malt"
        push_inst(m, 15, {o.exec_model, fn_id, name_word, 0u, kColor});
    }
    if (o.entry_point_count >= 1) push_inst(m, 16, {kMain, 7u}); // OriginUpperLeft
    if (o.entry_point_count >= 2) push_inst(m, 16, {kMain2, 7u});

    if (o.emit_set) push_inst(m, 71, {kTex, 34u, o.descriptor_set});
    if (o.emit_binding) push_inst(m, 71, {kTex, 33u, o.binding});
    push_inst(m, 71, {kColor, 30u, 0u}); // Location 0

    push_inst(m, 19, {kVoid});
    push_inst(m, 33, {kFn, kVoid});
    push_inst(m, 21, {kInt, 32u, o.signedness});
    const std::uint32_t coord_scalar = (o.signedness == 1 && o.unsigned_coord_on_sint) ? kU32 : kInt;
    const std::uint32_t coord_vec2 = (o.signedness == 1 && o.unsigned_coord_on_sint) ? kV2U32 : kV2Int;
    if (o.signedness == 1 && o.unsigned_coord_on_sint) {
        push_inst(m, 21, {kU32, 32u, 0u});
        push_inst(m, 23, {kV2U32, kU32, 2u});
    }
    push_inst(m, 23, {kV2Int, kInt, 2u});
    push_inst(m, 23, {kV4Int, kInt, 4u});
    push_inst(m, 22, {kF32, 32u});
    push_inst(m, 23, {kV2F32, kF32, 2u});
    push_inst(m, 23, {kV4F32, kF32, 4u});
    push_inst(m, 25, {kImg, kInt, 1u, 0u, 0u, 0u, 1u, 0u});

    if (o.add_int_arr2_type) {
        push_inst(m, 43, {kInt, kIntArrLen, 2u});
        push_inst(m, 28, {kIntArr2, kInt, kIntArrLen});
    }

    push_inst(m, 32, {kPtrImg, 0u, kImg});
    if (o.use_array) {
        push_inst(m, 43, {kInt, kArrLen, o.array_len});
        push_inst(m, 28, {kArr, kImg, kArrLen});
        push_inst(m, 32, {kPtrArr, 0u, kArr});
        push_inst(m, 59, {kPtrArr, kTex, 0u});
    } else {
        push_inst(m, 59, {kPtrImg, kTex, 0u});
    }
    push_inst(m, 32, {kPtrOut, 3u, kV4F32});
    push_inst(m, 59, {kPtrOut, kColor, 3u});

    if (o.null_scalar_zero) push_inst(m, 46, {kInt, kZero});
    else push_inst(m, 43, {kInt, kZero, 0u});

    const std::uint32_t lod_id = o.lod == 0 ? kZero : kLod;
    if (o.lod != 0) push_inst(m, 43, {kInt, kLod, o.lod});

    const std::uint32_t idx_id = o.arr_index == 0 ? kZero : kArrIdx;
    if (o.arr_index != 0) push_inst(m, 43, {kInt, kArrIdx, o.arr_index});

    const std::uint32_t cx_id = o.null_cx ? kZero : kCx;
    if (!o.null_cx) push_inst(m, 43, {coord_scalar, kCx, o.cx});
    const std::uint32_t cy_id = o.null_cy ? kZero : kCy;
    if (!o.null_cy) push_inst(m, 43, {coord_scalar, kCy, o.cy});

    const std::uint32_t comp_type = o.coord_composite_type ? o.coord_composite_type : coord_vec2;
    if (o.null_vec_coord) push_inst(m, 46, {comp_type, kCoord});
    else push_inst(m, 44, {comp_type, kCoord, cx_id, cy_id});

    if (o.extra_v4f_null) push_inst(m, 46, {kV4F32, kExtraNull});

    push_inst(m, 54, {kVoid, kMain, 0u, kFn});
    push_inst(m, 248, {kEntry});

    std::uint32_t use_zero = lod_id;
    std::uint32_t use_idx = idx_id;
    std::uint32_t use_coord = kCoord;
    if (o.copy_objects) {
        push_inst(m, 83, {kInt, kZeroCopy, lod_id});
        push_inst(m, 83, {comp_type, kCoordCopy, kCoord});
        use_zero = kZeroCopy;
        if (o.arr_index == o.lod) use_idx = kZeroCopy;
        use_coord = kCoordCopy;
    }

    std::uint32_t img_ptr = kTex;
    if (o.use_array) {
        const std::uint32_t chain_op = o.in_bounds_chain ? 66u : 65u;
        const std::uint32_t res_type = o.access_result_type ? o.access_result_type : kPtrImg;
        push_inst(m, chain_op, {res_type, kElem, kTex, use_idx});
        img_ptr = kElem;
    }
    if (o.copy_objects) {
        push_inst(m, 83, {kPtrImg, kPtrCopy, img_ptr});
        img_ptr = kPtrCopy;
    }

    push_inst(m, 61, {kImg, kLoaded, img_ptr});
    std::uint32_t use_img = kLoaded;
    if (o.copy_objects) {
        push_inst(m, 83, {kImg, kLoadedCopy, kLoaded});
        use_img = kLoadedCopy;
    }

    push_inst(m, 95, {kV4Int, kTexel, use_img, use_coord, o.fetch_mask, use_zero});
    push_inst(m, o.signedness ? 111u : 112u, {kV4F32, kOut, kTexel});
    std::uint32_t store_val = kOut;
    if (o.copy_objects) {
        push_inst(m, 83, {kV4F32, kOutCopy, kOut});
        store_val = kOutCopy;
    }
    push_inst(m, 62, {kColor, store_val});
    push_inst(m, 253, {});
    push_inst(m, 56, {});

    if (o.entry_point_count >= 2) {
        push_inst(m, 54, {kVoid, kMain2, 0u, kFn});
        push_inst(m, 248, {kEntry2});
        push_inst(m, 253, {});
        push_inst(m, 56, {});
    }
    return m;
}

}  // namespace

int main() {
    // Baseline direct and 1-element array modules.
    {
        BuildOpts o;
        o.signedness = 1;
        expect_kind("direct sint", build_module(o), "sint");
        o.signedness = 0;
        expect_kind("direct uint", build_module(o), "uint");
        o.signedness = 1;
        o.use_array = true;
        expect_kind("array[1] sint OpAccessChain", build_module(o), "sint");
        o.signedness = 0;
        expect_kind("array[1] uint OpAccessChain", build_module(o), "uint");
    }
    {
        BuildOpts o;
        o.signedness = 1;
        o.unsigned_coord_on_sint = true;
        expect_kind("uvec2 coordinates on sint image", build_module(o), "sint");
        o.cx = 7;
        o.cy = 7;
        expect_kind("max in-bounds coordinate (7, 7)", build_module(o), "sint");
    }

    // OpConstantNull (46) for scalar 0 and 2D integer vector (0, 0).
    {
        BuildOpts o;
        o.signedness = 1;
        o.use_array = true;
        o.null_scalar_zero = true;
        o.null_vec_coord = true;
        o.extra_v4f_null = true;
        expect_kind("OpConstantNull scalar + ivec2 sint", build_module(o), "sint");
        o.signedness = 0;
        expect_kind("OpConstantNull scalar + uvec2 uint", build_module(o), "uint");
    }
    {
        BuildOpts o;
        o.null_scalar_zero = true;
        o.null_cx = true;
        o.cy = 7;
        expect_kind("mixed OpConstantNull x=0 and OpConstant y=7", build_module(o), "sint");
    }
    {
        auto m = build_module(BuildOpts{});
        // Malformed 4-word OpConstantNull before function body -> rejected.
        push_inst(m, 46, {kInt, 41u, 0u});
        expect_kind("malformed OpConstantNull word count", m, nullptr);
    }

    // OpInBoundsAccessChain (66) and AccessChain result-type validation.
    {
        BuildOpts o;
        o.use_array = true;
        o.in_bounds_chain = true;
        o.signedness = 1;
        expect_kind("OpInBoundsAccessChain sint", build_module(o), "sint");
        o.signedness = 0;
        expect_kind("OpInBoundsAccessChain uint", build_module(o), "uint");

        o.access_result_type = kPtrArr;
        expect_kind("OpInBoundsAccessChain wrong UniformConstant pointer target", build_module(o), nullptr);
        o.access_result_type = kPtrOut;
        expect_kind("OpInBoundsAccessChain non-UniformConstant pointer result", build_module(o), nullptr);

        o.in_bounds_chain = false;
        o.access_result_type = kPtrArr;
        expect_kind("OpAccessChain wrong pointer result type", build_module(o), nullptr);
    }

    // OpCopyObject (83) across constant, coordinate, pointer and loaded image.
    {
        BuildOpts o;
        o.signedness = 1;
        o.copy_objects = true;
        expect_kind("OpCopyObject direct sint", build_module(o), "sint");
        o.use_array = true;
        o.in_bounds_chain = true;
        o.null_scalar_zero = true;
        o.null_vec_coord = true;
        expect_kind("OpCopyObject + OpInBoundsAccessChain + OpConstantNull sint", build_module(o), "sint");
        o.signedness = 0;
        expect_kind("OpCopyObject + OpInBoundsAccessChain + OpConstantNull uint", build_module(o), "uint");
    }
    {
        auto m = build_module(BuildOpts{});
        // Malformed 3-word OpCopyObject -> rejected.
        push_inst(m, 83, {kInt, 41u});
        expect_kind("malformed OpCopyObject word count", m, nullptr);
    }
    {
        auto m = build_module(BuildOpts{});
        // OpCopyObject producing %img from an unverified id -> rejected.
        push_inst(m, 83, {kImg, 42u, kTex});
        expect_kind("OpCopyObject image from non-loaded source", m, nullptr);
    }

    // Coordinate composite type must be a 2-component 32-bit integer vector.
    {
        BuildOpts o;
        o.add_int_arr2_type = true;
        o.coord_composite_type = kIntArr2;
        expect_kind("OpConstantComposite of int[2] array rejected", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.coord_composite_type = kV2F32;
        expect_kind("OpConstantComposite of v2float rejected", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.coord_composite_type = kV4Int;
        expect_kind("OpConstantComposite of v4int with 2 args rejected", build_module(o), nullptr);
    }
    {
        // OpTypeStruct { int, int } (opcode 30) and composite of that struct -> rejected.
        auto m = build_module(BuildOpts{});
        push_inst(m, 30, {43u, kInt, kInt});
        expect_kind("OpTypeStruct rejected", m, nullptr);
    }

    // Mandatory SPIR-V module preamble presence and uniqueness checks.
    {
        BuildOpts o;
        o.emit_shader_cap = false;
        expect_kind("missing OpCapability Shader", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.capability = 11; // Int64 instead of Shader
        expect_kind("non-Shader capability", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.emit_memory_model = false;
        expect_kind("missing OpMemoryModel", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.duplicate_memory_model = true;
        expect_kind("duplicate OpMemoryModel", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.addressing_model = 5348; // PhysicalStorageBuffer64
        expect_kind("non-Logical addressing model", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.entry_point_count = 0;
        expect_kind("missing OpEntryPoint", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.entry_point_count = 2;
        expect_kind("multiple OpEntryPoint declarations", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.exec_model = 5; // GLCompute
        expect_kind("non-Fragment OpEntryPoint", build_module(o), nullptr);
    }

    // Coordinate, Lod, descriptor, array and binary-framing rejections.
    {
        BuildOpts o;
        o.cx = 8;
        expect_kind("out-of-bounds x=8", build_module(o), nullptr);
        o.cx = 0;
        o.cy = 8;
        expect_kind("out-of-bounds y=8", build_module(o), nullptr);
        o.cx = 0xffffffffu;
        o.cy = 0;
        expect_kind("negative coordinate x=-1", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.lod = 1;
        expect_kind("non-zero Lod=1", build_module(o), nullptr);
        o.lod = 0;
        o.fetch_mask = 0;
        expect_kind("ImageFetch without explicit Lod mask", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.descriptor_set = 1;
        expect_kind("non-zero DescriptorSet", build_module(o), nullptr);
        o.descriptor_set = 0;
        o.emit_set = false;
        expect_kind("missing DescriptorSet decoration", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.binding = 3;
        expect_kind("non-zero Binding", build_module(o), nullptr);
        o.binding = 0;
        o.emit_binding = false;
        expect_kind("missing Binding decoration", build_module(o), nullptr);
    }
    {
        BuildOpts o;
        o.use_array = true;
        o.array_len = 2;
        expect_kind("array size 2 rejected", build_module(o), nullptr);
        o.array_len = 1;
        o.arr_index = 1;
        expect_kind("array index 1 rejected", build_module(o), nullptr);
    }
    {
        expect_kind("empty vector", {}, nullptr);
        expect_kind("short header", {0x07230203u, 0x00010000u, 0u, 1u}, nullptr);
        auto bad_magic = build_module(BuildOpts{});
        bad_magic[0] = 0xdeadbeefu;
        expect_kind("wrong SPIR-V magic", bad_magic, nullptr);

        auto zero_len = build_module(BuildOpts{});
        zero_len.push_back(17u); // word count 0
        expect_kind("zero-length instruction", zero_len, nullptr);

        auto truncated = build_module(BuildOpts{});
        truncated.pop_back();
        // Popping 1-word OpFunctionEnd leaves valid trailing, so append a 4-word header with only 1 operand.
        truncated.push_back((4u << 16) | 43u);
        truncated.push_back(kInt);
        expect_kind("truncated instruction", truncated, nullptr);
    }

    if (failures) return 1;
    std::printf("ps_sweep_integer_fetch: all checks passed\n");
    return 0;
}
