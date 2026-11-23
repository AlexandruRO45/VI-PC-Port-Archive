// D3D-479 (docs/d3d12_vulkan.md 3.5): a graphics PSO whose root signature is built from a synthetic PS\0\x09
// PSO-cache user-data record, drawn and read back.
//
// One record is one stage's user-data block (#4358, runtime/src/pso_user_signature.h). Two records of the measured
// shape are built by hand:
//   A: one 0x401 entry (4 32-bit constants), then one 0x300 SRV in table SS=1;
//   B: the same constants, then an SRV (II=0) and a CBV (II=2) in one table SS=1.
// Each is mapped with rt::pso::map_user_record, converted to a D3D12 1.1 description and serialized by the host
// d3d12.dll. The layer parses the blob back (rt::rootsig::parse_rts0) and packs the parameters into s0..s31.
// The draws use vs_draw.vert (s0/s1, a clip-space offset) and ps_pso_usersig.frag (s2/s3, a tint), so the record's
// constants parameter is the visible effect: two quads with different constants, each behind its own mapped
// signature, and the descriptor table bound beside them where the packed layout puts its VA at s4:s5.
// The record does not carry the parameter order, the 0x401 base register or the root-signature flags; the test uses
// the mapper's Record order and register 0, and adds only ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, which a graphics PSO
// with an input layout needs (the layer's RootSignature defaults flags to 0).
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/pso_user_signature.h"
#include "../src/root_signature.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_USERSIG_SPV)
#error "LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_USERSIG_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: CPU handle, slot 10: GPU handle

namespace {

rt::pso::UserEntry user_entry(std::uint32_t kind, std::uint32_t count, std::uint32_t a, std::uint32_t b = 0,
                              std::uint32_t size = 4) {
    rt::pso::UserEntry entry;
    entry.kind = kind;
    entry.count = count;
    entry.a = a;
    entry.b = b;
    entry.size = size;
    return entry;
}

rt::pso::UserSlot user_slot(std::uint32_t table, std::uint32_t range, std::uint32_t width = 1) {
    rt::pso::UserSlot slot;
    slot.table = table;
    slot.range = range;
    slot.width = width;
    return slot;
}

// A: the 0x401 constants parameter, then one SRV in table SS=1. The parameter order is the mapper's Record
// hypothesis; the draw depends on it (constants land at s0..s3, the table VA at s4:s5).
rt::pso::UserRecord srv_and_constants_record() {
    rt::pso::UserRecord record;
    record.entries = {user_entry(rt::pso::kKindConstants, 4, 4),
                      user_entry(rt::pso::kKindSrv, 1, 0, 1)};
    record.slots = {user_slot(1, rt::pso::kRangeSrv)};
    return record;
}

// B: the same constants, then an SRV and a CBV sharing table SS=1 (two ranges, both with offset append).
rt::pso::UserRecord srv_cbv_table_record() {
    rt::pso::UserRecord record;
    record.entries = {user_entry(rt::pso::kKindConstants, 4, 4),
                      user_entry(rt::pso::kKindSrv, 1, 0, 1),
                      user_entry(rt::pso::kKindCbv, 1, 2, 1)};
    record.slots = {user_slot(1, rt::pso::kRangeSrv), user_slot(1, rt::pso::kRangeCbv)};
    return record;
}

// Owns the arrays a D3D12_VERSIONED_ROOT_SIGNATURE_DESC points at; keep it alive until serialization is done.
struct D3D12Signature {
    std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;
    std::vector<D3D12_ROOT_PARAMETER1> params;
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
};

// Converts the portable signature map_user_record produced. Stops on a root descriptor: the mapper emits only
// tables and 32-bit constants, so a descriptor would mean the model changed and the test must not invent fields.
bool to_d3d12(const rt::rootsig::RootSignature& signature, D3D12Signature& out, std::string& error) {
    std::size_t range_count = 0;
    for (const rt::rootsig::RootParameter& p : signature.params) range_count += p.ranges.size();
    out.ranges.assign(range_count, D3D12_DESCRIPTOR_RANGE1{});
    out.params.clear();
    out.params.reserve(signature.params.size());
    out.samplers.clear();
    out.samplers.reserve(signature.static_samplers.size());

    std::size_t range = 0;
    for (const rt::rootsig::RootParameter& p : signature.params) {
        D3D12_ROOT_PARAMETER1 param{};
        param.ShaderVisibility = static_cast<D3D12_SHADER_VISIBILITY>(p.visibility);
        switch (p.type) {
            case rt::rootsig::ParamType::Constants:
                param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
                param.Constants = {p.shader_register, p.register_space, p.num_values};
                break;
            case rt::rootsig::ParamType::Table: {
                param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                const std::size_t first = range;
                for (const rt::rootsig::DescriptorRange& r : p.ranges) {
                    D3D12_DESCRIPTOR_RANGE1& d3d = out.ranges[range++];
                    d3d.RangeType = static_cast<D3D12_DESCRIPTOR_RANGE_TYPE>(r.range_type);
                    d3d.NumDescriptors = r.num_descriptors;
                    d3d.BaseShaderRegister = r.base_shader_register;
                    d3d.RegisterSpace = r.register_space;
                    d3d.Flags = static_cast<D3D12_DESCRIPTOR_RANGE_FLAGS>(r.flags);
                    d3d.OffsetInDescriptorsFromTableStart = r.offset_in_table;
                }
                param.DescriptorTable.NumDescriptorRanges = UINT(p.ranges.size());
                param.DescriptorTable.pDescriptorRanges = out.ranges.data() + first;
                break;
            }
            case rt::rootsig::ParamType::Descriptor:
                error = "map_user_record produced a root descriptor, which this converter does not model";
                return false;
        }
        out.params.push_back(param);
    }

    for (const rt::rootsig::StaticSampler& s : signature.static_samplers) {
        D3D12_STATIC_SAMPLER_DESC d3d{};
        d3d.Filter = static_cast<D3D12_FILTER>(s.filter);
        d3d.AddressU = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(s.address_u);
        d3d.AddressV = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(s.address_v);
        d3d.AddressW = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(s.address_w);
        d3d.MipLODBias = s.mip_lod_bias;
        d3d.MaxAnisotropy = s.max_anisotropy;
        d3d.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(s.comparison_func);
        d3d.BorderColor = static_cast<D3D12_STATIC_BORDER_COLOR>(s.border_color);
        d3d.MinLOD = s.min_lod;
        d3d.MaxLOD = s.max_lod;
        d3d.ShaderRegister = s.shader_register;
        d3d.RegisterSpace = s.register_space;
        d3d.ShaderVisibility = static_cast<D3D12_SHADER_VISIBILITY>(s.visibility);
        out.samplers.push_back(d3d);
    }

    out.desc = {};
    out.desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    out.desc.Desc_1_1.NumParameters = UINT(out.params.size());
    out.desc.Desc_1_1.pParameters = out.params.data();
    out.desc.Desc_1_1.NumStaticSamplers = UINT(out.samplers.size());
    out.desc.Desc_1_1.pStaticSamplers = out.samplers.empty() ? nullptr : out.samplers.data();
    // Low 16 bits of the portable flags are the D3D12 meaning. The record carries no flags, so this is just the
    // input-assembler bit the graphics PSO below needs.
    out.desc.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(
        (signature.flags & 0xFFFFu) | D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
    return true;
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

}  // namespace

int main() {
    // The mapper's refusals, at CPU level: a malformed record never becomes a root signature.
    {
        rt::pso::UserRecord zero_count;
        zero_count.entries = {user_entry(rt::pso::kKindSrv, 0, 0, 1)};
        zero_count.slots = {user_slot(1, rt::pso::kRangeSrv)};
        const rt::pso::MapResult r = rt::pso::map_user_record(zero_count);
        EXPECT(!r.ok && r.error.find("zero descriptor count") != std::string::npos);

        rt::pso::UserRecord extra_slot;
        extra_slot.entries = {user_entry(rt::pso::kKindConstants, 4, 4)};
        extra_slot.slots = {user_slot(1, rt::pso::kRangeSrv)};
        const rt::pso::MapResult r2 = rt::pso::map_user_record(extra_slot);
        EXPECT(!r2.ok && r2.error.find("extra 0x42 slot") != std::string::npos);
        EXPECT(!r2.notes.empty() && r2.notes.front().find("extra slot") != std::string::npos);

        rt::pso::UserRecord wrong_slot;
        wrong_slot.entries = {user_entry(rt::pso::kKindSrv, 1, 0, 1)};
        wrong_slot.slots = {user_slot(1, rt::pso::kRangeCbv)};
        const rt::pso::MapResult r3 = rt::pso::map_user_record(wrong_slot);
        EXPECT(!r3.ok && r3.error.find("pairs with a CBV slot") != std::string::npos);
        std::printf("mapper refusals: zero count, extra 0x42 slot and SRV/CBV mismatch all refused\n");
    }

    // Record A: [0] 4 constants (s0..s3), [1] one SRV table (VA at s4:s5).
    const rt::pso::MapResult srv_mapped = rt::pso::map_user_record(srv_and_constants_record());
    EXPECT(srv_mapped.ok);
    EXPECT(srv_mapped.signature.params.size() == 2);
    {
        const auto& constants = srv_mapped.signature.params[0];
        const auto& table = srv_mapped.signature.params[1];
        EXPECT(constants.type == rt::rootsig::ParamType::Constants && constants.num_values == 4);
        EXPECT(constants.visibility == rt::rootsig::Visibility::All);
        EXPECT(table.type == rt::rootsig::ParamType::Table && table.ranges.size() == 1);
        EXPECT(table.ranges[0].range_type == rt::pso::kRangeSrv && table.ranges[0].num_descriptors == 1);
        EXPECT(table.ranges[0].base_shader_register == 0 && table.ranges[0].register_space == 1);
        EXPECT(table.ranges[0].offset_in_table == rt::rootsig::kRangeOffsetAppend);
        // The layer's layout, which the shaders below observe: 4 constant dwords, then 2 for the table.
        EXPECT(srv_mapped.packed.ok && srv_mapped.packed.dword_count == 6 && srv_mapped.packed.d3d12_cost == 5);
        EXPECT(srv_mapped.packed.provenance.size() == 6);
        for (std::uint32_t d = 0; d < 4; ++d) {
            EXPECT(srv_mapped.packed.provenance[d].param_index == 0);
            EXPECT(srv_mapped.packed.provenance[d].sub_index == d);
        }
        EXPECT(srv_mapped.packed.provenance[4].param_index == 1 && srv_mapped.packed.provenance[5].param_index == 1);
        EXPECT(srv_mapped.console_dwords == 5);  // 1 per table + 4 constants
        EXPECT(std::find(srv_mapped.notes.begin(), srv_mapped.notes.end(),
                         "0x401 base shader register is not carried by the record (0 used)") !=
               srv_mapped.notes.end());
    }

    // Record B: [0] 4 constants, [1] one table with an SRV and a CBV range.
    const rt::pso::MapResult cbv_mapped = rt::pso::map_user_record(srv_cbv_table_record());
    EXPECT(cbv_mapped.ok);
    EXPECT(cbv_mapped.signature.params.size() == 2);
    {
        const auto& table = cbv_mapped.signature.params[1];
        EXPECT(table.type == rt::rootsig::ParamType::Table && table.ranges.size() == 2);
        EXPECT(table.ranges[0].range_type == rt::pso::kRangeSrv && table.ranges[0].num_descriptors == 1);
        EXPECT(table.ranges[1].range_type == rt::pso::kRangeCbv && table.ranges[1].num_descriptors == 1);
        EXPECT(table.ranges[1].base_shader_register == 2 && table.ranges[1].register_space == 1);
        EXPECT(table.ranges[1].offset_in_table == rt::rootsig::kRangeOffsetAppend);
        EXPECT(cbv_mapped.packed.ok && cbv_mapped.packed.dword_count == 6 && cbv_mapped.packed.d3d12_cost == 5);
    }
    std::printf("map_user_record: both synthetic records map and pack to 4 constants + 2 table dwords\n");

    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    // Shader modules and the translator-style index that maps their FSSM hashes to the compiled GLSL.
    const auto vs_program = fssm_program({0x56530101, 0xbf810000});
    const auto ps_program = fssm_program({0x50530101, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const testfiles::ScratchDir scratch("leonida_pso_usersig_draw_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), library / "vs_pso_usersig.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_USERSIG_SPV),
                                library / "ps_pso_usersig.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_pso_usersig\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps_pso_usersig\": {\"stage\": \"ps\", \"fssm_sha256\": \""
              << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    constexpr UINT size = 64;
    constexpr UINT pitch = size * 4;
    std::vector<std::uint8_t> pixels(std::size_t(pitch) * size);

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.subgroup = d.subgroup;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        // Both mapped signatures become serialized D3D12 descriptions and root signatures.
        D3D12Signature a_desc, b_desc;
        std::string convert_error;
        EXPECT(to_d3d12(srv_mapped.signature, a_desc, convert_error));
        EXPECT(convert_error.empty());
        EXPECT(a_desc.desc.Version == D3D_ROOT_SIGNATURE_VERSION_1_1 && a_desc.desc.Desc_1_1.NumParameters == 2);
        EXPECT(a_desc.params[0].ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS &&
               a_desc.params[0].Constants.Num32BitValues == 4);
        EXPECT(a_desc.params[1].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE &&
               a_desc.params[1].DescriptorTable.NumDescriptorRanges == 1);
        EXPECT(to_d3d12(cbv_mapped.signature, b_desc, convert_error));
        EXPECT(b_desc.desc.Desc_1_1.NumParameters == 2 &&
               b_desc.params[1].DescriptorTable.NumDescriptorRanges == 2);
        const std::vector<std::uint8_t> a_bytes = serialize_root_signature(a_desc.desc);
        const std::vector<std::uint8_t> b_bytes = serialize_root_signature(b_desc.desc);
        EXPECT(!a_bytes.empty() && !b_bytes.empty());
        Ref<ID3D12RootSignature> root_a, root_b;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, a_bytes.data(), a_bytes.size(), IID_PPV_ARGS(&root_a.p))));
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, b_bytes.data(), b_bytes.size(), IID_PPV_ARGS(&root_b.p))));

        // The PSO: vs_draw.vert / ps_pso_usersig.frag behind each root signature, POSITION + COLOR input layout.
        const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root_a.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso_a, pso_b;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso_a.p))));
        pso_desc.pRootSignature = root_b.p;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso_b.p))));

        // Resources: a quad in a DEFAULT vertex buffer, its indices and a 1x1 SRV in an upload heap, and the target.
        const Vertex vertices[4] = {{-1, 1, 255, 0, 0, 255}, {0, 1, 255, 0, 0, 255},
                                    {-1, 0, 255, 0, 0, 255}, {0, 0, 255, 0, 0, 255}};
        const std::uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vertex_buffer, upload, readback, target;
        const auto vb_desc = buffer_desc(sizeof(vertices));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&vertex_buffer.p))));
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(std::size_t(pitch) * size);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC target_desc{};
        target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        target_desc.Width = target_desc.Height = size;
        target_desc.DepthOrArraySize = 1;
        target_desc.MipLevels = 1;
        target_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        target_desc.SampleDesc.Count = 1;
        target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &target_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&target.p))));
        D3D12_RESOURCE_DESC texture_desc{};
        texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_desc.Width = texture_desc.Height = 1;
        texture_desc.DepthOrArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.SampleDesc.Count = 1;
        Ref<ID3D12Resource> texture;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&texture.p))));

        // Upload layout: vertices at 0, indices at 256, the texel at 1024, a 256-byte CBV block at 2048.
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        EXPECT(footprint.Footprint.RowPitch == 256 && 1024 + footprint.Footprint.RowPitch <= 4096);
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memcpy(up + 256, indices, sizeof(indices));
        const std::uint8_t texel[4] = {255, 0, 0, 255};
        std::memcpy(up + 1024, texel, sizeof(texel));
        const float cbv_block[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        std::memcpy(up + 2048, cbv_block, sizeof(cbv_block));
        upload->Unmap(0, nullptr);
        const D3D12_GPU_VIRTUAL_ADDRESS upload_va = upload->GetGPUVirtualAddress();

        // A shader-visible heap: slot 0 the SRV, slot 1 the CBV that record B's second range names.
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        const std::uint64_t cpu_handle = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 9))(srv_heap.p);
        const std::uint64_t gpu_handle = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 10))(srv_heap.p);
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(cpu_handle && gpu_handle >= 0x10000 && step >= 32);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D = {0, 1, 0, 0.0f};
        device->CreateShaderResourceView(texture.p, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu_handle)});
        const D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{upload_va + 2048, 256};
        device->CreateConstantBufferView(&cbv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(cpu_handle + step)});

        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso_a.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
        D3D12_TEXTURE_COPY_LOCATION tex_dst{};
        tex_dst.pResource = texture.p;
        tex_dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION tex_src{};
        tex_src.pResource = upload.p;
        tex_src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        tex_src.PlacedFootprint = footprint;
        tex_src.PlacedFootprint.Offset = 1024;
        list->CopyTextureRegion(&tex_dst, 0, 0, 0, &tex_src, nullptr);
        D3D12_RESOURCE_BARRIER barriers[3] = {
            transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
            transition(texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        list->ResourceBarrier(3, barriers);
        const FLOAT clear[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);

        list->SetGraphicsRootSignature(root_a.p);
        ID3D12DescriptorHeap* heaps[] = {srv_heap.p};
        list->SetDescriptorHeaps(1, heaps);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), sizeof(vertices), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &vbv);
        const D3D12_INDEX_BUFFER_VIEW ibv{upload_va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&ibv);
        const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        // Draw 1 through record A: constants at s0..s3 (offset 0,0 and a (64,128) tint), SRV table at s4:s5.
        const std::uint32_t constants_a[4] = {0, 0, float_bits(64.0f / 255.0f), float_bits(128.0f / 255.0f)};
        list->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{gpu_handle});
        list->SetGraphicsRoot32BitConstants(0, 4, constants_a, 0);
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);

        // Draw 2 through record B: same constant offsets, its two-range table bound at parameter 1. If the mapping
        // had put the table before the constants, s0/s1 would be the table VA and the quad would leave the screen.
        const std::uint32_t constants_b[4] = {float_bits(1.0f), float_bits(-1.0f),
                                              float_bits(192.0f / 255.0f), float_bits(32.0f / 255.0f)};
        list->SetGraphicsRootSignature(root_b.p);
        list->SetPipelineState(pso_b.p);
        list->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{gpu_handle});
        list->SetGraphicsRoot32BitConstants(0, 4, constants_b, 0);
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);

        D3D12_RESOURCE_BARRIER to_copy =
            transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, size, size, 1, pitch}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(pixels.data(), mapped, pixels.size());

        // Every child proxy holds a reference to the device proxy; release them before the count is meaningful.
        for (auto* r : {&vertex_buffer, &upload, &readback, &target, &texture}) r->reset();
        for (auto* p : {&pso_a, &pso_b}) p->reset();
        root_a.reset();
        root_b.reset();
        srv_heap.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    // Each 32x32 quadrant the draws covered is one colour; the rest is the clear colour.
    UINT mismatches = 0, top_left = 0, bottom_right = 0, background = 0;
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const bool top = row < size / 2, left = col < size / 2;
            using Pixel = std::array<std::uint8_t, 4>;
            const Pixel want = top && left     ? Pixel{64, 128, 0, 255}
                               : !top && !left ? Pixel{192, 32, 0, 255}
                                               : Pixel{0, 0, 255, 255};
            const std::uint8_t* got = pixels.data() + row * pitch + col * 4;
            if (!std::equal(want.begin(), want.end(), got)) {
                if (mismatches++ < 6)
                    std::fprintf(stderr, "pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row, got[0],
                                 got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
                continue;
            }
            if (top && left) ++top_left;
            else if (!top && !left) ++bottom_right;
            else ++background;
        }
    }
    EXPECT(!mismatches);
    EXPECT(top_left == 1024 && bottom_right == 1024 && background == 2048);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("D3D-479: two mapped PSO-cache records became D3D12 root signatures and drew: %u/%u pixels matched "
                "(top-left 64,128,0,255; bottom-right 192,32,0,255; background 0,0,255,255)\n",
                size * size, size * size);
    return 0;
}
