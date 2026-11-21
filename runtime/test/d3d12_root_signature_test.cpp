// Unit tests for the D3D12 root-signature parser (PR A). Synthetic blobs only, no game data.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "d3d12_root_signature.h"

namespace {

int failures = 0;

#define EXPECT(condition)                                                \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

using namespace rt::d3d12;

class Writer {
public:
    void u32(std::uint32_t value) {
        std::uint8_t bytes[4];
        std::memcpy(bytes, &value, 4);
        data.insert(data.end(), bytes, bytes + 4);
    }
    void f32(float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, 4);
        u32(bits);
    }
    std::vector<std::uint8_t> data;
};

std::vector<std::uint8_t> make_signature(std::uint32_t version) {
    Writer w;
    w.u32(version);
    w.u32(0x1 | 0x2);  // DENY_VERTEX_SHADER_ROOT_ACCESS | DENY_HULL_SHADER_ROOT_ACCESS
    w.u32(4);          // parameters

    // Descriptor table with a CBV range and an SRV range.
    w.u32(0);  // D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE
    w.u32(0);  // ALL
    w.u32(2);  // ranges
    for (int i = 0; i < 2; ++i) {
        w.u32(std::uint32_t(i == 0 ? 2 : 0));  // CBV, SRV
        w.u32(8);
        w.u32(3 + std::uint32_t(i));
        w.u32(0);
        w.u32(std::uint32_t(i * 8));
        if (version == 2) w.u32(0x1);  // DATA_VOLATILE
    }

    // 32-bit constants: 4 values at b2.
    w.u32(1);
    w.u32(1);  // VERTEX
    w.u32(2);
    w.u32(0);
    w.u32(4);

    // CBV descriptor at b0.
    w.u32(2);
    w.u32(5);  // PIXEL
    w.u32(0);
    w.u32(1);
    if (version == 2) w.u32(0x2);  // DATA_STATIC

    // UAV descriptor at u1.
    w.u32(4);
    w.u32(0);
    w.u32(1);
    w.u32(2);
    if (version == 2) w.u32(0);

    // One static sampler.
    w.u32(1);
    w.u32(0x55);  // filter
    w.u32(2);     // address u clamp
    w.u32(3);     // address v border
    w.u32(1);     // address w mirror
    w.f32(0.5f);
    w.u32(8);
    w.u32(4);
    w.u32(1);
    w.f32(0.0f);
    w.f32(16.0f);
    w.u32(5);
    w.u32(7);
    w.u32(5);  // PIXEL
    return w.data;
}

int test_parses_1_0() {
    const auto data = make_signature(1);
    RootSignatureDesc desc;
    std::string error;
    EXPECT(parse_root_signature(data.data(), data.size(), desc, &error));
    EXPECT(error.empty());
    EXPECT(desc.version == 1);
    EXPECT(desc.flags == 3u);
    EXPECT(desc.parameters.size() == 4);
    EXPECT(desc.parameters[0].type == RootParameterType::descriptor_table);
    EXPECT(desc.parameters[0].ranges.size() == 2);
    EXPECT(desc.parameters[0].ranges[0].type == DescriptorRangeType::cbv);
    EXPECT(desc.parameters[0].ranges[1].offset == 8u);
    EXPECT(desc.parameters[1].num_32bit_values == 4u);
    EXPECT(desc.parameters[2].visibility == ShaderVisibility::pixel);
    EXPECT(desc.parameters[3].shader_register == 1u);
    EXPECT(desc.static_samplers.size() == 1);
    EXPECT(desc.static_samplers[0].address_v == 3u);
    EXPECT(desc.static_samplers[0].max_lod == 16.0f);
    return failures;
}

int test_parses_1_1_with_flags() {
    const auto data = make_signature(2);
    RootSignatureDesc desc;
    std::string error;
    EXPECT(parse_root_signature(data.data(), data.size(), desc, &error));
    EXPECT(desc.version == 2);
    EXPECT(desc.parameters[0].ranges[0].flags == 1u);
    EXPECT(desc.parameters[2].flags == 2u);
    EXPECT(desc.parameters[3].flags == 0u);
    return failures;
}

int test_rejects_bad_input() {
    RootSignatureDesc desc;
    std::string error;
    EXPECT(!parse_root_signature(nullptr, 0, desc, &error));
    EXPECT(!error.empty());

    auto data = make_signature(1);
    EXPECT(!parse_root_signature(data.data(), 8, desc, &error));  // truncated header
    EXPECT(!parse_root_signature(data.data(), data.size() - 4, desc, &error));  // truncated sampler

    data = make_signature(1);
    data.push_back(0);
    EXPECT(!parse_root_signature(data.data(), data.size(), desc, &error));  // trailing byte
    return failures;
}

int test_rejects_unknown_values() {
    auto data = make_signature(1);
    // Version 3.
    std::uint32_t version = 3;
    std::memcpy(data.data(), &version, 4);
    RootSignatureDesc desc;
    std::string error;
    EXPECT(!parse_root_signature(data.data(), data.size(), desc, &error));

    // Unknown parameter type (first parameter starts at offset 12).
    data = make_signature(1);
    std::uint32_t bad_type = 99;
    std::memcpy(data.data() + 12, &bad_type, 4);
    EXPECT(!parse_root_signature(data.data(), data.size(), desc, &error));

    // Absurd parameter count.
    data = make_signature(1);
    std::uint32_t huge = 0xFFFFFFFFu;
    std::memcpy(data.data() + 8, &huge, 4);
    EXPECT(!parse_root_signature(data.data(), data.size(), desc, &error));
    return failures;
}

}  // namespace

int main() {
    test_parses_1_0();
    test_parses_1_1_with_flags();
    test_rejects_bad_input();
    test_rejects_unknown_values();

    if (failures) {
        std::fprintf(stderr, "root_signature_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("root_signature_test: all tests passed\n");
    return 0;
}
