// Differential graphics-state fuzzer: one seeded random program clears a colour target and an optional depth-stencil
// target, then draws triangles through PSOs with random blend state (factors, ops, write masks, OMSetBlendFactor),
// depth-stencil state (functions, stencil ops, masks, front and back faces, OMSetStencilRef), a second render target
// with IndependentBlendEnable some of the time, UINT targets with logic ops, and rasterizer state
// (cull mode, winding, constant depth bias, depth clip), under a random viewport (offset, size, depth range) and
// scissor, on the layer and on the host's D3D12 on the same GPU. Every byte of the colour target
// and of every depth-stencil plane is read back and compared.
// Each triangle has one depth (its three vertices share z) and one colour (flat, from the first vertex, as D3D12 and
// Vulkan both provoke), and its vertices sit on pixel corners, so neither interpolation nor sub-pixel rounding can
// create or hide a difference: the state is what is tested. Blending may round differently within D3D12's tolerance,
// so a colour channel may differ by one code (UNORM) or one ulp (float); a depth by one code (D16/D24) or two ulps
// (D32, depth bias); stencil compares exactly.
// The layer runs GLSL stand-ins written to the translator's contract (test/glsl/vs_pso.vert, ps_pso.frag), found by the
// FSSM hash of their containers; the reference runs the same programs in HLSL compiled by d3dcompiler_47.
// --seeds N --start S choose the programs (default 40 from 1); the reference is the host's D3D12 on the layer's own
// GPU (by LUID), or WARP with --warp;
// --reduce SEED drops draws and triangles while the layer still differs and prints the smallest program left;
// --all also generates a nonzero DepthBias, which the default leaves out until D3D-325 (#2718) lands.
// --verbose prints every program. The layer is the device the runtime makes (create_vulkan_device_standalone);
// LEONIDA_VK_VALIDATION=1 runs it under VK_LAYER_KHRONOS_validation. Needs a Vulkan GPU (exit 77 otherwise, and
// under Wine). No game data.
#include "test_temp_dir.h"
#include "host_wine.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#if !defined(LEONIDA_PSO_VS_SPV) || !defined(LEONIDA_PSO_PS_SPV) || !defined(LEONIDA_PSO_PS_MRT_SPV) || \
    !defined(LEONIDA_PSO_PS_UINT_SPV)
#error "LEONIDA_PSO_VS_SPV, LEONIDA_PSO_PS_SPV, LEONIDA_PSO_PS_MRT_SPV and LEONIDA_PSO_PS_UINT_SPV must name the compiled test shaders"
#endif

using d3d12test::Ref;

namespace {

constexpr UINT kSize = 32;  // the targets are kSize x kSize

constexpr char kHostVs[] =
    "struct O { float4 pos : SV_Position; nointerpolation float4 color : COLOR; };\n"
    "O main(float4 p : POSITION, float4 c : COLOR) { O o; o.pos = p; o.color = c; return o; }\n";
constexpr char kHostPs[] =
    "float4 main(float4 pos : SV_Position, nointerpolation float4 c : COLOR) : SV_Target { return c; }\n";
constexpr char kHostPsUint[] =
    "uint4 main(float4 pos : SV_Position, nointerpolation float4 c : COLOR) : SV_Target { return asuint(c); }\n";
constexpr char kHostPsMrt[] =
    "struct O { float4 t0 : SV_Target0; float4 t1 : SV_Target1; };\n"
    "O main(float4 pos : SV_Position, nointerpolation float4 c : COLOR) { O o; o.t0 = c; o.t1 = c.zyxw; return o; }\n";

bool compile_hlsl(pD3DCompile compile, const char* source, const char* target, std::vector<std::uint8_t>& out) {
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", target, 0, 0, &code,
                               &errors);
    if (errors) {
        std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
        errors->Release();
    }
    if (FAILED(hr) || !code) return false;
    const auto* p = static_cast<const std::uint8_t*>(code->GetBufferPointer());
    out.assign(p, p + code->GetBufferSize());
    code->Release();
    return true;
}

// A colour target format: how its texels compare.
struct ColorFormat {
    DXGI_FORMAT format;
    const char* name;
    enum Kind { unorm8, unorm10_2, half, single, r11g11b10, uint } kind;
    UINT bytes;
};
constexpr ColorFormat kColorFormats[] = {
    {DXGI_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM", ColorFormat::unorm8, 4},
    {DXGI_FORMAT_B8G8R8A8_UNORM, "B8G8R8A8_UNORM", ColorFormat::unorm8, 4},
    {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, "R8G8B8A8_UNORM_SRGB", ColorFormat::unorm8, 4},
    {DXGI_FORMAT_R10G10B10A2_UNORM, "R10G10B10A2_UNORM", ColorFormat::unorm10_2, 4},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, "R16G16B16A16_FLOAT", ColorFormat::half, 8},
    {DXGI_FORMAT_R32G32B32A32_FLOAT, "R32G32B32A32_FLOAT", ColorFormat::single, 16},
    {DXGI_FORMAT_R11G11B10_FLOAT, "R11G11B10_FLOAT", ColorFormat::r11g11b10, 4},
    {DXGI_FORMAT_R8G8B8A8_UINT, "R8G8B8A8_UINT", ColorFormat::uint, 4},
    {DXGI_FORMAT_R16G16B16A16_UINT, "R16G16B16A16_UINT", ColorFormat::uint, 8},
    {DXGI_FORMAT_R32G32B32A32_UINT, "R32G32B32A32_UINT", ColorFormat::uint, 16},
    {DXGI_FORMAT_R32_UINT, "R32_UINT", ColorFormat::uint, 4},
};

struct DepthFormat {
    DXGI_FORMAT format;  // DXGI_FORMAT_UNKNOWN: no depth-stencil target
    const char* name;
    bool stencil;
};
constexpr DepthFormat kDepthFormats[] = {
    {DXGI_FORMAT_UNKNOWN, "none", false},
    {DXGI_FORMAT_D16_UNORM, "D16_UNORM", false},
    {DXGI_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT", true},
    {DXGI_FORMAT_D32_FLOAT, "D32_FLOAT", false},
    {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, "D32_FLOAT_S8X24_UINT", true},
};

struct Vertex {
    float x, y, z, w;
    float r, g, b, a;
};

struct Triangle {
    int px[3], py[3];  // pixel corners
    float z;
    float color[4];
};

struct Draw {
    D3D12_BLEND_DESC blend{};
    D3D12_DEPTH_STENCIL_DESC depth{};
    D3D12_RASTERIZER_DESC raster{};
    float blend_factor[4]{};
    UINT stencil_ref = 0;
    D3D12_VIEWPORT viewport{0, 0, float(kSize), float(kSize), 0.0f, 1.0f};
    D3D12_RECT scissor{0, 0, LONG(kSize), LONG(kSize)};
    std::vector<Triangle> triangles;
};

struct Program {
    const ColorFormat* color = nullptr;
    const ColorFormat* color2 = nullptr;  // a second render target, or none
    const DepthFormat* depth = nullptr;
    bool blendable = true, blendable2 = true;
    float clear_color[4]{}, clear_color2[4]{};
    float clear_depth = 1.0f;
    UINT8 clear_stencil = 0;
    std::vector<Draw> draws;
};

// The formats both devices can render (and blend) and the depth formats both can use.
struct Support {
    bool logic_op = false;  // D3D12_FEATURE_DATA_D3D12_OPTIONS::OutputMergerLogicOp on both
    std::vector<const ColorFormat*> colors;
    std::vector<bool> blendable;  // by colors index
    std::vector<const DepthFormat*> depths;
};

bool format_support(ID3D12Device* device, DXGI_FORMAT format, UINT need1) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{format};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof fs)) &&
           (fs.Support1 & need1) == need1;
}

Support support(ID3D12Device* a, ID3D12Device* b) {
    Support s;
    D3D12_FEATURE_DATA_D3D12_OPTIONS oa{}, ob{};
    s.logic_op = SUCCEEDED(a->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &oa, sizeof oa)) &&
                 SUCCEEDED(b->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &ob, sizeof ob)) &&
                 oa.OutputMergerLogicOp && ob.OutputMergerLogicOp;
    for (const ColorFormat& c : kColorFormats) {
        if (!format_support(a, c.format, D3D12_FORMAT_SUPPORT1_RENDER_TARGET) ||
            !format_support(b, c.format, D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
            continue;
        s.colors.push_back(&c);
        s.blendable.push_back(format_support(a, c.format, D3D12_FORMAT_SUPPORT1_BLENDABLE) &&
                              format_support(b, c.format, D3D12_FORMAT_SUPPORT1_BLENDABLE));
    }
    for (const DepthFormat& d : kDepthFormats)
        if (d.format == DXGI_FORMAT_UNKNOWN || (format_support(a, d.format, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) &&
                                                format_support(b, d.format, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)))
            s.depths.push_back(&d);
    return s;
}

float color_value(std::mt19937_64& rng) {
    constexpr float values[] = {0.0f, 1.0f, 0.5f, 0.25f, 0.75f, 2.0f, -0.5f, 1.0f / 255.0f, 128.0f / 255.0f,
                                0.333333f, 0.1f, 0.9f};
    if (rng() % 3 == 0) return float(rng() % 1001) / 1000.0f;
    return values[rng() % std::size(values)];
}

// Known differences the default programs leave out; --all puts them back.
// - A nonzero DepthBias: the layer applies about twice D3D12's constant bias on NVIDIA (D3D-325, fixed by #2718 through
//   VK_EXT_depth_bias_control). Without it every PSO gets DepthBias 0. The random draws are the same either way.
bool with_depth_bias = false;

Program generate(std::uint64_t seed, const Support& sup) {
    std::mt19937_64 rng(seed * 0x9e3779b97f4a7c15ull + 0x5bd1e995);
    const auto pick = [&](UINT n) { return UINT(rng() % n); };
    Program p;
    const UINT c = pick(UINT(sup.colors.size()));
    p.color = sup.colors[c];
    p.blendable = sup.blendable[c];
    p.depth = sup.depths[pick(UINT(sup.depths.size()))];
    for (float& v : p.clear_color) v = color_value(rng);
    const bool uint_target = p.color->kind == ColorFormat::uint;
    if (!uint_target && pick(3) == 0) {  // two render targets (float, UNORM; a UINT one needs its own shader)
        UINT c2 = pick(UINT(sup.colors.size()));
        while (sup.colors[c2]->kind == ColorFormat::uint) c2 = (c2 + 1) % UINT(sup.colors.size());
        p.color2 = sup.colors[c2];
        p.blendable2 = sup.blendable[c2];
        for (float& v : p.clear_color2) v = color_value(rng);
    }
    p.clear_depth = float(pick(65)) / 64.0f;
    p.clear_stencil = UINT8(rng());
    constexpr D3D12_BLEND factors[] = {
        D3D12_BLEND_ZERO, D3D12_BLEND_ONE, D3D12_BLEND_SRC_COLOR, D3D12_BLEND_INV_SRC_COLOR, D3D12_BLEND_SRC_ALPHA,
        D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA, D3D12_BLEND_INV_DEST_ALPHA, D3D12_BLEND_DEST_COLOR,
        D3D12_BLEND_INV_DEST_COLOR, D3D12_BLEND_SRC_ALPHA_SAT, D3D12_BLEND_BLEND_FACTOR,
        D3D12_BLEND_INV_BLEND_FACTOR};
    constexpr D3D12_BLEND alpha_factors[] = {
        D3D12_BLEND_ZERO, D3D12_BLEND_ONE, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA,
        D3D12_BLEND_INV_DEST_ALPHA, D3D12_BLEND_SRC_ALPHA_SAT, D3D12_BLEND_BLEND_FACTOR,
        D3D12_BLEND_INV_BLEND_FACTOR};
    const UINT draws = 1 + pick(5);
    for (UINT d = 0; d < draws; ++d) {
        Draw draw;
        const auto make_target = [&](D3D12_RENDER_TARGET_BLEND_DESC& rt, bool blendable) {
            rt.BlendEnable = blendable && pick(4) != 0;
            rt.SrcBlend = factors[pick(UINT(std::size(factors)))];
            rt.DestBlend = factors[pick(UINT(std::size(factors)))];
            rt.BlendOp = D3D12_BLEND_OP(1 + pick(5));
            rt.SrcBlendAlpha = alpha_factors[pick(UINT(std::size(alpha_factors)))];
            rt.DestBlendAlpha = alpha_factors[pick(UINT(std::size(alpha_factors)))];
            rt.BlendOpAlpha = D3D12_BLEND_OP(1 + pick(5));
            rt.LogicOp = D3D12_LOGIC_OP_NOOP;
            rt.RenderTargetWriteMask = UINT8(pick(3) ? D3D12_COLOR_WRITE_ENABLE_ALL : pick(16));
        };
        // A UINT target can't blend; it gets a logic op (all sixteen) half the time where both devices have them.
        // With two targets, RenderTarget[0] serves both unless IndependentBlendEnable gives the second its own.
        const bool independent = p.color2 && pick(2);
        make_target(draw.blend.RenderTarget[0], p.blendable && (!p.color2 || independent || p.blendable2));
        if (independent) {
            draw.blend.IndependentBlendEnable = TRUE;
            make_target(draw.blend.RenderTarget[1], p.blendable2);
        }
        if (uint_target && sup.logic_op && pick(2)) {
            draw.blend.RenderTarget[0].LogicOpEnable = TRUE;
            draw.blend.RenderTarget[0].LogicOp = D3D12_LOGIC_OP(pick(16));
        }
        for (float& v : draw.blend_factor) v = color_value(rng);
        D3D12_DEPTH_STENCIL_DESC& ds = draw.depth;
        if (p.depth->format != DXGI_FORMAT_UNKNOWN) {
            ds.DepthEnable = pick(4) != 0;
            ds.DepthWriteMask = pick(3) ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
            ds.DepthFunc = D3D12_COMPARISON_FUNC(1 + pick(8));
            if (p.depth->stencil && pick(2)) {
                ds.StencilEnable = TRUE;
                ds.StencilReadMask = UINT8(pick(3) ? 0xff : rng());
                ds.StencilWriteMask = UINT8(pick(3) ? 0xff : rng());
                for (D3D12_DEPTH_STENCILOP_DESC* face : {&ds.FrontFace, &ds.BackFace}) {
                    face->StencilFailOp = D3D12_STENCIL_OP(1 + pick(8));
                    face->StencilDepthFailOp = D3D12_STENCIL_OP(1 + pick(8));
                    face->StencilPassOp = D3D12_STENCIL_OP(1 + pick(8));
                    face->StencilFunc = D3D12_COMPARISON_FUNC(1 + pick(8));
                }
            } else {
                ds.StencilReadMask = ds.StencilWriteMask = 0xff;
                ds.FrontFace = ds.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                              D3D12_COMPARISON_FUNC_ALWAYS};
            }
        } else {
            ds.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            ds.StencilReadMask = ds.StencilWriteMask = 0xff;
            ds.FrontFace = ds.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                          D3D12_COMPARISON_FUNC_ALWAYS};
        }
        draw.stencil_ref = pick(256);
        D3D12_RASTERIZER_DESC& rs = draw.raster;
        rs.FillMode = D3D12_FILL_MODE_SOLID;
        rs.CullMode = D3D12_CULL_MODE(1 + pick(3));
        rs.FrontCounterClockwise = pick(2);
        rs.DepthBias = pick(3) ? 0 : int(pick(41)) - 20;
        if (!with_depth_bias) rs.DepthBias = 0;
        rs.DepthBiasClamp = 0.0f;
        rs.SlopeScaledDepthBias = 0.0f;
        rs.DepthClipEnable = pick(4) != 0;
        // A viewport elsewhere than the whole target (its y flip, a depth range) and a scissor, some of the time.
        // Offsets and sizes are whole pixels and the depth range is in 64ths, so vertices stay on pixel corners or
        // half-pixels and depths stay exact.
        if (pick(2)) {
            constexpr float sizes[] = {8.0f, 16.0f, 24.0f, 32.0f, 48.0f};
            draw.viewport.Width = sizes[pick(UINT(std::size(sizes)))];
            draw.viewport.Height = sizes[pick(UINT(std::size(sizes)))];
            draw.viewport.TopLeftX = float(int(pick(41)) - 16);
            draw.viewport.TopLeftY = float(int(pick(41)) - 16);
            const UINT a = pick(65), b = pick(65);
            draw.viewport.MinDepth = float(std::min(a, b)) / 64.0f;
            draw.viewport.MaxDepth = float(std::max(a, b)) / 64.0f;
        }
        if (pick(3) == 0) {
            const LONG x0 = LONG(pick(kSize)), y0 = LONG(pick(kSize));
            draw.scissor = {x0, y0, x0 + 1 + LONG(pick(kSize - UINT(x0))), y0 + 1 + LONG(pick(kSize - UINT(y0)))};
        }
        const UINT triangles = 1 + pick(3);
        for (UINT t = 0; t < triangles; ++t) {
            Triangle tri{};
            for (int v = 0; v < 3; ++v) {
                tri.px[v] = int(pick(kSize + 9)) - 4;  // a few vertices past the target's edges
                tri.py[v] = int(pick(kSize + 9)) - 4;
            }
            tri.z = (float(pick(81)) - 8.0f) / 64.0f;  // -0.125 .. 1.125: some outside the depth range
            for (float& v : tri.color) v = color_value(rng);
            draw.triangles.push_back(tri);
        }
        p.draws.push_back(draw);
    }
    return p;
}

const char* blend_name(D3D12_BLEND b) {
    static const char* names[] = {"?", "ZERO", "ONE", "SRC_COLOR", "INV_SRC_COLOR", "SRC_ALPHA", "INV_SRC_ALPHA",
                                  "DEST_ALPHA", "INV_DEST_ALPHA", "DEST_COLOR", "INV_DEST_COLOR", "SRC_ALPHA_SAT", "?",
                                  "?", "BLEND_FACTOR", "INV_BLEND_FACTOR"};
    return UINT(b) < std::size(names) ? names[b] : "?";
}

void print_program(const Program& p) {
    std::printf("  colour %s%s, depth %s; clear (%g %g %g %g), depth %g, stencil %u\n", p.color->name,
                p.blendable ? "" : " (not blendable)", p.depth->name, p.clear_color[0], p.clear_color[1],
                p.clear_color[2], p.clear_color[3], p.clear_depth, p.clear_stencil);
    if (p.color2)
        std::printf("  second target %s%s, clear (%g %g %g %g)\n", p.color2->name, p.blendable2 ? "" : " (not blendable)",
                    p.clear_color2[0], p.clear_color2[1], p.clear_color2[2], p.clear_color2[3]);
    static const char* ops[] = {"?", "ADD", "SUB", "REV_SUB", "MIN", "MAX"};
    for (size_t i = 0; i < p.draws.size(); ++i) {
        const Draw& d = p.draws[i];
        const D3D12_RENDER_TARGET_BLEND_DESC& b = d.blend.RenderTarget[0];
        std::printf("  draw %zu: blend %s", i, b.BlendEnable ? "on" : "off");
        if (b.LogicOpEnable) std::printf(" logic op %d", int(b.LogicOp));
        if (b.BlendEnable)
            std::printf(" (%s*src %s %s*dst; alpha %s %s %s)", blend_name(b.SrcBlend), ops[b.BlendOp],
                        blend_name(b.DestBlend), blend_name(b.SrcBlendAlpha), ops[b.BlendOpAlpha],
                        blend_name(b.DestBlendAlpha));
        std::printf(" mask %x factor (%g %g %g %g)", b.RenderTargetWriteMask, d.blend_factor[0], d.blend_factor[1],
                    d.blend_factor[2], d.blend_factor[3]);
        if (d.blend.IndependentBlendEnable) {
            const D3D12_RENDER_TARGET_BLEND_DESC& b1 = d.blend.RenderTarget[1];
            std::printf("; target 1 blend %s", b1.BlendEnable ? "on" : "off");
            if (b1.BlendEnable)
                std::printf(" (%s*src %s %s*dst; alpha %s %s %s)", blend_name(b1.SrcBlend), ops[b1.BlendOp],
                            blend_name(b1.DestBlend), blend_name(b1.SrcBlendAlpha), ops[b1.BlendOpAlpha],
                            blend_name(b1.DestBlendAlpha));
            std::printf(" mask %x", b1.RenderTargetWriteMask);
        }
        const D3D12_DEPTH_STENCIL_DESC& s = d.depth;
        std::printf("; depth %s func %d write %d", s.DepthEnable ? "on" : "off", int(s.DepthFunc),
                    int(s.DepthWriteMask));
        if (s.StencilEnable)
            std::printf("; stencil ref %u r%02x w%02x front %d/%d/%d f%d back %d/%d/%d f%d", d.stencil_ref,
                        s.StencilReadMask, s.StencilWriteMask, s.FrontFace.StencilFailOp, s.FrontFace.StencilDepthFailOp,
                        s.FrontFace.StencilPassOp, s.FrontFace.StencilFunc, s.BackFace.StencilFailOp,
                        s.BackFace.StencilDepthFailOp, s.BackFace.StencilPassOp, s.BackFace.StencilFunc);
        std::printf("; cull %d %s bias %d clip %d; viewport %g,%g %gx%g depth %g..%g; scissor %ld,%ld-%ld,%ld\n",
                    int(d.raster.CullMode), d.raster.FrontCounterClockwise ? "ccw" : "cw", d.raster.DepthBias,
                    d.raster.DepthClipEnable, d.viewport.TopLeftX, d.viewport.TopLeftY, d.viewport.Width,
                    d.viewport.Height, d.viewport.MinDepth, d.viewport.MaxDepth, d.scissor.left, d.scissor.top,
                    d.scissor.right, d.scissor.bottom);
        for (const Triangle& t : d.triangles)
            std::printf("    tri (%d,%d) (%d,%d) (%d,%d) z %g colour (%g %g %g %g)\n", t.px[0], t.py[0], t.px[1],
                        t.py[1], t.px[2], t.py[2], t.z, t.color[0], t.color[1], t.color[2], t.color[3]);
    }
}

// The pipeline's shaders on one device: root signature, VS and PS bytecode.
struct Pass {
    const std::vector<std::uint8_t>* root;
    const std::vector<std::uint8_t>* vs;
    const std::vector<std::uint8_t>* ps;
    const std::vector<std::uint8_t>* ps_mrt;   // two targets
    const std::vector<std::uint8_t>* ps_uint;  // a UINT target
};

// A plane of the readback, rows without padding.
struct Plane {
    UINT plane;
    DXGI_FORMAT format;
    size_t start;
    UINT64 row_bytes;
    UINT rows;
};

struct Result {
    std::vector<std::uint8_t> color, color2;
    std::vector<std::uint8_t> depth;  // every plane, one after the other
    std::vector<Plane> depth_planes;
    std::string error;
};

D3D12_RESOURCE_DESC texture(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = kSize;
    d.Height = kSize;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc = {1, 0};
    d.Flags = flags;
    return d;
}

Result run(ID3D12Device* device, const Program& p, const Pass& pass) {
    Result out;
    const auto fail = [&](const char* why) {
        out.error = why;
        return out;
    };
    Ref<ID3D12RootSignature> root;
    if (FAILED(device->CreateRootSignature(0, pass.root->data(), pass.root->size(), IID_PPV_ARGS(&root.p))))
        return fail("root signature");
    const bool has_depth = p.depth->format != DXGI_FORMAT_UNKNOWN;
    std::vector<Ref<ID3D12PipelineState>> psos(p.draws.size());
    const D3D12_INPUT_ELEMENT_DESC elements[2] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
    for (size_t i = 0; i < p.draws.size(); ++i) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root.p;
        desc.VS = {pass.vs->data(), pass.vs->size()};
        const std::vector<std::uint8_t>& ps =
            p.color->kind == ColorFormat::uint ? *pass.ps_uint : p.color2 ? *pass.ps_mrt : *pass.ps;
        desc.PS = {ps.data(), ps.size()};
        desc.BlendState = p.draws[i].blend;
        desc.SampleMask = 0xffffffff;
        desc.RasterizerState = p.draws[i].raster;
        desc.DepthStencilState = p.draws[i].depth;
        desc.InputLayout = {elements, 2};
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = p.color2 ? 2 : 1;
        desc.RTVFormats[0] = p.color->format;
        if (p.color2) desc.RTVFormats[1] = p.color2->format;
        desc.DSVFormat = p.depth->format;
        desc.SampleDesc = {1, 0};
        if (FAILED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&psos[i].p)))) return fail("PSO refused");
    }
    D3D12_HEAP_PROPERTIES def{D3D12_HEAP_TYPE_DEFAULT}, up{D3D12_HEAP_TYPE_UPLOAD}, rb{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> color, color2, depth, vertices, readback;
    const auto cd = texture(p.color->format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &cd, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                               nullptr, IID_PPV_ARGS(&color.p))))
        return fail("colour target");
    const auto cd2 = texture(p.color2 ? p.color2->format : p.color->format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    if (p.color2 && FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &cd2,
                                                           D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                           IID_PPV_ARGS(&color2.p))))
        return fail("second colour target");
    if (has_depth) {
        const auto dd = texture(p.depth->format, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                   nullptr, IID_PPV_ARGS(&depth.p))))
            return fail("depth target");
    }
    std::vector<Vertex> verts;
    for (const Draw& d : p.draws)
        for (const Triangle& t : d.triangles)
            for (int v = 0; v < 3; ++v)
                verts.push_back({float(t.px[v]) / (kSize / 2.0f) - 1.0f, 1.0f - float(t.py[v]) / (kSize / 2.0f), t.z,
                                 1.0f, t.color[0], t.color[1], t.color[2], t.color[3]});
    const UINT64 vb_bytes = std::max<UINT64>(sizeof(Vertex), verts.size() * sizeof(Vertex));
    const auto vd = d3d12test::buffer_desc(vb_bytes);
    if (FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &vd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                               nullptr, IID_PPV_ARGS(&vertices.p))))
        return fail("vertex buffer");
    void* mapped = nullptr;
    if (FAILED(vertices->Map(0, nullptr, &mapped))) return fail("map vertices");
    std::memcpy(mapped, verts.data(), verts.size() * sizeof(Vertex));
    vertices->Unmap(0, nullptr);
    const auto rbd = d3d12test::buffer_desc(1 << 20);
    if (FAILED(device->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &rbd, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&readback.p))))
        return fail("readback");
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2};
    Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap.p)))) return fail("rtv heap");
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv_heap.p)))) return fail("dsv heap");
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE rtv2 = rtv;
    rtv2.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(color.p, nullptr, rtv);
    if (p.color2) device->CreateRenderTargetView(color2.p, nullptr, rtv2);
    if (has_depth) device->CreateDepthStencilView(depth.p, nullptr, dsv);

    D3D12_COMMAND_QUEUE_DESC qd{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return fail("queue");
    list->ClearRenderTargetView(rtv, p.clear_color, 0, nullptr);
    if (p.color2) list->ClearRenderTargetView(rtv2, p.clear_color2, 0, nullptr);
    if (has_depth)
        list->ClearDepthStencilView(dsv, p.depth->stencil ? D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL
                                                          : D3D12_CLEAR_FLAG_DEPTH,
                                    p.clear_depth, p.clear_stencil, 0, nullptr);
    list->SetGraphicsRootSignature(root.p);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2] = {rtv, rtv2};
    list->OMSetRenderTargets(p.color2 ? 2 : 1, rtvs, FALSE, has_depth ? &dsv : nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VERTEX_BUFFER_VIEW vbv{vertices->GetGPUVirtualAddress(), UINT(vb_bytes), sizeof(Vertex)};
    list->IASetVertexBuffers(0, 1, &vbv);
    UINT first = 0;
    for (size_t i = 0; i < p.draws.size(); ++i) {
        const Draw& d = p.draws[i];
        list->SetPipelineState(psos[i].p);
        list->RSSetViewports(1, &d.viewport);
        list->RSSetScissorRects(1, &d.scissor);
        list->OMSetBlendFactor(d.blend_factor);
        list->OMSetStencilRef(d.stencil_ref);
        const UINT count = UINT(d.triangles.size()) * 3;
        list->DrawInstanced(count, 1, first, 0);
        first += count;
    }
    // Read back the colour target and every depth-stencil plane.
    D3D12_RESOURCE_BARRIER barriers[3] = {
        d3d12test::transition(color.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    UINT nb = 1;
    if (p.color2)
        barriers[nb++] = d3d12test::transition(color2.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (has_depth)
        barriers[nb++] = d3d12test::transition(depth.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(nb, barriers);
    struct Region {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
        UINT rows;
        UINT64 row_bytes;
    };
    std::vector<Region> regions;
    UINT64 offset = 0;
    const auto copy_out = [&](ID3D12Resource* r, const D3D12_RESOURCE_DESC& desc, UINT subresource) {
        Region rg{};
        device->GetCopyableFootprints(&desc, subresource, 1, offset, &rg.fp, &rg.rows, &rg.row_bytes, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = rg.fp;
        D3D12_TEXTURE_COPY_LOCATION src{r, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.SubresourceIndex = subresource;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        regions.push_back(rg);
        offset = (rg.fp.Offset + UINT64(rg.rows) * rg.fp.Footprint.RowPitch + 511) / 512 * 512;
    };
    copy_out(color.p, cd, 0);
    if (p.color2) copy_out(color2.p, cd2, 0);
    const UINT planes = has_depth ? (p.depth->stencil ? 2u : 1u) : 0u;
    const auto dd = texture(p.depth->format, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    for (UINT plane = 0; plane < planes; ++plane) copy_out(depth.p, dd, plane);
    if (FAILED(list->Close())) return fail("Close");
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.p, 1);
    HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(1, e);
    const bool done = WaitForSingleObject(e, 20000) == WAIT_OBJECT_0;
    CloseHandle(e);
    if (!done) return fail("timed out");
    std::uint8_t* back = nullptr;
    const D3D12_RANGE range{0, SIZE_T(1 << 20)};
    if (FAILED(readback->Map(0, &range, reinterpret_cast<void**>(&back)))) return fail("map readback");
    const size_t colours = p.color2 ? 2 : 1;
    for (size_t i = 0; i < regions.size(); ++i) {
        const Region& rg = regions[i];
        std::vector<std::uint8_t>& to = i == 0 ? out.color : i < colours ? out.color2 : out.depth;
        const size_t start = to.size();
        for (UINT row = 0; row < rg.rows; ++row) {
            const std::uint8_t* from = back + rg.fp.Offset + UINT64(row) * rg.fp.Footprint.RowPitch;
            to.insert(to.end(), from, from + rg.row_bytes);
        }
        if (i >= colours)
            out.depth_planes.push_back({UINT(i - colours), rg.fp.Footprint.Format, start, rg.row_bytes, rg.rows});
    }
    readback->Unmap(0, nullptr);
    return out;
}

std::uint32_t load(const std::vector<std::uint8_t>& v, size_t at, size_t bytes) {
    std::uint32_t x = 0;
    std::memcpy(&x, v.data() + at, bytes);
    return x;
}

// Ordered distance between two floats' bit patterns (ulps), or a huge number when exactly one is a NaN.
std::uint64_t ulps32(std::uint32_t a, std::uint32_t b) {
    const auto key = [](std::uint32_t f) -> std::int64_t {
        return (f & 0x80000000u) ? -std::int64_t(f & 0x7fffffffu) : std::int64_t(f);
    };
    const bool na = (a & 0x7f800000u) == 0x7f800000u && (a & 0x7fffffu);
    const bool nb = (b & 0x7f800000u) == 0x7f800000u && (b & 0x7fffffu);
    if (na || nb) return na && nb ? 0 : ~0ull;
    const std::int64_t d = key(a) - key(b);
    return std::uint64_t(d < 0 ? -d : d);
}
std::uint64_t ulps16(std::uint32_t a, std::uint32_t b) {
    const auto key = [](std::uint32_t h) -> std::int64_t { return (h & 0x8000u) ? -std::int64_t(h & 0x7fffu) : std::int64_t(h); };
    const bool na = (a & 0x7c00u) == 0x7c00u && (a & 0x3ffu), nb = (b & 0x7c00u) == 0x7c00u && (b & 0x3ffu);
    if (na || nb) return na && nb ? 0 : ~0ull;
    const std::int64_t d = key(a) - key(b);
    return std::uint64_t(d < 0 ? -d : d);
}

struct Comparison {
    size_t exact_differences = 0;  // texels not bit-identical
    size_t beyond_tolerance = 0;   // of those, the ones past the tolerance
    std::string first;             // the first texel past it
};

void note(Comparison& c, const char* what, UINT x, UINT y, std::uint64_t mine, std::uint64_t ref) {
    if (c.beyond_tolerance++ == 0) {
        char text[200];
        std::snprintf(text, sizeof text, "%s at (%u,%u): layer %llx, reference %llx", what, x, y,
                      static_cast<unsigned long long>(mine), static_cast<unsigned long long>(ref));
        c.first = text;
    }
}

Comparison compare(const Program& p, const Result& layer, const Result& ref) {
    Comparison c;
    if (layer.color.size() != ref.color.size() || layer.color2.size() != ref.color2.size() ||
        layer.depth.size() != ref.depth.size()) {
        c.beyond_tolerance = 1;
        c.first = "readback sizes differ";
        return c;
    }
    const auto target = [&](const ColorFormat& format, const std::vector<std::uint8_t>& mine_bytes,
                            const std::vector<std::uint8_t>& ref_bytes, const char* label) {
    const UINT bytes = format.bytes;
    const size_t row_bytes = size_t(kSize) * bytes;
    for (UINT y = 0; y < kSize; ++y)
        for (UINT x = 0; x < kSize; ++x) {
            const size_t at = y * row_bytes + size_t(x) * bytes;
            if (!std::memcmp(mine_bytes.data() + at, ref_bytes.data() + at, bytes)) continue;
            ++c.exact_differences;
            bool within = true;
            switch (format.kind) {
            case ColorFormat::unorm8:
                for (UINT k = 0; k < 4; ++k) within = within && std::abs(int(mine_bytes[at + k]) - int(ref_bytes[at + k])) <= 1;
                break;
            case ColorFormat::unorm10_2: {
                const std::uint32_t a = load(mine_bytes, at, 4), b = load(ref_bytes, at, 4);
                for (UINT k = 0; k < 3; ++k)
                    within = within && std::abs(int((a >> (10 * k)) & 0x3ff) - int((b >> (10 * k)) & 0x3ff)) <= 1;
                within = within && (a >> 30) == (b >> 30);  // a 2-bit alpha has no rounding slack
                break;
            }
            case ColorFormat::half:
                for (UINT k = 0; k < 4; ++k)
                    within = within && ulps16(load(mine_bytes, at + 2 * k, 2), load(ref_bytes, at + 2 * k, 2)) <= 1;
                break;
            case ColorFormat::single:
                for (UINT k = 0; k < 4; ++k)
                    within = within && ulps32(load(mine_bytes, at + 4 * k, 4), load(ref_bytes, at + 4 * k, 4)) <= 2;
                break;
            case ColorFormat::uint:
                within = false;  // integers write exactly
                break;
            case ColorFormat::r11g11b10: {
                const std::uint32_t a = load(mine_bytes, at, 4), b = load(ref_bytes, at, 4);
                constexpr UINT shift[3] = {0, 11, 22}, bits[3] = {11, 11, 10};
                for (UINT k = 0; k < 3; ++k) {
                    const int da = int((a >> shift[k]) & ((1u << bits[k]) - 1)), db = int((b >> shift[k]) & ((1u << bits[k]) - 1));
                    within = within && std::abs(da - db) <= 1;
                }
                break;
            }
            }
            if (!within) {
                std::uint64_t mine = 0, theirs = 0;
                std::memcpy(&mine, mine_bytes.data() + at, std::min<UINT>(bytes, 8));
                std::memcpy(&theirs, ref_bytes.data() + at, std::min<UINT>(bytes, 8));
                note(c, label, x, y, mine, theirs);
            }
        }
    };
    target(*p.color, layer.color, ref.color, "colour");
    if (p.color2) target(*p.color2, layer.color2, ref.color2, "target 1");
    for (const Plane& pl : ref.depth_planes) {
        const UINT texel = UINT(pl.row_bytes / kSize);
        const bool stencil_plane = pl.plane == 1;
        for (UINT y = 0; y < pl.rows; ++y)
            for (UINT x = 0; x < kSize; ++x) {
                const size_t at = pl.start + y * pl.row_bytes + size_t(x) * texel;
                std::uint32_t a = load(layer.depth, at, std::min<UINT>(texel, 4));
                std::uint32_t b = load(ref.depth, at, std::min<UINT>(texel, 4));
                if (stencil_plane) {
                    a &= 0xff;
                    b &= 0xff;
                } else if (p.depth->format == DXGI_FORMAT_D24_UNORM_S8_UINT) {
                    a &= 0xffffff;  // the X8 byte of the depth plane is undefined
                    b &= 0xffffff;
                }
                if (a == b) continue;
                ++c.exact_differences;
                bool within;
                if (stencil_plane) within = false;
                else if (p.depth->format == DXGI_FORMAT_D16_UNORM || p.depth->format == DXGI_FORMAT_D24_UNORM_S8_UINT)
                    within = std::abs(std::int64_t(a) - std::int64_t(b)) <= 1;
                else
                    within = ulps32(a, b) <= 2;
                if (!within) note(c, stencil_plane ? "stencil" : "depth", x, y, a, b);
            }
    }
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    unsigned seeds = 40, start = 1, reduce = 0;
    bool verbose = false, warp = false, reducing = false;
    for (int i = 1; i < argc; ++i) {
        bool bad = false;
        const auto number = [&](unsigned& out) {
            const char* s = i + 1 < argc ? argv[i + 1] : "";
            char* end = nullptr;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (*s < '0' || *s > '9' || *end || v > 0xffffffffull) {
                bad = true;
                return;
            }
            out = unsigned(v);
            ++i;
        };
        if (!std::strcmp(argv[i], "--seeds")) number(seeds);
        else if (!std::strcmp(argv[i], "--start")) number(start);
        else if (!std::strcmp(argv[i], "--reduce")) { number(reduce); reducing = true; }
        else if (!std::strcmp(argv[i], "--warp")) warp = true;
        else if (!std::strcmp(argv[i], "--all")) with_depth_bias = true;
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
        if (bad) {
            std::fprintf(stderr, "%s needs a decimal count\n", argv[i]);
            return 2;
        }
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (hostwine::host_is_wine()) {
        std::puts("SKIP: Wine's D3D12 is not a reference");
        return 77;
    }
    // The layer's shaders: containers around synthetic FSSM programs, and the index that maps their hashes to the
    // compiled GLSL stand-ins.
    const auto vs_program = d3d12test::fssm_program({0x56530501, 0xbf810000});
    const auto ps_program = d3d12test::fssm_program({0x50530501, 0xbf810000});
    const auto vs_container = d3d12test::shader_container(vs_program);
    const auto ps_container = d3d12test::shader_container(ps_program);
    const auto ps_mrt_program = d3d12test::fssm_program({0x50530502, 0xbf810000});
    const auto ps_mrt_container = d3d12test::shader_container(ps_mrt_program);
    const auto ps_uint_program = d3d12test::fssm_program({0x50530503, 0xbf810000});
    const auto ps_uint_container = d3d12test::shader_container(ps_uint_program);
    const testfiles::ScratchDir scratch("leonida_pso_fuzz_test");
    const auto& library = scratch.path();
    if (!testfiles::copy_file(LEONIDA_PSO_VS_SPV, library / "vs.spv") ||
        !testfiles::copy_file(LEONIDA_PSO_PS_SPV, library / "ps.spv") ||
        !testfiles::copy_file(LEONIDA_PSO_PS_MRT_SPV, library / "ps_mrt.spv") ||
        !testfiles::copy_file(LEONIDA_PSO_PS_UINT_SPV, library / "ps_uint.spv"))
        return 1;
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 1, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]},\n"
                 " \"ps_mrt\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_mrt_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0, 1]},\n"
                 " \"ps_uint\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_uint_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }
    // The device the runtime itself makes (its features, depthClamp among them), reading the shaders from
    // LEONIDA_SHADERS.
    if (_putenv_s("LEONIDA_SHADERS", library.string().c_str()) != 0) return 1;
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer.p)))) {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }

    // The reference: the host's D3D12 on the layer's own GPU, matched by LUID (the D3D-188 rule), so that only the
    // translation can differ; or WARP with --warp. A GPU's rasterizer and blender may differ from WARP's within D3D12's
    // rules (an edge through a pixel centre, a rounding), which the same GPU's D3D12 shares with its Vulkan driver.
    std::wstring ref_name;
    Ref<ID3D12Device> ref;
    {
        Ref<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            if (warp) {
                Ref<IDXGIAdapter> adapter;
                if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) &&
                    SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ref.p))))
                    ref_name = L"WARP";
            } else if (const auto* info = rt::d3d12::vulkan_device_info(layer.p)) {
                VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
                const auto get = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                    info->get_instance_proc(info->instance, "vkGetPhysicalDeviceProperties2"));
                if (get) get(info->physical, &properties);
                LUID luid{};
                std::memcpy(&luid, id.deviceLUID, sizeof luid);
                Ref<IDXGIAdapter1> adapter;
                DXGI_ADAPTER_DESC1 desc{};
                if (id.deviceLUIDValid && SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter.p))) &&
                    SUCCEEDED(adapter->GetDesc1(&desc)) &&
                    SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ref.p))))
                    ref_name = desc.Description;
            }
        }
    }
    if (!ref.p) {
        std::puts("SKIP: no reference D3D12 device");
        return 77;
    }
    const HMODULE compiler = LoadLibraryExA("d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(reinterpret_cast<void*>(
                                        GetProcAddress(compiler, "D3DCompile")))
                                  : nullptr;
    std::vector<std::uint8_t> host_vs, host_ps, host_ps_mrt, host_ps_uint;
    if (!compile || !compile_hlsl(compile, kHostVs, "vs_5_0", host_vs) || !compile_hlsl(compile, kHostPs, "ps_5_0", host_ps) ||
        !compile_hlsl(compile, kHostPsMrt, "ps_5_0", host_ps_mrt) ||
        !compile_hlsl(compile, kHostPsUint, "ps_5_0", host_ps_uint)) {
        std::puts("SKIP: d3dcompiler_47 could not compile the reference's shaders");
        return 77;
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const auto root_blob = d3d12test::serialize_root_signature(root_desc);
    if (root_blob.empty()) return 1;
    const Pass layer_pass{&root_blob, &vs_container, &ps_container, &ps_mrt_container, &ps_uint_container};
    const Pass ref_pass{&root_blob, &host_vs, &host_ps, &host_ps_mrt, &host_ps_uint};
    const Support sup = support(layer.p, ref.p);
    if (sup.colors.empty()) {
        std::puts("SKIP: no colour format both devices render");
        return 77;
    }
    if (verbose)
        std::printf("%zu colour and %zu depth formats on both devices; logic ops %s\n", sup.colors.size(),
                    sup.depths.size(), sup.logic_op ? "on both" : "not on both (UINT targets draw without them)");

    const auto differs = [&](const Program& q, std::uint64_t, Comparison& c) {
        const Result b = run(ref.p, q, ref_pass);
        if (!b.error.empty()) return false;
        const Result a = run(layer.p, q, layer_pass);
        if (!a.error.empty()) {
            c.beyond_tolerance = 1;
            c.first = "layer: " + a.error;
            return true;
        }
        c = compare(q, a, b);
        return c.beyond_tolerance != 0;
    };
    if (reducing) {
        Program p = generate(reduce, sup);
        Comparison c;
        if (!differs(p, reduce, c)) {
            std::printf("seed %u: the layer matches %ls, nothing to reduce\n", reduce, ref_name.c_str());
            return 0;
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t i = p.draws.size(); i-- > 0;) {
                Program q = p;
                q.draws.erase(q.draws.begin() + std::ptrdiff_t(i));
                if (!q.draws.empty() && differs(q, reduce, c)) {
                    p = q;
                    changed = true;
                }
            }
            for (size_t i = 0; i < p.draws.size(); ++i)
                for (size_t t = p.draws[i].triangles.size(); t-- > 0;) {
                    if (p.draws[i].triangles.size() == 1) break;
                    Program q = p;
                    q.draws[i].triangles.erase(q.draws[i].triangles.begin() + std::ptrdiff_t(t));
                    if (differs(q, reduce, c)) {
                        p = q;
                        changed = true;
                    }
                }
        }
        differs(p, reduce, c);
        std::printf("seed %u reduced to %zu draw(s), %zu texel(s) past the tolerance; first: %s\n", reduce,
                    p.draws.size(), c.beyond_tolerance, c.first.c_str());
        print_program(p);
        return 1;
    }

    unsigned ran = 0, differ = 0, refused = 0;
    size_t within = 0;
    for (unsigned k = 0; k < seeds; ++k) {
        const std::uint64_t seed = std::uint64_t(start) + k;
        const Program p = generate(seed, sup);
        const Result b = run(ref.p, p, ref_pass);
        if (!b.error.empty()) {
            if (verbose) std::printf("seed %llu: reference: %s\n", static_cast<unsigned long long>(seed), b.error.c_str());
            continue;
        }
        ++ran;
        const Result a = run(layer.p, p, layer_pass);
        if (!a.error.empty()) {
            ++refused;
            std::printf("seed %llu: layer: %s\n", static_cast<unsigned long long>(seed), a.error.c_str());
            if (verbose) print_program(p);
            continue;
        }
        const Comparison c = compare(p, a, b);
        within += c.exact_differences - c.beyond_tolerance;
        if (c.beyond_tolerance) {
            ++differ;
            std::printf("seed %llu: %zu texel(s) differ from %ls past the tolerance, first %s\n",
                        static_cast<unsigned long long>(seed), c.beyond_tolerance, ref_name.c_str(), c.first.c_str());
            if (verbose) print_program(p);
        } else if (verbose) {
            std::printf("seed %llu: same (%s%s%s, %s, %zu draws; %zu texels within the tolerance)\n",
                        static_cast<unsigned long long>(seed), p.color->name, p.color2 ? " + " : "",
                        p.color2 ? p.color2->name : "", p.depth->name, p.draws.size(), c.exact_differences);
        }
    }
    std::printf("pso fuzz (layer against %ls): %u seeds run, %u differ, %u refused by the layer; %zu texels within "
                "the rounding tolerance\n",
                ref_name.c_str(), ran, differ, refused, within);
    return differ || refused ? 1 : 0;
}
