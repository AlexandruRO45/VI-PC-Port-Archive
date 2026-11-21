#include "dxr_fixture.h"
#include "dxr_opacity_reference.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static_assert(dxr_opacity::forceOpaque == D3D12_RAY_FLAG_FORCE_OPAQUE);
static_assert(dxr_opacity::forceNonOpaque == D3D12_RAY_FLAG_FORCE_NON_OPAQUE);
static_assert(dxr_opacity::acceptFirst == D3D12_RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH);
static_assert(dxr_opacity::cullOpaque == D3D12_RAY_FLAG_CULL_OPAQUE);
static_assert(dxr_opacity::cullNonOpaque == D3D12_RAY_FLAG_CULL_NON_OPAQUE);
static_assert(dxr_opacity::geometryOpaque == D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE);
static_assert(dxr_opacity::instanceForceOpaque == D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE);
static_assert(dxr_opacity::instanceForceNonOpaque == D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE);

bool check_dxr_opacity(ID3D12Device* device) {
    // The geometric reference must reproduce the hand-derived table on every
    // machine, including those that skip the GPU comparison.
    const auto scene = dxr_opacity::instances();
    const auto probes = dxr_opacity::rays();
    const auto mesh = dxr_opacity::vertices();
    const auto flags = dxr_opacity::geometryFlags();
    if (dxr_opacity::first_reference_mismatch() != probes.size())
        throw std::runtime_error("Opacity reference disagrees with its hand-derived table");
    if (!dxr_fixture::supported(device)) return false;
    std::vector<dxr_fixture::Vertex> vertices(mesh.begin(), mesh.end());
    std::vector<D3D12_RAYTRACING_GEOMETRY_FLAGS> geometries;
    for (const auto value : flags) geometries.push_back(static_cast<D3D12_RAYTRACING_GEOMETRY_FLAGS>(value));
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances;
    for (const auto& source : scene) {
        auto desc = dxr_fixture::instance(source.id, 255, source.translateX);
        desc.Flags = source.flags;
        instances.push_back(desc);
    }
    std::vector<dxr_fixture::Ray> rays;
    for (const auto& probe : probes)
        rays.push_back({probe.origin[0], probe.origin[1], probe.origin[2], probe.nearT,
                        probe.direction[0], probe.direction[1], probe.direction[2], probe.farT, 255, probe.flags});
    const auto words = dxr_fixture::trace(device, vertices, instances, rays, nullptr, &geometries);
    UINT hits = 0, shaderCommits = 0, firstHitRays = 0, mismatches = 0;
    for (UINT i = 0; i < rays.size(); ++i) {
        const auto expected = dxr_opacity::evaluate(mesh, flags, scene, probes[i]);
        const bool firstHit = (probes[i].flags & dxr_opacity::acceptFirst) != 0;
        const UINT* value = words.data() + 4 * i;
        const bool hit = value[0] & 1, afterCommit = value[0] >> 1 & 1, anomaly = value[0] >> 2 & 1;
        const UINT seen = value[0] >> 8, geometry = value[2] & 0xffff, primitive = value[2] >> 16;
        float distance;
        std::memcpy(&distance, value + 3, sizeof(distance));
        // A closest-hit search has one answer. A first-hit search may end on
        // any committable hit, but must stop presenting candidates after it.
        bool ok = !anomaly && (value[0] & 0xfffff0f8u) == 0 && hit == expected.closest.hit;
        if (hit && ok) {
            const bool allowed = firstHit ? (expected.committable >> geometry & 1) != 0
                                          : geometry == expected.closest.geometry;
            ok = allowed && geometry < dxr_opacity::geometryCount && primitive == 0 &&
                 value[1] == expected.closest.instance &&
                 std::abs(distance - expected.distance[geometry]) <= 1e-5 * expected.distance[geometry];
        } else if (ok) {
            ok = value[1] == 0 && value[2] == 0 && value[3] == 0;
        }
        UINT required = expected.required;
        if (hit && geometry < dxr_opacity::geometryCount && (expected.nonOpaque >> geometry & 1)) required |= 1u << geometry;
        ok = ok && (seen & required) == required && (seen & ~expected.nonOpaque) == 0 && !(firstHit && afterCommit);
        if (!ok) {
            if (!mismatches)
                std::fprintf(stderr, "ray=%u flags=0x%x expected hit=%u instance=%u geometry=%u distance=%g "
                             "committable=0x%x candidates required=0x%x allowed=0x%x actual word0=0x%x instance=%u "
                             "geometry=%u primitive=%u distance=%g\n", i, probes[i].flags, expected.closest.hit,
                             expected.closest.instance, expected.closest.geometry, expected.closest.distance,
                             expected.committable, required, expected.nonOpaque, value[0], value[1], geometry,
                             primitive, distance);
            ++mismatches;
        }
        hits += expected.closest.hit;
        shaderCommits += expected.closest.hit && (expected.nonOpaque >> expected.closest.geometry & 1);
        firstHitRays += firstHit;
    }
    std::printf("dxr_opacity rays=%zu hits=%u shader_commits=%u misses=%zu first_hit_rays=%u mismatches=%u\n",
                rays.size(), hits, shaderCommits, rays.size() - hits, firstHitRays, mismatches);
    if (mismatches) throw std::runtime_error("DXR opacity or candidate handling disagrees with the CPU reference");
    if (hits != 94 || shaderCommits != 42 || firstHitRays != 36)
        throw std::runtime_error("Opacity fixture lost its expected coverage");
    return true;
}
