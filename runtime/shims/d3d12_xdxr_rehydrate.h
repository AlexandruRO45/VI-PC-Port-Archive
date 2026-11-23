// A private interface of the layer's command lists (d3d12_com.cpp): XboxBvhBuilder.dll's rehydration of a
// dehydrated XDXR BLAS, recorded into the list (src/xdxr_rehydrate.h; docs/bvh.md §4a). The XboxBvhBuilder shim
// (xbox_bvh_builder.cpp) asks a list for it with QueryInterface, so it needs no link to the layer: a list that isn't
// the layer's (the Windows D3D12 backend) answers E_NOINTERFACE.
#pragma once

#include <windows.h>

#include <d3d12.h>

// {a5c1e0d2-6b3f-4e8a-9d17-3c2b8f604e91}
inline constexpr GUID IID_ILeonidaXdxrRehydrate{0xa5c1e0d2, 0x6b3f, 0x4e8a,
                                                {0x9d, 0x17, 0x3c, 0x2b, 0x8f, 0x60, 0x4e, 0x91}};

struct ILeonidaXdxrRehydrate : IUnknown {
    // One structure, as RehydrateRaytracingAccelerationStructureX takes it:
    //   SourceAccelerationStructureData   the dehydrated structure
    //   DestAccelerationStructureData     the rehydrated structure (the same size; a different buffer)
    //   ScratchAccelerationStructureData  room for the GPU-written plan (XdxrRehydrator::plan_bytes)
    //   Inputs                            a bottom level, with one triangle geometry per geometry index the leaves use
    // E_INVALIDARG for a description it can't record; the list itself stays usable.
    virtual HRESULT STDMETHODCALLTYPE RehydrateRaytracingAccelerationStructure(
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc) = 0;
};
