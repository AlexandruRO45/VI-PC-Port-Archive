// The dxgi.dll surface the layer provides: one factory, one adapter describing the host GPU, and swapchains that
// present through the shared-device WSI presenter.
//
// The COM objects below follow the SDK's interface order through stubs generated at build time from dxgi.h and
// dxgi1_2.h (tools/gen_d3d12_stubs.py), so their vtables match what a game expects even where a method is only
// logged. The implementation is deliberately small: what a game needs to open a window and present a frame.
#pragma once

#include <windows.h>
#include <dxgi.h>
#include <dxgi1_2.h>

namespace rt::d3d12 {

// CreateDXGIFactory / CreateDXGIFactory1 / CreateDXGIFactory2 behind the exported functions below.
HRESULT create_dxgi_factory(REFIID riid, void** out);

// Keeps this file's shim table in the link and says so at run time. The table registers itself through a static
// initialiser, which the linker drops with the object unless something references it; gta6rt and the shim DLL call
// this once at startup, which is why the dxgi.dll imports of a game resolve to the functions below.
void link_dxgi_shims();

}  // namespace rt::d3d12

// The dxgi.dll exports a game imports. A game's import is bound to these through the shim registry
// (shims/registry.h); the PE harness test calls them directly.
extern "C" HRESULT WINAPI CreateDXGIFactory(REFIID riid, void** out);
extern "C" HRESULT WINAPI CreateDXGIFactory1(REFIID riid, void** out);
extern "C" HRESULT WINAPI CreateDXGIFactory2(UINT flags, REFIID riid, void** out);
