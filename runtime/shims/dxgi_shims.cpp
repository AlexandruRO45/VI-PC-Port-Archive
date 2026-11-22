// Registers the dxgi.dll shims. This file is compiled straight into gta6rt and into the shim DLL (not only into the
// static library), so the table's static initialiser is never dropped from the link by an unreferenced object, while
// its references keep shims/dxgi.cpp in the static library.

#include "dxgi.h"

#include "shims.h"

SHIM_TABLE(dxgi, {"dxgi.dll", "CreateDXGIFactory", reinterpret_cast<void*>(&CreateDXGIFactory)},
                 {"dxgi.dll", "CreateDXGIFactory1", reinterpret_cast<void*>(&CreateDXGIFactory1)},
                 {"dxgi.dll", "CreateDXGIFactory2", reinterpret_cast<void*>(&CreateDXGIFactory2)});

namespace rt::d3d12 {

void link_dxgi_shims() {}

}  // namespace rt::d3d12
