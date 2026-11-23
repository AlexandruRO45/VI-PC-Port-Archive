// Interface identifiers exported by the ERA d3d12_xs.dll (@23–@55).
// Games resolve these by name for QueryInterface and device creation.
// Values from the ERA d3d12_xs.dll PDB (GDK 261000 gameKit/symbols).
// Standard D3D12 and DXGI GUIDs match the desktop Windows SDK; the two
// Xbox-specific ones (ID3D12XboxDmaCommandList and IGraphicsUnknown)
// are defined here from the ERA binary.
#include <initguid.h>
#include <d3d12.h>
#include <dxgi1_6.h>

// Xbox-specific interfaces not in the desktop SDK.
// {30BC77B9-8379-4B25-95DE-B8588C42275D}
DEFINE_GUID(IID_ID3D12XboxDmaCommandList,
    0x30BC77B9, 0x8379, 0x4B25, 0x95, 0xDE, 0xB8, 0x58, 0x8C, 0x42, 0x27, 0x5D);

// {ACEEEA63-E0A9-4A1C-BBEC-71B2F485F758}
DEFINE_GUID(IID_IGraphicsUnknown,
    0xACEEEA63, 0xE0A9, 0x4A1C, 0xBB, 0xEC, 0x71, 0xB2, 0xF4, 0x85, 0xF7, 0x58);
