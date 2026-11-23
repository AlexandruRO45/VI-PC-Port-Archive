// Lanterns stage 6: a compressed texture with its whole mip chain, created on the device and uploaded the way the
// D3D12 API offers it: GetCopyableFootprints, an upload buffer, one CopyTextureRegion per mip level. The layer lays the
// texels out; nothing here knows or writes the console's tiled layout (a title that wrote BC blocks straight into a
// texture's memory would need that layout, and Lanterns does not do it).
#pragma once

#include "lanterns_bc.h"
#include "lanterns_gpu.h"

namespace lanterns {

// Creates the texture (DEFAULT heap), uploads every level on queue, waits, and leaves it in PIXEL_SHADER_RESOURCE.
bool upload_texture(ID3D12Device* device, ID3D12CommandQueue* queue, const BcTexture& tex, Ref<ID3D12Resource>& out);

}  // namespace lanterns
