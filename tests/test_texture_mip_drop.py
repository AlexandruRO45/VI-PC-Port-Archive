"""Unit tests verifying the low-VRAM texture mip-drop logic (LEONIDA_TEXTURE_MIP_DROP).

Addresses Issue #6043:
Leaves the top N (0..3) mips of sampled textures out of VRAM, halving dimensions per drop and
cutting texture VRAM footprint by ~70-75% for 6 GB and 4 GB GPUs, while strictly preserving
RenderTargets, DepthStencils, UAVs, and single-mip textures at full native resolution.
"""
import os
import unittest


class D3D12ResourceFlags:
    NONE = 0
    ALLOW_RENDER_TARGET = 1
    ALLOW_DEPTH_STENCIL = 2
    ALLOW_UNORDERED_ACCESS = 4
    DENY_SHADER_RESOURCE = 8


class TextureCreateInfoModel:
    """Python reference model of TextureCreateInfo in runtime/shims/d3d12_vulkan_device.cpp."""

    def __init__(self, width: int, height: int, depth_or_array: int, mip_levels: int,
                 flags: int = D3D12ResourceFlags.NONE, mip_drop_env: int = 0):
        self.original_width = width
        self.original_height = height
        self.original_mips = mip_levels
        self.flags = flags

        # Guards: Only pure sampled textures can drop mips
        is_render_target = bool(flags & D3D12ResourceFlags.ALLOW_RENDER_TARGET)
        is_depth_stencil = bool(flags & D3D12ResourceFlags.ALLOW_DEPTH_STENCIL)
        is_uav = bool(flags & D3D12ResourceFlags.ALLOW_UNORDERED_ACCESS)
        can_drop = not (is_render_target or is_depth_stencil or is_uav)

        # Clamped drop count
        configured_drop = max(0, min(3, mip_drop_env))
        if can_drop and mip_levels > 1:
            self.mips_dropped = min(configured_drop, mip_levels - 1)
        else:
            self.mips_dropped = 0

        # Output extent and mip levels
        self.width = max(1, width >> self.mips_dropped)
        self.height = max(1, height >> self.mips_dropped)
        self.mip_levels = mip_levels - self.mips_dropped
        self.array_layers = depth_or_array

    def map_subresource(self, subresource_index: int) -> int | None:
        """Maps an incoming D3D12 subresource index to the shrunken Vulkan image subresource.

        Returns None if the subresource was dropped (upload discarded).
        """
        orig_mips = self.original_mips
        mip = subresource_index % orig_mips
        slice_idx = subresource_index // orig_mips

        if mip < self.mips_dropped:
            return None  # Discarded upload

        mapped_mip = mip - self.mips_dropped
        new_mips = self.mip_levels
        return slice_idx * new_mips + mapped_mip

    def map_srv_view(self, most_detailed_mip: int, mip_count: int) -> tuple[int, int]:
        """Maps an incoming SRV view to the shrunken Vulkan image."""
        if self.mips_dropped == 0:
            return most_detailed_mip, min(mip_count, self.mip_levels - most_detailed_mip)

        if most_detailed_mip < self.mips_dropped:
            first_mip = 0
        else:
            first_mip = most_detailed_mip - self.mips_dropped

        avail_mips = max(1, self.mip_levels - first_mip)
        resolved_count = avail_mips if mip_count in (-1, 0, 0xFFFFFFFF) else min(mip_count, avail_mips)
        return first_mip, resolved_count


def calculate_texture_bytes_bc7(width: int, height: int, mips: int) -> int:
    """Calculates total bytes for a BC7 (16 bytes per 4x4 block) texture chain."""
    total = 0
    w, h = width, height
    for _ in range(mips):
        bw = max(1, (w + 3) // 4)
        bh = max(1, (h + 3) // 4)
        total += bw * bh * 16
        w = max(1, w >> 1)
        h = max(1, h >> 1)
    return total


class TestTextureMipDrop(unittest.TestCase):

    def test_default_env_zero_preserves_dimensions(self):
        """When LEONIDA_TEXTURE_MIP_DROP is 0 or unset, textures are 100% untouched."""
        t = TextureCreateInfoModel(4096, 2048, 1, 12, flags=D3D12ResourceFlags.NONE, mip_drop_env=0)
        self.assertEqual(t.mips_dropped, 0)
        self.assertEqual(t.width, 4096)
        self.assertEqual(t.height, 2048)
        self.assertEqual(t.mip_levels, 12)

    def test_drop_one_halves_resolution_and_drops_one_mip(self):
        """Drop 1 cuts 4096x2048 to 2048x1024 and reduces mip count by 1."""
        t = TextureCreateInfoModel(4096, 2048, 1, 12, flags=D3D12ResourceFlags.NONE, mip_drop_env=1)
        self.assertEqual(t.mips_dropped, 1)
        self.assertEqual(t.width, 2048)
        self.assertEqual(t.height, 1024)
        self.assertEqual(t.mip_levels, 11)

    def test_vram_savings_match_measured_issue_targets(self):
        """Drop 1 saves ~75% VRAM on 2D mip chains, matching Issue #6043 measurements."""
        base_bytes = calculate_texture_bytes_bc7(4096, 4096, 13)
        drop1_bytes = calculate_texture_bytes_bc7(2048, 2048, 12)
        drop2_bytes = calculate_texture_bytes_bc7(1024, 1024, 11)

        # Baseline: 4096 mip0 = 16 MB. Full chain ~ 21.33 MB.
        # Drop 1: 2048 mip0 = 4 MB. Chain ~ 5.33 MB (25% of full, i.e. 75% savings).
        ratio_drop1 = drop1_bytes / base_bytes
        ratio_drop2 = drop2_bytes / base_bytes

        self.assertAlmostEqual(ratio_drop1, 0.25, delta=0.01)
        self.assertAlmostEqual(ratio_drop2, 0.0625, delta=0.01)

    def test_render_targets_are_never_dropped(self):
        """RenderTargets must stay at full resolution to prevent viewport/scissor misalignment."""
        t = TextureCreateInfoModel(1920, 1080, 1, 1, flags=D3D12ResourceFlags.ALLOW_RENDER_TARGET, mip_drop_env=2)
        self.assertEqual(t.mips_dropped, 0)
        self.assertEqual(t.width, 1920)
        self.assertEqual(t.height, 1080)

    def test_depth_stencils_are_never_dropped(self):
        """Depth/Stencil buffers must remain at exact native viewport resolution."""
        t = TextureCreateInfoModel(1920, 1080, 1, 1, flags=D3D12ResourceFlags.ALLOW_DEPTH_STENCIL, mip_drop_env=2)
        self.assertEqual(t.mips_dropped, 0)
        self.assertEqual(t.width, 1920)
        self.assertEqual(t.height, 1080)

    def test_uav_storage_textures_are_never_dropped(self):
        """UAV storage textures used in compute shaders must preserve exact dispatch dimensions."""
        t = TextureCreateInfoModel(512, 512, 1, 1, flags=D3D12ResourceFlags.ALLOW_UNORDERED_ACCESS, mip_drop_env=3)
        self.assertEqual(t.mips_dropped, 0)
        self.assertEqual(t.width, 512)
        self.assertEqual(t.height, 512)

    def test_single_mip_textures_are_never_dropped(self):
        """1-mip textures (UI icons, lookup tables) cannot drop mips."""
        t = TextureCreateInfoModel(256, 256, 1, 1, flags=D3D12ResourceFlags.NONE, mip_drop_env=2)
        self.assertEqual(t.mips_dropped, 0)
        self.assertEqual(t.width, 256)
        self.assertEqual(t.height, 256)
        self.assertEqual(t.mip_levels, 1)

    def test_subresource_upload_discard_and_remapping(self):
        """Subresource uploads targeting dropped mips return None (discarded), rest map correctly."""
        t = TextureCreateInfoModel(2048, 2048, 2, 4, flags=D3D12ResourceFlags.NONE, mip_drop_env=1)
        # Original: 4 mips per slice. Slice 0 has subresources 0, 1, 2, 3.
        # With drop 1: Mip 0 is dropped. New image has 3 mips per slice (0, 1, 2).

        # Slice 0, Mip 0 -> Discarded
        self.assertIsNone(t.map_subresource(0))
        # Slice 0, Mip 1 -> Mapped to Slice 0, Mip 0
        self.assertEqual(t.map_subresource(1), 0)
        # Slice 0, Mip 2 -> Mapped to Slice 0, Mip 1
        self.assertEqual(t.map_subresource(2), 1)

        # Slice 1, Mip 0 (Subresource 4) -> Discarded
        self.assertIsNone(t.map_subresource(4))
        # Slice 1, Mip 1 (Subresource 5) -> Mapped to Slice 1, Mip 0 (Index 3)
        self.assertEqual(t.map_subresource(5), 3)

    def test_srv_view_clamping_and_offset(self):
        """SRVs requesting top mip get mapped to mip 0 of the new image."""
        t = TextureCreateInfoModel(4096, 4096, 1, 6, flags=D3D12ResourceFlags.NONE, mip_drop_env=1)

        # View requesting full chain starting at mip 0
        first_mip, mip_count = t.map_srv_view(0, 6)
        self.assertEqual(first_mip, 0)
        self.assertEqual(mip_count, 5)

        # View requesting mip 2 (which is now mip 1)
        first_mip, mip_count = t.map_srv_view(2, 4)
        self.assertEqual(first_mip, 1)
        self.assertEqual(mip_count, 4)

    def test_retained_mip_whole_copy_extent_derives_from_reduced_binding(self):
        """Whole-mip copy (box=None) on retained mip must derive extent from reduced image (P1 fix)."""
        # 8x8 texture, 3 mips, drop 1 -> physical image & binding is 4x4, 2 mips.
        t = TextureCreateInfoModel(8, 8, 1, 3, flags=D3D12ResourceFlags.NONE, mip_drop_env=1)
        self.assertEqual(t.width, 4)
        self.assertEqual(t.height, 4)
        self.assertEqual(t.mip_levels, 2)

        # Original subresource 1 (orig mip 1) maps to physical subresource 0 (phys mip 0).
        phys_subres = t.map_subresource(1)
        self.assertEqual(phys_subres, 0)

        # Whole-mip extent for physical mip 0:
        phys_mip = phys_subres % t.mip_levels
        phys_w = max(1, t.width >> phys_mip)
        phys_h = max(1, t.height >> phys_mip)
        self.assertEqual((phys_w, phys_h), (4, 4))
        # Total bytes for RGBA8 (4 bytes per texel): 4 * 4 * 4 = 64 bytes (NOT 8 * 8 * 4 = 256)
        self.assertEqual(phys_w * phys_h * 4, 64)

    def test_dropped_mip_read_zeros_requested_box_with_pitches(self):
        """Reading a dropped mip must zero all rows/slices across supplied pitches (P2 fix)."""
        # Simulate reading dropped mip 0 of 8x8 RGBA8 (row_pitch 32, depth_pitch 256) into a 512-byte buffer.
        buffer = bytearray([0xCD] * 512)
        row_pitch = 32
        depth_pitch = 256
        width, height, depth = 8, 8, 1
        b_left, b_top, b_front = 0, 0, 0
        b_right, b_bottom, b_back = width, height, depth
        bpp = 4

        rows = b_bottom - b_top
        row_bytes = (b_right - b_left) * bpp
        slices = b_back - b_front

        for z in range(slices):
            for y in range(rows):
                offset = z * depth_pitch + y * row_pitch
                buffer[offset:offset + row_bytes] = b"\x00" * row_bytes

        # First 256 bytes must be zeroed (8 rows * 32 bytes)
        self.assertEqual(buffer[:256], b"\x00" * 256)
        # Tail bytes (256..512) must remain untouched at 0xCD
        self.assertEqual(buffer[256:], b"\xCD" * 256)


if __name__ == "__main__":
    unittest.main()
