"""Semantic interpreter tests for flat scratch memory pointer generation and per-wave stack frames.

RDNA2 (GFX10) flat scratch memory layout and pointer semantics:
  * Scratch memory provides thread-private storage used for spill/reload and stack frames.
  * Backing memory is organized per-wave and interleaved (swizzled) at 4-byte (dword) granularity:
      physical_address = wave_base + ((dword_index * wavefront_size) + lane_id) * 4 + byte_in_dword
    where dword_index = offset // 4, byte_in_dword = offset % 4.
  * scratch_load_dword supports multiple addressing modes:
      - Mode A: Scalar stack pointer (saddr) + immediate offset (vaddr = off)
      - Mode B: Vector offset (vaddr) + immediate offset (saddr = off)
      - Mode C: Vector offset (vaddr) + scalar base (saddr) + immediate offset
      - Mode D: Immediate-only offset (vaddr = off, saddr = off)
  * Immediate offsets can be positive (local spill slots) or negative (caller frame references).
  * Flat pointers falling within the private scratch aperture translate to wave scratch addresses.
  * Vector operations respect the execution mask (EXEC), updating only active lanes.
  * Misaligned accesses and out-of-bounds frame accesses are trapped.

Run locally:
    python -I tests/semantic/test_flat_scratch.py
    python -m unittest discover -s tests/semantic -v
"""
import struct
import sys
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


class ScratchFault(Exception):
    """Base exception for scratch memory faults."""


class AlignmentFault(ScratchFault):
    """Raised when a dword memory access is not 4-byte aligned."""


class OutOfBoundsFault(ScratchFault):
    """Raised when an access exceeds the allocated per-wave scratch bounds."""


class UnderflowFault(ScratchFault):
    """Raised when a negative offset underflows below the wave scratch base."""


class InvalidApertureFault(ScratchFault):
    """Raised when a flat pointer does not fall within the scratch aperture."""


def compute_scratch_address(wave_base, lane_id, offset, wavefront_size=32):
    """Calculate the physical byte address for a given wave, lane, and byte offset in scratch memory."""
    if offset < 0:
        raise UnderflowFault(f"negative offset {offset} underflows scratch frame")
    dword_index = offset // 4
    byte_in_dword = offset % 4
    return wave_base + ((dword_index * wavefront_size) + lane_id) * 4 + byte_in_dword


class ScratchMemory:
    """Simulated backing store for scratch memory allocations across waves."""

    def __init__(self, size=65536):
        self.size = size
        self.buffer = bytearray(size)

    def read_u32(self, addr):
        if addr < 0 or addr + 4 > self.size:
            raise OutOfBoundsFault(f"address {addr:#x} out of bounds (size={self.size:#x})")
        return struct.unpack_from("<I", self.buffer, addr)[0]

    def write_u32(self, addr, val):
        if addr < 0 or addr + 4 > self.size:
            raise OutOfBoundsFault(f"address {addr:#x} out of bounds (size={self.size:#x})")
        struct.pack_into("<I", self.buffer, addr, val & MASK32)


class FlatScratchAperture:
    """Aperture detector and address translator for 64-bit flat addresses to scratch offsets."""

    def __init__(self, base_addr=0x00007FFF00000000, size=0x10000000):
        self.base_addr = base_addr
        self.size = size

    def is_scratch(self, flat_addr):
        return self.base_addr <= flat_addr < (self.base_addr + self.size)

    def to_scratch_offset(self, flat_addr):
        if not self.is_scratch(flat_addr):
            raise InvalidApertureFault(f"flat address {flat_addr:#018x} outside scratch aperture")
        return flat_addr - self.base_addr


class WavefrontContext:
    """Models an RDNA2 wavefront executing scratch instructions."""

    def __init__(self, wave_id, memory, wave_scratch_size=4096, wavefront_size=32):
        self.wave_id = wave_id
        self.memory = memory
        self.wavefront_size = wavefront_size
        self.wave_scratch_size = wave_scratch_size
        self.wave_base = wave_id * wave_scratch_size * wavefront_size
        self.exec_mask = (1 << wavefront_size) - 1
        self.sgprs = [0] * 106
        self.vgprs = [[0] * wavefront_size for _ in range(256)]
        self.flat_scratch_lo = self.wave_base & MASK32
        self.flat_scratch_hi = (self.wave_base >> 32) & MASK32
        self.aperture = FlatScratchAperture()

    def set_exec_lane(self, lane, active):
        if active:
            self.exec_mask |= (1 << lane)
        else:
            self.exec_mask &= ~(1 << lane)

    def scratch_load_dword(self, vdst, vaddr=None, saddr=None, offset=0):
        """Execute scratch_load_dword vdst, vaddr, saddr offset:offset."""
        scalar_base = self.sgprs[saddr] if saddr is not None else 0
        for lane in range(self.wavefront_size):
            if not (self.exec_mask & (1 << lane)):
                continue
            vector_off = self.vgprs[vaddr][lane] if vaddr is not None else 0
            effective_offset = scalar_base + vector_off + offset
            if effective_offset % 4 != 0:
                raise AlignmentFault(f"lane {lane} unaligned scratch load: offset={effective_offset}")
            if effective_offset < 0:
                raise UnderflowFault(f"lane {lane} scratch load underflow: offset={effective_offset}")
            if effective_offset >= self.wave_scratch_size:
                raise OutOfBoundsFault(
                    f"lane {lane} scratch load out of wave bounds: "
                    f"offset={effective_offset} >= limit={self.wave_scratch_size}"
                )
            phys_addr = compute_scratch_address(self.wave_base, lane, effective_offset, self.wavefront_size)
            self.vgprs[vdst][lane] = self.memory.read_u32(phys_addr)

    def scratch_store_dword(self, vaddr=None, vdata=0, saddr=None, offset=0):
        """Execute scratch_store_dword vaddr, vdata, saddr offset:offset."""
        scalar_base = self.sgprs[saddr] if saddr is not None else 0
        for lane in range(self.wavefront_size):
            if not (self.exec_mask & (1 << lane)):
                continue
            vector_off = self.vgprs[vaddr][lane] if vaddr is not None else 0
            effective_offset = scalar_base + vector_off + offset
            if effective_offset % 4 != 0:
                raise AlignmentFault(f"lane {lane} unaligned scratch store: offset={effective_offset}")
            if effective_offset < 0:
                raise UnderflowFault(f"lane {lane} scratch store underflow: offset={effective_offset}")
            if effective_offset >= self.wave_scratch_size:
                raise OutOfBoundsFault(
                    f"lane {lane} scratch store out of wave bounds: "
                    f"offset={effective_offset} >= limit={self.wave_scratch_size}"
                )
            phys_addr = compute_scratch_address(self.wave_base, lane, effective_offset, self.wavefront_size)
            val = self.vgprs[vdata][lane]
            self.memory.write_u32(phys_addr, val)

    def scratch_load_dwordx2(self, vdst, vaddr=None, saddr=None, offset=0):
        """Execute scratch_load_dwordx2: load two consecutive dwords into vdst and vdst+1."""
        self.scratch_load_dword(vdst, vaddr=vaddr, saddr=saddr, offset=offset)
        self.scratch_load_dword(vdst + 1, vaddr=vaddr, saddr=saddr, offset=offset + 4)

    def scratch_load_dwordx4(self, vdst, vaddr=None, saddr=None, offset=0):
        """Execute scratch_load_dwordx4: load four consecutive dwords into vdst..vdst+3."""
        for i in range(4):
            self.scratch_load_dword(vdst + i, vaddr=vaddr, saddr=saddr, offset=offset + (i * 4))

    def flat_load_dword(self, vdst, vaddr_pair):
        """Execute flat_load_dword through flat pointer decoding."""
        for lane in range(self.wavefront_size):
            if not (self.exec_mask & (1 << lane)):
                continue
            lo = self.vgprs[vaddr_pair][lane]
            hi = self.vgprs[vaddr_pair + 1][lane]
            flat_addr = (hi << 32) | lo
            scratch_off = self.aperture.to_scratch_offset(flat_addr)
            if scratch_off % 4 != 0:
                raise AlignmentFault(f"lane {lane} unaligned flat load: offset={scratch_off}")
            if scratch_off >= self.wave_scratch_size:
                raise OutOfBoundsFault(f"lane {lane} flat scratch out of bounds: offset={scratch_off}")
            phys_addr = compute_scratch_address(self.wave_base, lane, scratch_off, self.wavefront_size)
            self.vgprs[vdst][lane] = self.memory.read_u32(phys_addr)


# Hand-derived address mapping table for Wave32, WaveBase=0x1000
# Format: (lane, offset) -> expected physical address
ORACLE_SCRATCH_ADDRESSES_WAVE32 = {
    (0, 0): 0x1000,
    (1, 0): 0x1004,
    (2, 0): 0x1008,
    (31, 0): 0x107C,
    (0, 4): 0x1080,
    (1, 4): 0x1084,
    (31, 4): 0x10FC,
    (0, 8): 0x1100,
    (16, 8): 0x1140,
    (31, 8): 0x117C,
    (0, 64): 0x1800,
    (31, 64): 0x187C,
}


class FlatScratchAddressGenerationTests(unittest.TestCase):
    """Validates the mathematical swizzle formula and address generation against hand-derived ISA rules."""

    def test_oracle_address_mapping_wave32(self):
        wave_base = 0x1000
        for (lane, offset), expected_addr in ORACLE_SCRATCH_ADDRESSES_WAVE32.items():
            with self.subTest(lane=lane, offset=offset):
                addr = compute_scratch_address(wave_base, lane, offset, wavefront_size=32)
                self.assertEqual(addr, expected_addr)

    def test_lane_stride_is_four_bytes(self):
        wave_base = 0x2000
        for lane in range(31):
            addr_curr = compute_scratch_address(wave_base, lane, 0, wavefront_size=32)
            addr_next = compute_scratch_address(wave_base, lane + 1, 0, wavefront_size=32)
            self.assertEqual(addr_next - addr_curr, 4)

    def test_dword_stride_is_wavefront_times_four(self):
        wave_base = 0x4000
        stride = 32 * 4
        for dword in range(16):
            addr_0 = compute_scratch_address(wave_base, 0, dword * 4, wavefront_size=32)
            addr_1 = compute_scratch_address(wave_base, 0, (dword + 1) * 4, wavefront_size=32)
            self.assertEqual(addr_1 - addr_0, stride)

    def test_wave64_address_generation(self):
        wave_base = 0x8000
        self.assertEqual(compute_scratch_address(wave_base, 0, 0, wavefront_size=64), 0x8000)
        self.assertEqual(compute_scratch_address(wave_base, 63, 0, wavefront_size=64), 0x8000 + 63 * 4)
        self.assertEqual(compute_scratch_address(wave_base, 0, 4, wavefront_size=64), 0x8000 + 64 * 4)
        self.assertEqual(compute_scratch_address(wave_base, 63, 4, wavefront_size=64), 0x8000 + 64 * 4 + 63 * 4)


class ScratchLoadDwordAddressingModesTests(unittest.TestCase):
    """Tests all addressing modes of scratch_load_dword on an RDNA2 wavefront."""

    def setUp(self):
        self.mem = ScratchMemory(size=262144)
        self.wave = WavefrontContext(wave_id=0, memory=self.mem, wave_scratch_size=4096, wavefront_size=32)

    def test_mode_a_scalar_base_plus_immediate_offset(self):
        self.wave.sgprs[32] = 64
        for lane in range(32):
            self.wave.vgprs[1][lane] = 0x1000 + lane
        self.wave.scratch_store_dword(vaddr=None, vdata=1, saddr=32, offset=16)

        self.wave.scratch_load_dword(vdst=2, vaddr=None, saddr=32, offset=16)
        for lane in range(32):
            self.assertEqual(self.wave.vgprs[2][lane], 0x1000 + lane)

    def test_mode_b_vector_offset_plus_immediate_offset(self):
        for lane in range(32):
            self.wave.vgprs[10][lane] = lane * 4
            self.wave.vgprs[1][lane] = 0xA000 + lane
        self.wave.scratch_store_dword(vaddr=10, vdata=1, saddr=None, offset=8)

        self.wave.scratch_load_dword(vdst=2, vaddr=10, saddr=None, offset=8)
        for lane in range(32):
            self.assertEqual(self.wave.vgprs[2][lane], 0xA000 + lane)

    def test_mode_c_combined_vector_scalar_immediate_offset(self):
        self.wave.sgprs[32] = 32
        for lane in range(32):
            self.wave.vgprs[10][lane] = (31 - lane) * 4
            self.wave.vgprs[1][lane] = 0xC000 + (lane * 7)
        self.wave.scratch_store_dword(vaddr=10, vdata=1, saddr=32, offset=12)

        self.wave.scratch_load_dword(vdst=2, vaddr=10, saddr=32, offset=12)
        for lane in range(32):
            self.assertEqual(self.wave.vgprs[2][lane], 0xC000 + (lane * 7))

    def test_mode_d_immediate_only_offset(self):
        for lane in range(32):
            self.wave.vgprs[1][lane] = 0x7700 | lane
        self.wave.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=24)

        self.wave.scratch_load_dword(vdst=5, vaddr=None, saddr=None, offset=24)
        for lane in range(32):
            self.assertEqual(self.wave.vgprs[5][lane], 0x7700 | lane)

    def test_negative_offset_relative_to_frame_pointer(self):
        self.wave.sgprs[32] = 128
        for lane in range(32):
            self.wave.vgprs[1][lane] = 0xFEED0000 | lane
        self.wave.scratch_store_dword(vaddr=None, vdata=1, saddr=32, offset=-64)

        self.wave.scratch_load_dword(vdst=4, vaddr=None, saddr=32, offset=-64)
        for lane in range(32):
            self.assertEqual(self.wave.vgprs[4][lane], 0xFEED0000 | lane)


class PerWaveStackFrameTests(unittest.TestCase):
    """Validates multi-wave stack frame isolation and stack pointer arithmetic."""

    def setUp(self):
        self.mem = ScratchMemory(size=262144)
        self.wave0 = WavefrontContext(wave_id=0, memory=self.mem, wave_scratch_size=1024, wavefront_size=32)
        self.wave1 = WavefrontContext(wave_id=1, memory=self.mem, wave_scratch_size=1024, wavefront_size=32)
        self.wave2 = WavefrontContext(wave_id=2, memory=self.mem, wave_scratch_size=1024, wavefront_size=32)

    def test_multi_wave_stack_isolation(self):
        for lane in range(32):
            self.wave0.vgprs[1][lane] = 0x11110000 | lane
            self.wave1.vgprs[1][lane] = 0x22220000 | lane
            self.wave2.vgprs[1][lane] = 0x33330000 | lane

        self.wave0.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=32)
        self.wave1.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=32)
        self.wave2.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=32)

        self.wave0.scratch_load_dword(vdst=2, vaddr=None, saddr=None, offset=32)
        self.wave1.scratch_load_dword(vdst=2, vaddr=None, saddr=None, offset=32)
        self.wave2.scratch_load_dword(vdst=2, vaddr=None, saddr=None, offset=32)

        for lane in range(32):
            self.assertEqual(self.wave0.vgprs[2][lane], 0x11110000 | lane)
            self.assertEqual(self.wave1.vgprs[2][lane], 0x22220000 | lane)
            self.assertEqual(self.wave2.vgprs[2][lane], 0x33330000 | lane)

    def test_stack_push_pop_frame_lifecycle(self):
        s32 = 32
        self.wave0.sgprs[s32] = 0
        for lane in range(32):
            self.wave0.vgprs[0][lane] = 0xAAAA0000 | lane
        self.wave0.scratch_store_dword(vaddr=None, vdata=0, saddr=s32, offset=0)

        # Allocate new nested stack frame of 64 bytes
        self.wave0.sgprs[s32] += 64
        for lane in range(32):
            self.wave0.vgprs[1][lane] = 0xBBBB0000 | lane
        self.wave0.scratch_store_dword(vaddr=None, vdata=1, saddr=s32, offset=0)

        # Load local variable in current frame
        self.wave0.scratch_load_dword(vdst=2, vaddr=None, saddr=s32, offset=0)
        # Load caller variable in parent frame via negative offset
        self.wave0.scratch_load_dword(vdst=3, vaddr=None, saddr=s32, offset=-64)

        for lane in range(32):
            self.assertEqual(self.wave0.vgprs[2][lane], 0xBBBB0000 | lane)
            self.assertEqual(self.wave0.vgprs[3][lane], 0xAAAA0000 | lane)

        # Pop stack frame
        self.wave0.sgprs[s32] -= 64
        self.wave0.scratch_load_dword(vdst=4, vaddr=None, saddr=s32, offset=0)
        for lane in range(32):
            self.assertEqual(self.wave0.vgprs[4][lane], 0xAAAA0000 | lane)


class ExecutionMaskTests(unittest.TestCase):
    """Tests execution mask (EXEC) filtering on scratch memory loads and stores."""

    def setUp(self):
        self.mem = ScratchMemory(size=65536)
        self.wave = WavefrontContext(wave_id=0, memory=self.mem, wave_scratch_size=1024, wavefront_size=32)

    def test_inactive_lanes_preserve_stored_data(self):
        for lane in range(32):
            self.wave.vgprs[1][lane] = 0x55555555
        self.wave.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=16)

        # Deactivate odd lanes
        self.wave.exec_mask = 0x55555555
        for lane in range(32):
            self.wave.vgprs[1][lane] = 0xAAAAAAAA
        self.wave.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=16)

        # Reactivate all lanes and read back
        self.wave.exec_mask = 0xFFFFFFFF
        self.wave.scratch_load_dword(vdst=2, vaddr=None, saddr=None, offset=16)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(self.wave.vgprs[2][lane], 0xAAAAAAAA)
            else:
                self.assertEqual(self.wave.vgprs[2][lane], 0x55555555)

    def test_inactive_lanes_preserve_destination_vgpr(self):
        for lane in range(32):
            self.wave.vgprs[1][lane] = 0x99990000 | lane
            self.wave.vgprs[2][lane] = 0x11111111
        self.wave.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=32)

        # Active only lanes 0..15
        self.wave.exec_mask = 0x0000FFFF
        self.wave.scratch_load_dword(vdst=2, vaddr=None, saddr=None, offset=32)

        for lane in range(32):
            if lane < 16:
                self.assertEqual(self.wave.vgprs[2][lane], 0x99990000 | lane)
            else:
                self.assertEqual(self.wave.vgprs[2][lane], 0x11111111)


class MultiDwordAndFlatApertureTests(unittest.TestCase):
    """Tests multi-dword vector scratch instructions and 64-bit flat aperture translation."""

    def setUp(self):
        self.mem = ScratchMemory(size=131072)
        self.wave = WavefrontContext(wave_id=0, memory=self.mem, wave_scratch_size=2048, wavefront_size=32)

    def test_scratch_load_dwordx2(self):
        for lane in range(32):
            self.wave.vgprs[0][lane] = 0x1000 + lane
            self.wave.vgprs[1][lane] = 0x2000 + lane
        self.wave.scratch_store_dword(vaddr=None, vdata=0, saddr=None, offset=0)
        self.wave.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=4)

        self.wave.scratch_load_dwordx2(vdst=2, vaddr=None, saddr=None, offset=0)
        for lane in range(32):
            self.assertEqual(self.wave.vgprs[2][lane], 0x1000 + lane)
            self.assertEqual(self.wave.vgprs[3][lane], 0x2000 + lane)

    def test_scratch_load_dwordx4(self):
        for dword in range(4):
            for lane in range(32):
                self.wave.vgprs[dword][lane] = (dword << 16) | lane
            self.wave.scratch_store_dword(vaddr=None, vdata=dword, saddr=None, offset=dword * 4)

        self.wave.scratch_load_dwordx4(vdst=4, vaddr=None, saddr=None, offset=0)
        for dword in range(4):
            for lane in range(32):
                self.assertEqual(self.wave.vgprs[4 + dword][lane], (dword << 16) | lane)

    def test_flat_load_dword_inside_aperture(self):
        for lane in range(32):
            self.wave.vgprs[1][lane] = 0xCAFE0000 | lane
        self.wave.scratch_store_dword(vaddr=None, vdata=1, saddr=None, offset=48)

        # Set flat pointers in v[10:11]
        base = self.wave.aperture.base_addr + 48
        for lane in range(32):
            self.wave.vgprs[10][lane] = base & MASK32
            self.wave.vgprs[11][lane] = (base >> 32) & MASK32

        self.wave.flat_load_dword(vdst=5, vaddr_pair=10)
        for lane in range(32):
            self.assertEqual(self.wave.vgprs[5][lane], 0xCAFE0000 | lane)


class ScratchFaultHandlingTests(unittest.TestCase):
    """Verifies that invalid scratch accesses raise appropriate ISA fault conditions."""

    def setUp(self):
        self.mem = ScratchMemory(size=32768)
        self.wave = WavefrontContext(wave_id=0, memory=self.mem, wave_scratch_size=512, wavefront_size=32)

    def test_unaligned_load_raises_alignment_fault(self):
        for unaligned_off in (1, 2, 3, 5, 7, 9):
            with self.subTest(offset=unaligned_off):
                with self.assertRaises(AlignmentFault):
                    self.wave.scratch_load_dword(vdst=0, vaddr=None, saddr=None, offset=unaligned_off)

    def test_unaligned_store_raises_alignment_fault(self):
        with self.assertRaises(AlignmentFault):
            self.wave.scratch_store_dword(vaddr=None, vdata=0, saddr=None, offset=6)

    def test_out_of_bounds_wave_stack_raises_fault(self):
        with self.assertRaises(OutOfBoundsFault):
            self.wave.scratch_load_dword(vdst=0, vaddr=None, saddr=None, offset=512)

    def test_negative_underflow_raises_fault(self):
        self.wave.sgprs[32] = 0
        with self.assertRaises(UnderflowFault):
            self.wave.scratch_load_dword(vdst=0, vaddr=None, saddr=32, offset=-4)

    def test_flat_address_outside_aperture_raises_fault(self):
        for lane in range(32):
            self.wave.vgprs[10][lane] = 0x00001000
            self.wave.vgprs[11][lane] = 0x00000000
        with self.assertRaises(InvalidApertureFault):
            self.wave.flat_load_dword(vdst=0, vaddr_pair=10)


if __name__ == "__main__":
    unittest.main()
