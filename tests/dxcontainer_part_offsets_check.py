"""Synthetic DXContainer directory-order and overlap regression fixtures."""

from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from resource_inventory import dxil_program, parts, shader_markers


def container(*, reversed_directory=False, overlapping=False, extra_fourcc=None):
    # This is a deliberately minimal framing probe, not executable DXIL.
    # No third-party shaders or game assets are required.
    program = struct.pack("<II4sIII", 0, 8, b"DXIL", 0, 16, 4) + b"BC\xc0\xde" + bytes(4)
    assert len(program) == 32
    dxil = b"DXIL" + struct.pack("<I", len(program)) + program
    note = b"NOTE" + struct.pack("<I", 4) + b"test"
    if extra_fourcc is not None:
        assert len(extra_fourcc) == 4 and not overlapping
    sections = [dxil, note]
    if extra_fourcc is not None:
        sections.append(extra_fourcc + struct.pack("<I", 5) + b"extra")
    table_end = 32 + 4 * len(sections)
    offsets = []
    cursor = table_end
    for section in sections:
        offsets.append(cursor)
        cursor += len(section)
    physical = b"".join(sections)
    if overlapping:
        # Make a second valid part header WITHIN the first one's payload.
        # In this case both intervals may be in-bounds, but overlap.
        physical = b"OUTR" + struct.pack("<I", 28) + b"INNR" + struct.pack("<I", 4) + b"test" + bytes(16)
        offsets = [table_end, table_end + 8]
    if reversed_directory:
        offsets.reverse()
    size = table_end + len(physical)
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, size, len(offsets))
            + struct.pack(f"<{len(offsets)}I", *offsets) + physical)


class DXContainerPartOffsets(unittest.TestCase):
    def test_directory_order_is_independent_of_physical_part_order(self):
        normal, reversed_offsets = container(), container(reversed_directory=True)
        self.assertNotEqual(normal, reversed_offsets)
        self.assertEqual(parts(normal), parts(reversed_offsets))
        self.assertTrue(dxil_program(normal, 0))
        self.assertTrue(dxil_program(reversed_offsets, 0))
        self.assertEqual(shader_markers(normal)["framed_dxil_programs"], 1)
        self.assertEqual(shader_markers(reversed_offsets)["framed_dxil_programs"], 1)

    def test_overlap_rejected_regardless_of_directory_order(self):
        for reversed_directory in (False, True):
            with self.subTest(reversed_directory=reversed_directory):
                data = container(overlapping=True, reversed_directory=reversed_directory)
                with self.assertRaises(ValueError):
                    parts(data)
                self.assertFalse(dxil_program(data, 0))

    def test_duplicate_fourcc_rejected_by_both_readers(self):
        # Two separate NOTE sections overlap neither each other nor the DXIL
        # section. A unique FourCC control proves valid DXIL framing is present.
        for reverse in (False, True):
            with self.subTest(reversed_directory=reverse):
                control = container(reversed_directory=reverse, extra_fourcc=b"UNIQ")
                duplicate = container(reversed_directory=reverse, extra_fourcc=b"NOTE")
                self.assertEqual(len(control), len(duplicate))
                self.assertEqual(set(parts(control)), {b"DXIL", b"NOTE", b"UNIQ"})
                self.assertTrue(dxil_program(control, 0))
                self.assertEqual(shader_markers(control)["framed_dxil_programs"], 1)
                self.assertIn(b"BC\xc0\xde", duplicate)
                with self.assertRaisesRegex(ValueError, "Invalid container part"):
                    parts(duplicate)
                self.assertFalse(dxil_program(duplicate, 0))
                self.assertEqual(shader_markers(duplicate)["framed_dxil_programs"], 0)

    def test_part_outside_container_remains_invalid(self):
        data = bytearray(container())
        struct.pack_into("<I", data, 36, len(data) - 4)
        with self.assertRaises(ValueError):
            parts(data)
        self.assertFalse(dxil_program(data, 0))


if __name__ == "__main__":
    unittest.main()
