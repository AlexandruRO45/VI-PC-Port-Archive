"""Unit tests for rdna2spirv.map_vertex_inputs (standard library only, no game data).

    python -I tests/test_vertex_input_locations.py

D3D-388: each element in a packed ISG1 register has a unique Vulkan location and retains its VGPR columns.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402

FLOAT, UINT = 3, 1


class VertexInputLocationsTest(unittest.TestCase):
    def map_inputs(self, sig):
        gen = rdna2spirv.Gen("vs")
        rdna2spirv.map_vertex_inputs(gen, sig)
        return gen.vertex_inputs, gen.vertex_routes

    def test_distinct_registers_keep_order(self):
        sig = [(2, 0xF, 0, FLOAT), (0, 0x7, 0, FLOAT), (1, 0x3, 0, UINT)]
        self.assertEqual(self.map_inputs(sig), ([(2, FLOAT), (0, FLOAT), (1, UINT)], {}))

    def test_packed_register_uses_unique_locations(self):
        sig = [(0, 0x3, 0, FLOAT), (0, 0xC, 0, FLOAT), (1, 0xF, 0, FLOAT)]
        self.assertEqual(self.map_inputs(sig), ([(0, FLOAT), (2, FLOAT), (1, FLOAT)], {0: (0, 0x3), 2: (0, 0xC)}))

    def test_three_elements_in_one_register(self):
        sig = [(4, 0x1, 0, UINT), (4, 0x2, 0, UINT), (4, 0xC, 0, UINT)]
        self.assertEqual(self.map_inputs(sig), ([(4, UINT), (0, UINT), (1, UINT)], {4: (4, 0x1), 0: (4, 0x2), 1: (4, 0xC)}))

    def test_mixed_component_types_keep_their_own_locations(self):
        sig = [(0, 0x3, 0, FLOAT), (0, 0xC, 0, UINT)]
        self.assertEqual(self.map_inputs(sig), ([(0, FLOAT), (1, UINT)], {0: (0, 0x3), 1: (0, 0xC)}))

    def test_empty_signature(self):
        self.assertEqual(self.map_inputs([]), ([], {}))


if __name__ == "__main__":
    unittest.main()
