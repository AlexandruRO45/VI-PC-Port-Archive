"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Cubemap Suite.

Subsystem Feature Bundle 16 covers the 4 RDNA2 cubemap coordinate projection instructions:
  1. v_cubeid_f32  - Cubemap face index identification (0.0 to 5.0)
  2. v_cubesc_f32  - Cubemap S coordinate projection (unnormalized S coordinate)
  3. v_cubetc_f32  - Cubemap T coordinate projection (unnormalized T coordinate)
  4. v_cubema_f32  - Cubemap Major Axis magnitude calculation (2 * major_axis, +0.0 for ±0)

RDNA2 Cubemap Projection Architecture:
  Given 3D vector coordinates (x, y, z):
  1. Major axis selection follows priority:
       |z| >= |x| and |z| >= |y|  => Z major
       else |y| >= |x|            => Y major
       else                       => X major
  2. Face mapping:
       +X (face 0): sc = -z, tc = -y, ma = 2*x
       -X (face 1): sc = +z, tc = -y, ma = 2*x
       +Y (face 2): sc = +x, tc = +z, ma = 2*y
       -Y (face 3): sc = +x, tc = -z, ma = 2*y
       +Z (face 4): sc = +x, tc = -y, ma = 2*z
       -Z (face 5): sc = -x, tc = -y, ma = 2*z
  3. v_cubema_f32 returns +0.0 (never -0.0) whenever the selected major axis is ±0.0.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF


def u32_to_f32(u: int) -> float:
    return struct.unpack(">f", struct.pack(">I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack(">I", struct.pack(">f", f))[0]


def select_cube_face(x: float, y: float, z: float) -> tuple[int, float, float, float]:
    """RDNA2 hardware cubemap projection model returning (face_id, sc, tc, ma)."""
    ax, ay, az = abs(x), abs(y), abs(z)
    if az >= ax and az >= ay:
        # Z major
        if z < 0.0:
            face = 5
            sc = -x
            tc = -y
            ma = 2.0 * z
        else:
            face = 4
            sc = x
            tc = -y
            ma = 2.0 * z
    elif ay >= ax:
        # Y major
        if y < 0.0:
            face = 3
            sc = x
            tc = -z
            ma = 2.0 * y
        else:
            face = 2
            sc = x
            tc = z
            ma = 2.0 * y
    else:
        # X major
        if x < 0.0:
            face = 1
            sc = z
            tc = -y
            ma = 2.0 * x
        else:
            face = 0
            sc = -z
            tc = -y
            ma = 2.0 * x
    # Normalize ±0.0 to +0.0 for cubema
    if ma == 0.0:
        ma = 0.0
    return face, sc, tc, ma

# -----------------------------------------------------------------------------
# Opcode 1: v_cubeid_f32
# -----------------------------------------------------------------------------
def oracle_cubeid_f32(x: float, y: float, z: float) -> float:
    return float(select_cube_face(x, y, z)[0])


class VCubeIdOracleTests(unittest.TestCase):
    def test_cubeid_cardinal_faces(self):
        # +X -> face 0.0
        self.assertEqual(oracle_cubeid_f32(1.0, 0.0, 0.0), 0.0)
        # -X -> face 1.0
        self.assertEqual(oracle_cubeid_f32(-1.0, 0.0, 0.0), 1.0)
        # +Y -> face 2.0
        self.assertEqual(oracle_cubeid_f32(0.0, 1.0, 0.0), 2.0)
        # -Y -> face 3.0
        self.assertEqual(oracle_cubeid_f32(0.0, -1.0, 0.0), 3.0)
        # +Z -> face 4.0
        self.assertEqual(oracle_cubeid_f32(0.0, 0.0, 1.0), 4.0)
        # -Z -> face 5.0
        self.assertEqual(oracle_cubeid_f32(0.0, 0.0, -1.0), 5.0)

if __name__ == "__main__":
    unittest.main()
