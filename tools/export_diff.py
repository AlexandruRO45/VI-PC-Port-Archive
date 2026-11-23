r"""FUZZ-114: differential fuzzer for VS/PS export shapes (standard library only).

Usage:
    python -I tools/export_diff.py --programs N [--seed S]
    python -I tools/export_diff.py --self-test
    python -I tools/export_diff.py --list-ops
"""
import os
import struct
import sys
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

# D3D_NAME system values the export rules read (rdna2spirv.output_signature_full / pos_export_layout).
D3D_NAME = {"clip": 2, "cull": 3, "rtai": 4, "vai": 5, "shading_rate": 23, "target": 64, "depth": 65}

# One 32-byte OSG1 signature element: system value, semantic index, register, component mask, read/write mask.
Osg1Element = namedtuple("Osg1Element", "sysv index register wmask_lo wmask_hi")


def osg1_dxbc(elements):
    """A synthetic DXBC container holding one OSG1 (output signature) chunk with `elements`.

    The layout is the one rdna2spirv.output_signature_full() reads: a <II (count, offset) header, then 32-byte
    elements of <6I (stream, name, semantic index, system value, type, register), the component mask at byte 24 and
    the read/write mask at byte 25. Name 0 and type 3 (float), as in the reference builder of
    tests/semantic/test_sparse_sv_target.py.
    """
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for e in elements:
        elem = bytearray(32)
        struct.pack_into("<6I", elem, 0, 0, 0, e.index, e.sysv, 3, e.register)
        elem[24], elem[25] = e.wmask_lo, e.wmask_hi
        payload += elem
    part = b"OSG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part
