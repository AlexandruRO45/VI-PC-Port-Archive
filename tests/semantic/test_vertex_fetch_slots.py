"""Which VGPRs a fetch-calling vertex shader finds its attributes in.

`s_swappc_b64` at the start of a VS calls the driver's fetch program. It hands out VGPRs only to the input-signature
registers the shader reads (ISG1 read mask, byte 25 of the element, is not 0): the k-th of them is in
v(4 + 4k .. 7 + 4k). A register that is declared and never read gets none, so the registers above it move down
(docs/design.md, Vertex fetch). The Vulkan location stays the D3D input register.

Every program and signature here is written by hand; the attribute values are arbitrary bit patterns, so a value
that reaches the wrong register shows up as the wrong word.
"""
import struct
import unittest

import support

rdna2spirv = support.rdna2spirv

FLOAT, UINT, SINT = 3, 1, 2
SV_VERTEX_ID = 6


def isg1_dxbc(elements):
    """A DXBC container with one ISG1 part. elements: (register, declared mask, read mask, system value, type)."""
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for reg, mask, read_mask, sysv, ctype in elements:
        elem = bytearray(32)
        struct.pack_into("<6I", elem, 0, 0, 0, 0, sysv, ctype, reg)
        elem[24], elem[25] = mask, read_mask
        payload += elem
    part = b"ISG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


def f(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


# One value per component of four input registers: register r, component c holds 0x40r0c000 | marker.
ATTR = {r: tuple(0x40000000 | r << 20 | c << 12 | 0x123 for c in range(4)) for r in range(4)}


def inputs(types):
    """Input values for %vin<r>: the words of ATTR[r], as floats for a float register."""
    return {f"%vin{r}": tuple(f(w) for w in ATTR[r]) if t == FLOAT else ATTR[r] for r, t in types.items()}


def run(lines, dxbc, types):
    spv = support.translate_lines("vs_fetch", lines, dxbc=dxbc)
    return spv, support.run_fragment(spv, inputs=inputs(types))


class FetchSlotRule(unittest.TestCase):
    def test_k_th_read_register(self):
        slots = rdna2spirv.vertex_fetch_slots
        e = lambda reg, read: (reg, 0xF, 0, FLOAT, read)                          # noqa: E731
        self.assertEqual(slots([e(0, 7), e(1, 15), e(2, 3)]), {0: 0, 1: 1, 2: 2})
        self.assertEqual(slots([e(0, 7), e(1, 0), e(2, 3)]), {0: 0, 2: 1})
        self.assertEqual(slots([e(0, 0), e(1, 0), e(2, 1), e(5, 8)]), {2: 0, 5: 1})
        self.assertEqual(slots([e(2, 3), e(0, 7)]), {0: 0, 2: 1})                 # register order, not file order
        self.assertEqual(slots([e(0, 0), e(0, 4), e(1, 1)]), {0: 0, 1: 1})        # two elements share register 0
        self.assertEqual(slots([e(0, 0)]), {})
        self.assertEqual(slots([]), {})

    def test_read_mask_is_byte_25(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            with open(os.path.join(tmp, "vs_x.dxbc"), "wb") as fh:
                fh.write(isg1_dxbc([(0, 7, 5, 0, FLOAT), (1, 1, 0, SV_VERTEX_ID, UINT)]))
            path = os.path.join(tmp, "disasm", "vs_x.s")
            self.assertEqual(rdna2spirv.input_signature_read(path, "vs_x"),
                             [(0, 7, 0, FLOAT, 5), (1, 1, SV_VERTEX_ID, UINT, 0)])
            self.assertEqual(rdna2spirv.input_signature(path, "vs_x"), [(0, 7, 0, FLOAT), (1, 1, SV_VERTEX_ID, UINT)])


class FetchedRegisters(unittest.TestCase):
    # register 0 xyz (read), 1 xyzw (never read), 2 xy (read), 3 xyzw uint (read), and SV_VertexID, which is a
    # system value and takes no attribute slot.
    SIG = [(0, 7, 7, 0, FLOAT), (1, 15, 0, 0, FLOAT), (2, 3, 3, 0, FLOAT), (3, 15, 15, 0, UINT),
           (4, 1, 1, SV_VERTEX_ID, UINT)]
    TYPES = {0: FLOAT, 1: FLOAT, 2: FLOAT, 3: UINT}
    # Reads slot 0 (v4..v6), slot 1 (v8, v9) and slot 2 (v12..v15).
    LINES = ["s_swappc_b64 s[4:5], s[4:5]",
             "exp pos0 v4, v5, v6, v6 done",
             "exp param0 v8, v9, v12, v15",
             "exp param1 v13, v14, v8, v9",
             "s_endpgm"]

    def test_registers_above_an_unread_one_move_down(self):
        spv, out = run(self.LINES, isg1_dxbc(self.SIG), self.TYPES)
        a = ATTR
        self.assertEqual(tuple(out["%out_param0"]), (a[2][0], a[2][1], a[3][0], a[3][3]))
        self.assertEqual(tuple(out["%out_param1"]), (a[3][1], a[3][2], a[2][0], a[2][1]))
        pos = [v for k, v in out.items() if k not in ("%out_param0", "%out_param1")]
        self.assertEqual(len(pos), 1)
        self.assertEqual(tuple(pos[0])[:3], (a[0][0], a[0][1], a[0][2]))

    def test_locations_stay_the_input_registers(self):
        spv, _ = run(self.LINES, isg1_dxbc(self.SIG), self.TYPES)
        for reg in range(4):
            self.assertIn(f"OpDecorate %vin{reg} Location {reg}\n", spv)
        self.assertNotIn("%vin4", spv)                              # the system value is not a vertex input
        self.assertIn("OpLoad %v4float %vin2", spv)
        self.assertNotIn("OpLoad %v4float %vin1", spv)              # never read: nothing is fetched for it

    def test_every_register_read_is_the_old_rule(self):
        sig = [(0, 7, 7, 0, FLOAT), (1, 15, 15, 0, FLOAT), (2, 3, 3, 0, FLOAT), (3, 15, 15, 0, UINT)]
        _, out = run(self.LINES, isg1_dxbc(sig), self.TYPES)
        a = ATTR
        self.assertEqual(tuple(out["%out_param0"]), (a[1][0], a[1][1], a[2][0], a[2][3]))
        self.assertEqual(tuple(out["%out_param1"]), (a[2][1], a[2][2], a[1][0], a[1][1]))

    def test_unread_register_at_the_top_changes_nothing(self):
        sig = [(0, 7, 7, 0, FLOAT), (1, 15, 15, 0, FLOAT), (2, 3, 3, 0, FLOAT), (3, 15, 0, 0, UINT)]
        lines = ["s_swappc_b64 s[4:5], s[4:5]", "exp pos0 v4, v5, v6, v6 done", "exp param0 v8, v11, v12, v13",
                 "s_endpgm"]
        _, out = run(lines, isg1_dxbc(sig), self.TYPES)
        a = ATTR
        self.assertEqual(tuple(out["%out_param0"]), (a[1][0], a[1][3], a[2][0], a[2][1]))

    def test_a_component_outside_the_read_mask_is_still_fetched(self):
        # The fetch program fills all four VGPRs of a register it fetches (format defaults for missing ones).
        sig = [(0, 7, 7, 0, FLOAT), (1, 15, 0, 0, FLOAT), (2, 3, 3, 0, FLOAT)]
        lines = ["s_swappc_b64 s[4:5], s[4:5]", "exp pos0 v4, v5, v6, v7 done", "exp param0 v8, v9, v10, v11",
                 "s_endpgm"]
        _, out = run(lines, isg1_dxbc(sig), {0: FLOAT, 1: FLOAT, 2: FLOAT})
        self.assertEqual(tuple(out["%out_param0"]), ATTR[2])

    def test_reading_past_the_fetched_registers_is_refused(self):
        # Three registers are read, so v16 and up hold nothing: under the old rule v16..v19 were register 3.
        lines = ["s_swappc_b64 s[4:5], s[4:5]", "exp pos0 v4, v5, v6, v6 done", "exp param0 v8, v12, v16, v16",
                 "s_endpgm"]
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "fetch mapping"):
            support.translate_lines("vs_fetch", lines, dxbc=isg1_dxbc(self.SIG))

    def test_no_register_read(self):
        sig = [(0, 7, 0, 0, FLOAT)]
        lines = ["s_swappc_b64 s[4:5], s[4:5]", "v_mov_b32 v1, 1.0", "exp pos0 v1, v1, v1, v1 done", "s_endpgm"]
        spv = support.translate_lines("vs_fetch", lines, dxbc=isg1_dxbc(sig))
        self.assertIn("OpDecorate %vin0 Location 0\n", spv)
        self.assertNotIn("OpLoad %v4float %vin0", spv)
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "fetch mapping"):
            support.translate_lines("vs_fetch", lines[:1] + ["exp pos0 v4, v4, v4, v4 done", "s_endpgm"],
                                    dxbc=isg1_dxbc(sig))


if __name__ == "__main__":
    unittest.main()
