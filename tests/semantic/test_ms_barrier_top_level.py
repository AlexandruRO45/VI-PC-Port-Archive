"""A barrier in a block that every path reaches is not put behind a %pc guard (native structured form).

The program below is runtime/test/shaders/ms_wave_info_s3.asm without its LDS count: 128 threads in two wave64
waves, wave 0 skips MSG_GS_ALLOC_REQ with s_cbranch_scc1 and both waves meet at the s_barrier after it.
cfg_structure.structure guards that join with `if (%pc == 2)` like any other block, and AMD's Windows driver 24.10.24
(Radeon gfx1036) crashes inside vkCreateGraphicsPipelines on a mesh shader with a barrier in any selection
construct, even one on a push constant (hand-edited variants of the translated ms_wave_info_s3). %pc is 2 on every
path into that guard, so the translator drops it (the same known-%pc analysis as the emulated wave64 tree) and the
barrier is at the top level, where every invocation runs it. The send stays behind its guard.

Standard library only; the spirv-val check needs SPIRV-Tools.
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402  (path set above)
import rdna2spirv  # noqa: E402

# llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -mattr=+wavefrontsize64,-wavefrontsize32 -show-encoding (the branch
# offset needs the encodings).
LISTING = """\
; ms_join_barrier  hand-written test program (ms_wave_info_s3.asm without the LDS count)
	v_mbcnt_lo_u32_b32 v0, -1, 0            ; encoding: [0x00,0x00,0x65,0xd7,0xc1,0x00,0x01,0x02]
	v_mbcnt_hi_u32_b32 v0, -1, v0           ; encoding: [0x00,0x00,0x66,0xd7,0xc1,0x00,0x02,0x02]
	s_bfe_u32 s4, s3, 0x40018               ; encoding: [0x03,0xff,0x84,0x93,0x18,0x00,0x04,0x00]
	v_lshl_add_u32 v0, s4, 6, v0            ; encoding: [0x00,0x00,0x46,0xd7,0x04,0x0c,0x01,0x04]
	v_mov_b32_e32 v8, 0.5                   ; encoding: [0xf0,0x02,0x10,0x7e]
	v_cmp_eq_u32_e32 vcc, 1, v0             ; encoding: [0x81,0x00,0x84,0x7d]
	v_cndmask_b32_e64 v1, -1.0, 0x40400000, vcc ; encoding: [0x01,0x00,0x01,0xd5,0xf3,0xfe,0xa9,0x01,0x00,0x00,0x40,0x40]
	v_cndmask_b32_e64 v3, 0, 2.0, vcc       ; encoding: [0x03,0x00,0x01,0xd5,0x80,0xe8,0xa9,0x01]
	v_cmp_eq_u32_e32 vcc, 2, v0             ; encoding: [0x82,0x00,0x84,0x7d]
	v_cndmask_b32_e64 v2, -1.0, 0x40400000, vcc ; encoding: [0x02,0x00,0x01,0xd5,0xf3,0xfe,0xa9,0x01,0x00,0x00,0x40,0x40]
	v_cndmask_b32_e64 v4, 0, 2.0, vcc       ; encoding: [0x04,0x00,0x01,0xd5,0x80,0xe8,0xa9,0x01]
	v_mov_b32_e32 v6, 0                     ; encoding: [0x80,0x02,0x0c,0x7e]
	v_mov_b32_e32 v7, 1.0                   ; encoding: [0xf2,0x02,0x0e,0x7e]
	s_mov_b32 m0, 0x1003                    ; encoding: [0xff,0x03,0xfc,0xbe,0x03,0x10,0x00,0x00]
	s_cmp_eq_u32 s4, 0                      ; encoding: [0x04,0x80,0x06,0xbf]
	s_cbranch_scc1 1                        ; encoding: [0x01,0x00,0x85,0xbf]
	s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)     ; encoding: [0x09,0x00,0x90,0xbf]
	s_barrier                               ; encoding: [0x00,0x00,0x8a,0xbf]
	v_cmpx_gt_u32_e32 3, v0                 ; encoding: [0x83,0x00,0xa8,0x7d]
	exp pos0, v1, v2, v6, v7 done           ; encoding: [0xcf,0x08,0x00,0xf8,0x01,0x02,0x06,0x07]
	exp param0, v3, v4, v8, v7              ; encoding: [0x0f,0x02,0x00,0xf8,0x03,0x04,0x08,0x07]
	v_cmpx_eq_u32_e32 0, v0                 ; encoding: [0x80,0x00,0xa4,0x7d]
	v_mov_b32_e32 v9, 0x200400              ; encoding: [0xff,0x02,0x12,0x7e,0x00,0x04,0x20,0x00]
	exp prim, v9, off, off, off done        ; encoding: [0x41,0x09,0x00,0xf8,0x09,0x00,0x00,0x00]
	s_endpgm                                ; encoding: [0x00,0x00,0x81,0xbf]
"""


def _dxbc(chunks):
    """A DXBC container with the given (fourcc, payload) parts."""
    hdr = 32 + 4 * len(chunks)
    offsets, body = [], bytearray()
    for tag, payload in chunks:
        offsets.append(hdr + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    out = bytearray(b"DXBC") + bytes(16) + struct.pack("<III", 1, hdr + len(body), len(chunks))
    for off in offsets:
        out += struct.pack("<I", off)
    return bytes(out) + bytes(body)


def _mesh_psv0(threads, max_vertices, max_primitives):
    """PSV0 of a triangle mesh shader (stage 13)."""
    r = bytearray(48)
    struct.pack_into("<HH", r, 12, max_vertices, max_primitives)
    r[24] = 13
    r[27] = 2
    struct.pack_into("<III", r, 36, threads, 1, 1)
    return struct.pack("<I", len(r)) + bytes(r)


def _translate():
    old = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = False
    try:
        spv = support.translate_listing("ms_join_barrier", LISTING.encode("utf-8"),
                                        dxbc=_dxbc([(b"PSV0", _mesh_psv0(128, 64, 64))]))
        flow = rdna2spirv.translate.last_gen.control_flow
    finally:
        rdna2spirv.WAVE64_EMULATE = old
    return spv, flow


def _nesting(spv):
    """For each %blkN label and each OpControlBarrier, in order: how many selection/loop constructs it is in."""
    open_merges, seen = [], []
    for line in spv.splitlines():
        line = line.strip()
        m = re.match(r"(%\w+) = OpLabel$", line)
        if m:
            while open_merges and open_merges[-1] == m.group(1):
                open_merges.pop()
            if m.group(1).startswith("%blk"):
                seen.append((m.group(1), len(open_merges)))
            continue
        m = re.match(r"Op(?:Selection|Loop)Merge (%\w+)", line)
        if m:
            open_merges.append(m.group(1))
        elif line.startswith("OpControlBarrier"):
            seen.append(("barrier", len(open_merges)))
    return seen


class MeshBarrierTopLevelTests(unittest.TestCase):
    def test_join_barrier_is_not_guarded(self):
        spv, flow = _translate()
        self.assertEqual(flow, "structured")
        seen = _nesting(spv)
        self.assertEqual([name for name, _ in seen if name.startswith("%blk")], ["%blk0", "%blk1", "%blk2"])
        depth = dict(seen)
        self.assertEqual(depth["%blk0"], 0)
        self.assertGreater(depth["%blk1"], 0, "the send must stay behind its guard: wave 0 skips it")
        self.assertEqual(depth["%blk2"], 0, "the join after the skipped send runs in every invocation")
        # blk2's s_barrier, then the mesh epilogue's before OpSetMeshOutputsEXT: neither in a construct.
        self.assertEqual([d for name, d in seen if name == "barrier"], [0, 0])

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
    def test_assembles_and_validates(self):
        spv, _ = _translate()
        with tempfile.TemporaryDirectory() as tmp:
            asm, out = os.path.join(tmp, "ms.spvasm"), os.path.join(tmp, "ms.spv")
            with open(asm, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", out],
                        ["spirv-val", "--target-env", "vulkan1.2", out]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
