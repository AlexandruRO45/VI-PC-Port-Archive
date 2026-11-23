r"""Measure the pixel shader's system VGPRs on an AMD GPU: POS_X/Y/Z/W_FLOAT, FRONT_FACE, ANCILLARY, SAMPLE_COVERAGE and
POS_FIXED_PT (raw RDNA2 ISA through ps_oracle.py).

docs/design.md's "PS system inputs" row fills v2.. from Vulkan's FragCoord and a front-face float, from Mesa and a
corpus count; whether the hardware delivers those words was not measured. The SPI writes each enabled input into
consecutive VGPRs after the barycentric pairs, in the order X, Y, Z, W, front face, ancillary, sample coverage, fixed
point. This enables all eight (SPI_PS_INPUT_ENA and _ADDR in the PAL metadata), draws a triangle into an 8 x 8 target
and exports the registers. Three vertex shaders cover the cases: a w of 1, 2 and 4 at the three vertices
(perspective), a constant w of 2, and both windings.

Checks (each printed with its result):

  pos_xy        v2 / v3 are the pixel centre, x + 0.5 and y + 0.5, at every pixel
  pos_z         v4 is the NDC z (0.25 here) at every pixel
  pos_w_const   v5 is 2.0 for a vertex w of 2, where Vulkan's FragCoord.w is 0.5: the clip w, not 1/w
  pos_w_persp   v5 is 1 / (the interpolated 1/w) for vertex w 1, 2 and 4 (relative error under 1e-6), not the
                interpolated w
  front_mask    SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS = 1 (the driver's setting): front-facing 0xffffffff, back 0
  front_float   FRONT_FACE_ALL_BITS = 0: front +1.0 (0x3f800000), back -1.0 (0xbf800000)
  coverage      v8 is 1 (one sample, covered) at every pixel
  fixed_pt      v9 is the integer pixel position, y << 16 | x

Which winding is front-facing is not assumed: the template pixel shader writes Vulkan's FrontFacing, drawn by the
driver's own compile of it, and the registers are compared with that. The ancillary word is printed, not checked.

    python -I tools/ps_sysvals_probe.py --build <runtime build> [--json out.json]

Select the GPU with LEONIDA_VK_DEVICE, as for ps_oracle. Exit codes: 0 every check holds (the gfx1036 measurement),
1 a check differs, 2 a harness error, 77 no pixel-shader oracle on this device (see ps_oracle). Standard library
only, no game files.
"""
import argparse
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ps_oracle  # noqa: E402

SIZE = (8, 8)
Z_NDC = 0.25
ENABLED = ("persp_center_ena", "pos_x_float_ena", "pos_y_float_ena", "pos_z_float_ena", "pos_w_float_ena",
           "front_face_ena", "ancillary_ena", "sample_coverage_ena", "pos_fixed_pt_ena")
LOW = ["exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]         # x, y, z, w
HIGH = ["exp mrt0 v6, v7, v8, v9 done vm", "s_endpgm"]        # front face, ancillary, sample coverage, fixed point


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def vertex_shader(swap, ws):
    """Full-screen triangle (-1,-1), (3,-1), (-1,3) with vertices 1 and 2 swapped if `swap`; the clip position is
    (x * w, y * w, Z_NDC * w, w) with the vertex's w from `ws`, so the NDC position does not depend on w."""
    w0, w1, w2 = ws
    pick = ("b2", "b1") if swap else ("b1", "b2")
    return f"""\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %gl_pos %vid %o0 %o1
               OpDecorate %gl_pos BuiltIn Position
               OpDecorate %vid BuiltIn VertexIndex
               OpDecorate %o0 Location 0
               OpDecorate %o1 Location 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
         %v4 = OpTypeVector %float 4
       %p_o4 = OpTypePointer Output %v4
       %p_ii = OpTypePointer Input %int
     %gl_pos = OpVariable %p_o4 Output
         %o0 = OpVariable %p_o4 Output
         %o1 = OpVariable %p_o4 Output
        %vid = OpVariable %p_ii Input
         %i1 = OpConstant %int 1
         %i2 = OpConstant %int 2
         %m1 = OpConstant %float -1
         %p3 = OpConstant %float 3
         %zz = OpConstant %float {Z_NDC}
         %w0 = OpConstant %float {w0}
         %w1 = OpConstant %float {w1}
         %w2 = OpConstant %float {w2}
         %f0 = OpConstant %float 0
         %f1 = OpConstant %float 1
         %c0 = OpConstantComposite %v4 %f0 %f0 %f0 %f1
       %main = OpFunction %void None %fn
          %e = OpLabel
          %v = OpLoad %int %vid
         %b1 = OpIEqual %bool %v %i1
         %b2 = OpIEqual %bool %v %i2
          %x = OpSelect %float %{pick[0]} %p3 %m1
          %y = OpSelect %float %{pick[1]} %p3 %m1
         %wa = OpSelect %float %b1 %w1 %w0
          %w = OpSelect %float %b2 %w2 %wa
         %xw = OpFMul %float %x %w
         %yw = OpFMul %float %y %w
         %zw = OpFMul %float %zz %w
          %p = OpCompositeConstruct %v4 %xw %yw %zw %w
               OpStore %gl_pos %p
               OpStore %o0 %c0
               OpStore %o1 %c0
               OpReturn
               OpFunctionEnd
"""


# Template pixel shader: out = Vulkan's FrontFacing as 1.0 / 0.0 in every channel, built by the driver as usual.
PS_FRONT = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %ff %out
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %ff BuiltIn FrontFacing
               OpDecorate %out Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %bool = OpTypeBool
         %v4 = OpTypeVector %float 4
       %p_ib = OpTypePointer Input %bool
       %p_o4 = OpTypePointer Output %v4
         %ff = OpVariable %p_ib Input
        %out = OpVariable %p_o4 Output
         %f0 = OpConstant %float 0
         %f1 = OpConstant %float 1
       %main = OpFunction %void None %fn
          %e = OpLabel
          %b = OpLoad %bool %ff
          %s = OpSelect %float %b %f1 %f0
          %c = OpCompositeConstruct %v4 %s %s %s %s
               OpStore %out %c
               OpReturn
               OpFunctionEnd
"""


def enable_sets(all_bits=None):
    sets = [(f".graphics_registers/.{reg}/.{k}", True) for k in ENABLED
            for reg in ("spi_ps_input_ena", "spi_ps_input_addr")]
    if all_bits is not None:
        sets.append((".graphics_registers/.spi_baryc_cntl/.front_face_all_bits", all_bits))
    return sets


def draw(build, swap, ws, lines, all_bits=None):
    o = ps_oracle.Oracle(build, vs=vertex_shader(swap, ws), ps=PS_FRONT, size=SIZE)
    try:
        return o.device, o.template_pixels, o.run(lines, enable_sets(all_bits))
    finally:
        o.close()


def measure(build):
    """The device name and {case: (driver FrontFacing words, register words)} for each draw."""
    res = {}
    device = ""
    for name, swap, ws, lines, bits in (
            ("persp_low", False, (1, 2, 4), LOW, None),
            ("const_low", False, (2, 2, 2), LOW, None),
            ("a_high", False, (2, 2, 2), HIGH, True), ("b_high", True, (2, 2, 2), HIGH, True),
            ("a_float", False, (2, 2, 2), HIGH, False), ("b_float", True, (2, 2, 2), HIGH, False)):
        device, template, px = draw(build, swap, ws, lines, bits)
        res[name] = (template, px)
    return device, res


def analyze(res):
    w, h = SIZE
    checks = []
    xy = [(i % w, i // w) for i in range(w * h)]

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    _t, px = res["persp_low"]
    check("pos_xy", all(f32(p[0]) == x + 0.5 and f32(p[1]) == y + 0.5 for p, (x, y) in zip(px, xy)))
    check("pos_z", all(f32(p[2]) == Z_NDC for p in px))
    _t, cpx = res["const_low"]
    check("pos_w_const", all(f32(p[3]) == 2.0 for p in cpx), f"v5 = {f32(cpx[0][3])} for a vertex w of 2")
    ws, worst_a, worst_b = (1.0, 2.0, 4.0), 0.0, 0.0
    for p, (x, y) in zip(px, xy):
        l1, l2 = (x + 0.5) / (2 * w), (y + 0.5) / (2 * h)       # the triangle spans 2 x the target from its corner
        l0 = 1 - l1 - l2
        inv = l0 / ws[0] + l1 / ws[1] + l2 / ws[2]               # FragCoord.w
        reg, lin = f32(p[3]), l0 * ws[0] + l1 * ws[1] + l2 * ws[2]
        worst_a, worst_b = max(worst_a, abs(reg * inv - 1)), max(worst_b, abs(reg / lin - 1))
    check("pos_w_persp", worst_a < 1e-6,
          f"max relative error vs 1/interp(1/w) {worst_a:.1e}, vs interp(w) {worst_b:.2f}")
    for tag, case, front_bits, back_bits in (("mask", "high", 0xFFFFFFFF, 0), ("float", "float", 0x3F800000, 0xBF800000)):
        ok, detail = True, []
        for var in ("a", "b"):
            template, fpx = res[f"{var}_{case}"]
            facing = {int(f32(t[0])) for t in template}          # Vulkan FrontFacing from the driver's compile: 1 or 0
            regs = {p[0] for p in fpx}
            if len(facing) != 1 or len(regs) != 1:
                ok = False
                detail.append(f"{var}: mixed facing {sorted(facing)} / register {sorted(map(hex, regs))}")
                continue
            want = front_bits if facing == {1} else back_bits
            ok &= regs == {want}
            detail.append(f"{'front' if facing == {1} else 'back'} -> 0x{next(iter(regs)):08x}")
        check(f"front_{tag}", ok, ", ".join(detail))
    _t, hpx = res["a_high"]
    check("coverage", all(p[2] == 1 for p in hpx))
    check("fixed_pt", all(p[3] == (y << 16 | x) for p, (x, y) in zip(hpx, xy)),
          f"first pixels 0x{hpx[0][3]:x} 0x{hpx[1][3]:x} 0x{hpx[w][3]:x}")
    return checks, {hex(p[1]) for p in hpx}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="runtime build folder with vk_harness")
    ap.add_argument("--json", help="write {device, checks, words} here")
    a = ap.parse_args(argv)
    try:
        device, res = measure(a.build)
    except ps_oracle.Unavailable as e:
        print(f"ps_sysvals_probe unavailable: {e}")
        return 77
    except (OSError, ValueError, RuntimeError) as e:
        print(f"ps_sysvals_probe: {e}")
        return 2
    print(f"device: {device}")
    checks, ancillary = analyze(res)
    for name, ok, detail in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else ""))
    print(f"ancillary word (not checked): {', '.join(sorted(ancillary))}")
    good = sum(ok for _n, ok, _d in checks)
    print(f"{good}/{len(checks)} checks hold")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"device": device, "checks": [[n, ok, d] for n, ok, d in checks],
                       "words": {n: [list(p) for p in px] for n, (_t, px) in res.items()}}, fh, indent=1)
    return 0 if good == len(checks) else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
