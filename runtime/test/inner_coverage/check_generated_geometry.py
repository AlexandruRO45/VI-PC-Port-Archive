import os, re, struct, subprocess
from pathlib import Path
import coverage as b
p = b.p
root = b.root
compiler = b.compiler
b.art = root / 'generated_geometry'
b.art.mkdir(parents=True, exist_ok=True)
triangles = [[(-0.8, -0.8), (0.8, -0.8), (-0.8, 0.8)], [(-0.4, -0.4), (0.9, -0.4), (-0.4, 0.9)]]

def geometry_shader(order, effects=False):
    points = ', '.join((f'vec2({x},{y})' for tri in order for x, y in tri))
    source = """#version 460
#extension GL_EXT_buffer_reference : require
layout(triangles) in;
layout(triangle_strip, max_vertices=6) out;
layout(location=0) in vec4 unused0[];
layout(location=1) in vec4 unused1[];
layout(location=7) flat in uint instances[];
layout(location=7) flat out uint instanceOut;
layout(location=8) flat out uint privatePrimitive;
layout(buffer_reference, std430, buffer_reference_align=4) buffer Counter {uint value[];};
layout(push_constant) uniform PC {layout(offset=128) Counter heap;} pc;
const vec2 points[6]=vec2[6](""" + points + ');\nvoid main() {\n' + ('atomicAdd(pc.heap.value[4],1);\n' if effects else '') + """
 for (uint triangle=0;triangle<2;triangle++) {
  for (uint vertex=0;vertex<3;vertex++) {
   gl_Position=vec4(points[triangle*3+vertex],0,1);
   gl_PrimitiveID=gl_PrimitiveIDIn;
   privatePrimitive=uint(gl_PrimitiveIDIn)*2+triangle;
   instanceOut=instances[0];
   EmitVertex();
  }
  EndPrimitive();
 }
}
"""
    path = b.art / f'gs_{int(effects)}.geom'
    path.write_text(source, encoding='utf-8', newline='\n')
    out = path.with_suffix('.spv')
    r = subprocess.run([compiler, '-V', '--target-env', 'vulkan1.2', str(path), '-o', str(out)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return str(out)

def generated_inputs(code, private):
    code = code.replace('%out %pc %coord', '%out %pc %coord %generated_id')
    code = code.replace('OpDecorate %coord BuiltIn FragCoord', """OpDecorate %generated_id Location 8
OpDecorate %generated_id Flat
OpDecorate %coord BuiltIn FragCoord""")
    if '%pprimitive' in code:
        code = code.replace('%primitive_id = OpVariable %pprimitive Input', """%primitive_id = OpVariable %pprimitive Input
%generated_id = OpVariable %pprimitive Input""")
    else:
        code = code.replace('%batch_instance = OpVariable %pbi Input', """%batch_instance = OpVariable %pbi Input
%generated_id = OpVariable %pbi Input""")
    if private:
        code = code.replace('%primitive = OpLoad %uint %primitive_id', '%primitive = OpLoad %uint %generated_id')
    return code
vs_input = p.spirv(b.batch_vertex(p.VS_TRI), str(b.art / 'input_vs.spv'), str(b.art))
for order_index, order in enumerate((triangles, triangles[::-1])):
    gs = geometry_shader(order)
    vs_direct = p.spirv(b.batch_vertex(b.vertex_shader(order)), str(b.art / 'direct_vs.spv'), str(b.art))
    mask_direct = p.spirv(b.batch_shader(b.storage_shader(True, False, mask_only=True), primitives=2, mask_only=True), str(b.art / 'direct_mask.spv'), str(b.art))
    gf_direct = p.spirv(b.batch_shader(b.storage_shader(False, False), primitives=2), str(b.art / 'direct_guest.spv'), str(b.art))
    mask_code = generated_inputs(b.batch_shader(b.storage_shader(True, False, mask_only=True), primitives=2, mask_only=True), True)
    guest_code = generated_inputs(b.batch_shader(b.storage_shader(False, False), primitives=2), True)
    guest_code = guest_code.replace('%color = OpCompositeConstruct %v4 %flag %marker %flag %one', """%original_pid = OpLoad %uint %primitive_id
%original_pid_f = OpConvertUToF %float %original_pid
%color = OpCompositeConstruct %v4 %flag %marker %original_pid_f %one""")
    gf = p.spirv(guest_code, str(b.art / 'generated_guest.spv'), str(b.art))
    for msaa in (False, True):
        tag = f'o{order_index}_m{int(msaa)}'
        b.mask = mask_direct
        reference = b.run(vs_direct, gf_direct, tag + '_direct', True, msaa, False, True, {'VKH_BATCH_MASK': '1', 'VKH_PROBE_VERTICES': '6'})
        b.mask = p.spirv(mask_code, str(b.art / 'generated_mask.spv'), str(b.art))
        opts = {'VKH_BATCH_MASK': '1', 'VKH_GS': gs, 'VKH_GS_HELPER': '1', 'VKH_MASK_PRIMITIVES': '2'}
        actual = b.run(vs_input, gf, tag + '_gs', True, msaa, False, True, opts)
        assert reference[1] == actual[1] == 69
        assert all(((r[0], r[1], r[3]) == (v[0], v[1], v[3]) for r, v in zip(reference[2], actual[2])))
        assert all((not hit or actual[2][i][2] == 0 for i, hit in enumerate(actual[0])))
        bad_mask = generated_inputs(b.batch_shader(b.storage_shader(True, False, mask_only=True), primitives=2, mask_only=True), False)
        bad_guest = generated_inputs(b.batch_shader(b.storage_shader(False, False), primitives=2), False)
        b.mask = p.spirv(bad_mask, str(b.art / 'bad_mask.spv'), str(b.art))
        bad_gf = p.spirv(bad_guest, str(b.art / 'bad_guest.spv'), str(b.art))
        bad = b.run(vs_input, bad_gf, tag + '_negative_shared_id', True, msaa, False, True, opts)
        diff = sum((r[0] != v[0] for r, v in zip(actual[2], bad[2])))
        assert diff > 0
        print(tag, 'PASS private emitted IDs; guest PrimitiveId preserved; shared-ID negative=', diff, flush=True)
gs = geometry_shader(triangles, effects=True)
rf = p.spirv(generated_inputs(b.reference_shader(b.legacy_shader(False, False)), False), str(b.art / 'effects_reference.spv'), str(b.art))
b.mask = p.spirv(mask_code, str(b.art / 'effects_mask.spv'), str(b.art))
opts = {'VKH_BATCH_MASK': '1', 'VKH_GS': gs, 'VKH_GS_HELPER': '1', 'VKH_MASK_PRIMITIVES': '2', 'VKH_GS_EFFECTS': '1'}
ref = b.run(vs_input, rf, 'effects_ref', False, False, False, False, opts)
actual = b.run(vs_input, gf, 'effects_batch', True, False, False, True, opts)
a = struct.unpack_from('<I', (b.art / 'effects_ref.heap').read_bytes(), 16)[0]
z = struct.unpack_from('<I', (b.art / 'effects_batch.heap').read_bytes(), 16)[0]
assert (a, z) == (1, 2), (a, z)
print('Unsupported case reproduced: GS atomic effects reference=', a, 'two-pass=', z, flush=True)
print('PASS 4 pure-GS comparisons and four shared-ID negative controls; side-effecting GS remains unsupported. Validation active, zero messages.', flush=True)
