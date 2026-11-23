import os, re, struct, subprocess
from pathlib import Path
import coverage as b
p = b.p
root = b.root
compiler = b.compiler
b.art = root / 'geometry_capture'
b.art.mkdir(parents=True, exist_ok=True)

def compile_glsl(name, source):
    path = b.art / name
    path.write_text(source, encoding='utf-8', newline='\n')
    out = path.with_suffix('.spv')
    r = subprocess.run([compiler, '-V', '--target-env', 'vulkan1.2', str(path), '-o', str(out)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return str(out)
generated_inputs = b.generated_inputs
header = """#version 460
#extension GL_EXT_buffer_reference : require
layout(buffer_reference,std430,buffer_reference_align=4) buffer Heap {uint value[];};
layout(push_constant) uniform PC {layout(offset=128) Heap heap;} pc;
"""
original_vs = compile_glsl('original.vert', header + """
layout(location=0) out vec4 a;
layout(location=1) out vec4 b;
layout(location=7) flat out uint instanceOut;
void main() {atomicAdd(pc.heap.value[5],1);gl_Position=vec4(0,0,0,1);a=b=vec4(0);instanceOut=gl_InstanceIndex;}
""")
replay_vs = compile_glsl('replay.vert', header + """
layout(location=7) flat out uint instanceOut;
layout(location=8) flat out uint privatePrimitive;
layout(location=9) flat out uint originalPrimitive;
void main() {
 uint i=2048+uint(gl_VertexIndex)*8;
 gl_Position=uintBitsToFloat(uvec4(pc.heap.value[i],pc.heap.value[i+1],pc.heap.value[i+2],pc.heap.value[i+3]));
 instanceOut=pc.heap.value[i+4];privatePrimitive=pc.heap.value[i+5];originalPrimitive=pc.heap.value[i+6];
}
""")

def gs(effects):
    return compile_glsl('gs' + str(int(effects)) + '.geom', header + """
layout(triangles) in;
layout(triangle_strip,max_vertices=6) out;
layout(location=0) in vec4 a[];
layout(location=1) in vec4 b[];
layout(location=7) flat in uint instanceIn[];
layout(location=7) flat out uint instanceOut;
layout(location=8) flat out uint privatePrimitive;
layout(location=9) flat out uint originalPrimitive;
const vec2 points[6]=vec2[6](vec2(-.8,-.8),vec2(.8,-.8),vec2(-.8,.8),vec2(-.4,-.4),vec2(.9,-.4),vec2(-.4,.9));
void main() {
 uint ticket=""" + ('atomicAdd(pc.heap.value[4],1)' if effects else '0u') + """;
 for(uint triangle=0;triangle<2;triangle++) {
  for(uint v=0;v<3;v++) {
   vec4 pos=vec4(points[triangle*3+v]+vec2(float(ticket)*0.1,0),0,1);
   uint id=uint(gl_PrimitiveIDIn)*2+triangle;
   uint index=2048+(id*3+v)*8;
   for(uint c=0;c<4;c++) pc.heap.value[index+c]=floatBitsToUint(pos[c]);
   pc.heap.value[index+4]=instanceIn[0];pc.heap.value[index+5]=id;pc.heap.value[index+6]=uint(gl_PrimitiveIDIn);
   gl_Position=pos;gl_PrimitiveID=gl_PrimitiveIDIn;
   instanceOut=instanceIn[0];privatePrimitive=id;originalPrimitive=uint(gl_PrimitiveIDIn);
   EmitVertex();
  }
  EndPrimitive();
 }
}
""")

def original_id_input(code):
    code = code.replace('%out %pc %coord', '%out %pc %coord %original_id')
    code = code.replace('OpDecorate %coord BuiltIn FragCoord', """OpDecorate %original_id Location 9
OpDecorate %original_id Flat
OpDecorate %coord BuiltIn FragCoord""")
    pointer = '%pprimitive' if '%pprimitive' in code else '%pbi'
    declaration = '%original_id = OpVariable ' + pointer + ' Input\n'
    code = code.replace('%main = OpFunction', declaration + '%main = OpFunction')
    return code
mask_code = original_id_input(generated_inputs(b.batch_shader(b.storage_shader(True, False, mask_only=True), primitives=2, mask_only=True), True))
b.mask = p.spirv(mask_code, str(b.art / 'mask.spv'), str(b.art))
code = original_id_input(generated_inputs(b.batch_shader(b.storage_shader(False, False), primitives=2), True))
code = code.replace('%color = OpCompositeConstruct %v4 %flag %marker %flag %one', """%original_value = OpLoad %uint %original_id
%original_float = OpConvertUToF %float %original_value
%color = OpCompositeConstruct %v4 %flag %marker %original_float %one""")
gf = p.spirv(code, str(b.art / 'guest.spv'), str(b.art))
reference_fs = p.spirv(original_id_input(generated_inputs(b.reference_shader(b.legacy_shader(False, False)), False)), str(b.art / 'reference.spv'), str(b.art))
pure_gs = gs(False)
effect_gs = gs(True)
pure_vs = compile_glsl('pure.vert', header + """
layout(location=0) out vec4 a;layout(location=1) out vec4 b;layout(location=7) flat out uint instanceOut;
void main(){gl_Position=vec4(0,0,0,1);a=b=vec4(0);instanceOut=gl_InstanceIndex;}
""")
opts = {'VKH_BATCH_MASK': '1', 'VKH_GS_HELPER': '1', 'VKH_MASK_PRIMITIVES': '2', 'VKH_GS_EFFECTS': '1'}
for msaa in (False, True):
    oracle = b.run(pure_vs, gf, f'oracle_{msaa}', True, msaa, False, True, opts | {'VKH_GS': pure_gs})
    native = b.run(original_vs, reference_fs, f'native_{msaa}', False, msaa, False, False, opts | {'VKH_GS': effect_gs})
    actual = b.run(original_vs, gf, f'capture_{msaa}', True, msaa, False, True, opts | {'VKH_GS': effect_gs, 'VKH_CAPTURE_REPLAY_VS': replay_vs, 'VKH_REPLAY_VERTICES': '6'})
    assert actual[2] == oracle[2] and actual[0] == native[0] and (actual[1] == native[1] == 69)
    ref_counts = struct.unpack_from('<2I', (b.art / f'native_{msaa}.heap').read_bytes(), 16)
    actual_counts = struct.unpack_from('<2I', (b.art / f'capture_{msaa}.heap').read_bytes(), 16)
    assert ref_counts == actual_counts == (1, 3), (ref_counts, actual_counts)
    unsafe = b.run(original_vs, gf, f'unsafe_{msaa}', True, msaa, False, True, opts | {'VKH_GS': effect_gs})
    bad_counts = struct.unpack_from('<2I', (b.art / f'unsafe_{msaa}.heap').read_bytes(), 16)
    assert bad_counts == (2, 6) and unsafe[2] != actual[2]
    print('MSAA', 4 if msaa else 1, 'PASS capture once: GS=1 VS=3 PS=69; replay-twice negative GS=2 VS=6 changes geometry', flush=True)
print('PASS 2 capture/replay comparisons; validation active, zero validation messages. Bounded original synthetic GS fixture, not a general stage rewriter.', flush=True)
