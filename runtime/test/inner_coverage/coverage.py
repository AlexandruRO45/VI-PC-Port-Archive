"""Synthetic shaders and readback helpers for the bounded InnerCoverage mask test.

These shaders are hand-authored. No Xbox ABI or general shader rewriting is inferred.
"""
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
import ps_oracle as p
from harness_result import check_harness_result
root = Path(os.environ['LEONIDA_INNER_WORK'])
exe = os.environ['LEONIDA_INNER_HARNESS']
compiler = os.environ['LEONIDA_INNER_GLSLANG']
art = root / 'initial'
art.mkdir(parents=True, exist_ok=True)
mask = None
base = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out %pc
OpExecutionMode %main OriginUpperLeft
OpExecutionMode %main EarlyFragmentTests
OpDecorate %out Location 0
OpDecorate %S Block
OpMemberDecorate %S 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%v4 = OpTypeVector %float 4
%S = OpTypeStruct %float
%pS = OpTypePointer PushConstant %S
%pf = OpTypePointer PushConstant %float
%po = OpTypePointer Output %v4
%pc = OpVariable %pS PushConstant
%out = OpVariable %po Output
%zero = OpConstant %uint 0
%one = OpConstant %float 1
%marker = OpConstant %float 123
%main = OpFunction %void None %fn
%e = OpLabel
%ptr = OpAccessChain %pf %pc %zero
%flag = OpLoad %float %ptr
%color = OpCompositeConstruct %v4 %flag %marker %flag %one
OpStore %out %color
OpReturn
OpFunctionEnd
"""

def shader(early, discard, depthwrite=False):
    s = base.replace('OpCapability Shader', """OpCapability Shader
OpCapability Int64
OpCapability PhysicalStorageBufferAddresses""").replace('OpMemoryModel Logical GLSL450', 'OpMemoryModel PhysicalStorageBuffer64 GLSL450')
    s = s.replace('%out %pc', '%out %pc %coord' + (' %depth' if depthwrite else ''))
    if not early:
        s = s.replace('OpExecutionMode %main EarlyFragmentTests', '')
    if depthwrite:
        s = s.replace('OpExecutionMode %main OriginUpperLeft', """OpExecutionMode %main OriginUpperLeft
OpExecutionMode %main DepthReplacing""")
    s = s.replace('OpDecorate %S Block', 'OpDecorate %coord BuiltIn FragCoord\n' + ('OpDecorate %depth BuiltIn FragDepth\n' if depthwrite else '') + 'OpDecorate %S Block\nOpMemberDecorate %S 1 Offset 128')
    s = s.replace('%S = OpTypeStruct %float', """%u64 = OpTypeInt 64 0
%bool = OpTypeBool
%S = OpTypeStruct %float %u64""")
    s = s.replace('%out = OpVariable %po Output', """%out = OpVariable %po Output
%pv4 = OpTypePointer Input %v4
%coord = OpVariable %pv4 Input
%pu64 = OpTypePointer PushConstant %u64
%physical = OpTypePointer PhysicalStorageBuffer %uint
%uint1 = OpConstant %uint 1
%relaxed = OpConstant %uint 0
%scope = OpConstant %uint 1
%depthval = OpConstant %float 0.75
""" + ("""%pdepth = OpTypePointer Output %float
%depth = OpVariable %pdepth Output
""" if depthwrite else ''))
    code = """%ap = OpAccessChain %pu64 %pc %uint1
%address = OpLoad %u64 %ap
%counter = OpConvertUToPtr %physical %address
%old = OpAtomicIAdd %uint %counter %scope %relaxed %uint1
"""
    if discard:
        code += """%position = OpLoad %v4 %coord
%x = OpCompositeExtract %float %position 0
%ix = OpConvertFToU %uint %x
%parity = OpBitwiseAnd %uint %ix %uint1
%kill = OpIEqual %bool %parity %zero
OpSelectionMerge %merge None
OpBranchConditional %kill %killed %merge
%killed = OpLabel
OpKill
%merge = OpLabel
"""
    if depthwrite:
        code += 'OpStore %depth %depthval\n'
    return s.replace('OpStore %out %color', code + 'OpStore %out %color')
legacy_shader = shader

def storage_shader(early, discard, depthwrite=False, mask_only=False):
    s = legacy_shader(early, discard, depthwrite)
    s = s.replace('%S = OpTypeStruct %float %u64', '%S = OpTypeStruct %float %uint %u64')
    s = s.replace('OpMemberDecorate %S 1 Offset 128', """OpMemberDecorate %S 1 Offset 4
OpMemberDecorate %S 2 Offset 128""")
    s = s.replace('%pu64 = OpTypePointer PushConstant %u64', """%pu64 = OpTypePointer PushConstant %u64
%pu = OpTypePointer PushConstant %uint""")
    s = s.replace('%depthval = OpConstant %float 0.75', """%depthval = OpConstant %float 0.75
%uint2 = OpConstant %uint 2
%uint4 = OpConstant %uint 4
%width = OpConstant %uint 8
%off4096 = OpConstant %u64 4096
%float0 = OpConstant %float 0""")
    s = s.replace('%ap = OpAccessChain %pu64 %pc %uint1', '%ap = OpAccessChain %pu64 %pc %uint2')
    s = s.replace('%flag = OpLoad %float %ptr', """%mask_ap = OpAccessChain %pu64 %pc %uint2
%mask_base = OpLoad %u64 %mask_ap
%mask_pos = OpLoad %v4 %coord
%mask_xf = OpCompositeExtract %float %mask_pos 0
%mask_yf = OpCompositeExtract %float %mask_pos 1
%mask_x = OpConvertFToU %uint %mask_xf
%mask_y = OpConvertFToU %uint %mask_yf
%mask_row = OpIMul %uint %mask_y %width
%mask_index = OpIAdd %uint %mask_row %mask_x
%mask_offset = OpIMul %uint %mask_index %uint4
%mask_offset64 = OpUConvert %u64 %mask_offset
%mask_data = OpIAdd %u64 %mask_base %off4096
%mask_address = OpIAdd %u64 %mask_data %mask_offset64
%mask_ptr = OpConvertUToPtr %physical %mask_address
%mask_value = OpLoad %uint %mask_ptr Aligned 4
%tag_ptr = OpAccessChain %pu %pc %uint1
%tag = OpLoad %uint %tag_ptr
%inside = OpIEqual %bool %mask_value %tag
%flag = OpSelect %float %inside %one %float0""")
    if mask_only:
        s = s.replace('%old = OpAtomicIAdd %uint %counter %scope %relaxed %uint1', '%old = OpAtomicExchange %uint %mask_ptr %scope %relaxed %tag')
    return s

def batch_shader(code, width=8, height=8, primitives=1, base_instance=0, mask_only=False, collapse=False):
    code = code.replace('OpCapability Shader', 'OpCapability Shader\nOpCapability Geometry')
    code = code.replace('%out %pc %coord', '%out %pc %coord %primitive_id %batch_instance')
    code = code.replace('OpDecorate %coord BuiltIn FragCoord', """OpDecorate %primitive_id BuiltIn PrimitiveId
OpDecorate %primitive_id Flat
OpDecorate %batch_instance Location 7
OpDecorate %batch_instance Flat
OpDecorate %coord BuiltIn FragCoord""")
    code = code.replace('%pv4 = OpTypePointer Input %v4', """%pprimitive = OpTypePointer Input %uint
%primitive_id = OpVariable %pprimitive Input
%batch_instance = OpVariable %pprimitive Input
%pv4 = OpTypePointer Input %v4""")
    code = code.replace('%width = OpConstant %uint 8', f'%width = OpConstant %uint {width}\n%pixel_count = OpConstant %uint {width * height}\n%triangle_count = OpConstant %uint {primitives}\n%base_instance = OpConstant %uint {base_instance}\n%five = OpConstant %uint 5\n%thirtyone = OpConstant %uint 31')
    load = '%primitive = OpCopyObject %uint %zero' if collapse else '%primitive = OpLoad %uint %primitive_id'
    address = load + """
%instance_raw = OpLoad %uint %batch_instance
%instance_local = OpISub %uint %instance_raw %base_instance
%instance_offset = OpIMul %uint %instance_local %triangle_count
%global_primitive = OpIAdd %uint %instance_offset %primitive
%primitive_base = OpIMul %uint %global_primitive %pixel_count
%global_index = OpIAdd %uint %primitive_base %mask_index
%word_index = OpShiftRightLogical %uint %global_index %five
%bit_index = OpBitwiseAnd %uint %global_index %thirtyone
%mask_bit = OpShiftLeftLogical %uint %uint1 %bit_index
%mask_offset = OpIMul %uint %word_index %uint4"""
    code = code.replace('%mask_offset = OpIMul %uint %mask_index %uint4', address)
    code = code.replace('%inside = OpIEqual %bool %mask_value %tag', """%masked = OpBitwiseAnd %uint %mask_value %mask_bit
%inside = OpINotEqual %bool %masked %zero""")
    if mask_only:
        code = code.replace('%old = OpAtomicExchange %uint %mask_ptr %scope %relaxed %tag', '%old = OpAtomicOr %uint %mask_ptr %scope %relaxed %mask_bit')
    return code

def batch_vertex(code):
    import re
    existing = 'OpDecorate %iid BuiltIn InstanceIndex' in code
    variable = '%iid' if existing else '%batch_iid'
    code = code.replace('OpEntryPoint Vertex %main "main"', 'OpEntryPoint Vertex %main "main" %batch_out' + ('' if existing else ' %batch_iid'))
    decorations = 'OpDecorate %batch_out Location 7\n' + ('' if existing else 'OpDecorate %batch_iid BuiltIn InstanceIndex\n')
    code = code.replace('OpDecorate %pos BuiltIn Position', 'OpDecorate %pos BuiltIn Position\n' + decorations)
    declarations = """%bu = OpTypeInt 32 0
%bpo = OpTypePointer Output %bu
%batch_out = OpVariable %bpo Output
"""
    if not existing:
        declarations += """%bpi = OpTypePointer Input %bu
%batch_iid = OpVariable %bpi Input
"""
    code = code.replace('%main = OpFunction', declarations + '%main = OpFunction')
    load = ("""%batch_instance_s = OpLoad %i %iid
%batch_instance_u = OpBitcast %bu %batch_instance_s
""" if existing else '%batch_instance_u = OpLoad %bu %batch_iid\n') + 'OpStore %batch_out %batch_instance_u\n'
    code = code.replace('OpStore %pos ', load + 'OpStore %pos ', 1)
    if '%uint = OpTypeInt 32 0' in code:
        code = code.replace('%bu = OpTypeInt 32 0\n', '').replace('%bu', '%uint')
    return code

def reference_shader(code):
    if '%coord' in code:
        code = code.replace('%out %pc %coord', '%out %pc %coord %batch_instance')
    else:
        code = code.replace('%out %pc', '%out %pc %batch_instance')
    code = code.replace('OpDecorate %S Block', """OpDecorate %batch_instance Location 7
OpDecorate %batch_instance Flat
OpDecorate %S Block""")
    return code.replace('%S = OpTypeStruct', """%pbi = OpTypePointer Input %uint
%batch_instance = OpVariable %pbi Input
%S = OpTypeStruct""")

def vertex_shader(triangles):
    flat = [point for tri in triangles for point in tri]
    declarations = []
    names = []
    for i, (x, y) in enumerate(flat):
        declarations.extend([f'%x{i} = OpConstant %f {x}', f'%y{i} = OpConstant %f {y}', f'%point{i} = OpConstantComposite %v4 %x{i} %y{i} %z %one'])
        names.append(f'%point{i}')
    return """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %id %pos
OpDecorate %id BuiltIn VertexIndex
OpDecorate %pos BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%f = OpTypeFloat 32
%i = OpTypeInt 32 1
%v4 = OpTypeVector %f 4
%z = OpConstant %f 0
%one = OpConstant %f 1
%n = OpConstant %i """ + str(len(flat)) + """
%arr = OpTypeArray %v4 %n
%pa = OpTypePointer Function %arr
%pv = OpTypePointer Function %v4
%pi = OpTypePointer Input %i
%po = OpTypePointer Output %v4
%id = OpVariable %pi Input
%pos = OpVariable %po Output
""" + '\n'.join(declarations) + '\n%points = OpConstantComposite %arr ' + ' '.join(names) + """
%main = OpFunction %void None %fn
%e = OpLabel
%a = OpVariable %pa Function
OpStore %a %points
%idx = OpLoad %i %id
%p = OpAccessChain %pv %a %idx
%v = OpLoad %v4 %p
OpStore %pos %v
OpReturn
OpFunctionEnd
"""

def instanced_vs(triangles, base_instance):
    s = vertex_shader(triangles)
    s = s.replace('%id %pos', '%id %pos %iid').replace('OpDecorate %id BuiltIn VertexIndex', """OpDecorate %iid BuiltIn InstanceIndex
OpDecorate %id BuiltIn VertexIndex""")
    s = s.replace('%id = OpVariable %pi Input', '%id = OpVariable %pi Input\n%iid = OpVariable %pi Input')
    s = s.replace('%main = OpFunction', '%stride = OpConstant %i 3\n%base_instance = OpConstant %i ' + str(base_instance) + '\n%main = OpFunction')
    s = s.replace('%idx = OpLoad %i %id', """%vertex = OpLoad %i %id
%instance_raw = OpLoad %i %iid
%instance = OpISub %i %instance_raw %base_instance
%offset = OpIMul %i %instance %stride
%idx = OpIAdd %i %offset %vertex""")
    return s

def derivative(code, helper):
    code = code.replace('OpCapability Shader', 'OpCapability Shader\nOpCapability DerivativeControl')
    if helper:
        code = code.replace('%out %pc %coord', '%out %pc %coord %helper')
        code = code.replace('OpDecorate %coord BuiltIn FragCoord', """OpDecorate %helper BuiltIn HelperInvocation
OpDecorate %coord BuiltIn FragCoord""")
        code = code.replace('%pv4 = OpTypePointer Input %v4', """%ph = OpTypePointer Input %bool
%helper = OpVariable %ph Input
%pv4 = OpTypePointer Input %v4""")
    code = code.replace('%depthval = OpConstant %float 0.75', """%hzero = OpConstant %float 0
%depthval = OpConstant %float 0.75""")
    if helper:
        deriv = """%h = OpLoad %bool %helper
%hf = OpSelect %float %h %one %hzero
%dx = OpDPdxFine %float %hf
%dy = OpDPdyFine %float %hf
"""
    else:
        deriv = """%dp = OpLoad %v4 %coord
%px = OpCompositeExtract %float %dp 0
%py = OpCompositeExtract %float %dp 1
%dx = OpDPdxFine %float %px
%dy = OpDPdyFine %float %py
"""
    return code.replace('%color = OpCompositeConstruct %v4 %flag %marker %flag %one', deriv + '%color = OpCompositeConstruct %v4 %flag %dx %dy %one')

def sample_shader(code, discard):
    code = code.replace('OpCapability Shader', 'OpCapability Shader\nOpCapability SampleRateShading')
    code = code.replace('%out %pc %coord', '%out %pc %coord %sample')
    code = code.replace('OpDecorate %coord BuiltIn FragCoord', """OpDecorate %sample BuiltIn SampleId
OpDecorate %sample Flat
OpDecorate %coord BuiltIn FragCoord""")
    code = code.replace('%pv4 = OpTypePointer Input %v4', """%psample = OpTypePointer Input %uint
%sample = OpVariable %psample Input
%pv4 = OpTypePointer Input %v4""")
    code = code.replace('%depthval = OpConstant %float 0.75', """%hist64 = OpConstant %u64 64
%stride4 = OpConstant %uint 4
%depthval = OpConstant %float 0.75""")
    code = code.replace('%color = OpCompositeConstruct %v4 %flag %marker %flag %one', """%sample_value = OpLoad %uint %sample
%sample_float = OpConvertUToF %float %sample_value
%color = OpCompositeConstruct %v4 %flag %marker %sample_float %one""")
    code = code.replace('%old = OpAtomicIAdd %uint %counter %scope %relaxed %uint1', """%old = OpAtomicIAdd %uint %counter %scope %relaxed %uint1
%hist_offset = OpIMul %uint %sample_value %stride4
%hist_offset64 = OpUConvert %u64 %hist_offset
%hist_base = OpIAdd %u64 %address %hist64
%hist_address = OpIAdd %u64 %hist_base %hist_offset64
%hist_ptr = OpConvertUToPtr %physical %hist_address
%hist_old = OpAtomicIAdd %uint %hist_ptr %scope %relaxed %uint1""")
    if discard:
        code = code.replace('%parity = OpBitwiseAnd %uint %ix %uint1', """%mixed = OpIAdd %uint %ix %sample_value
%parity = OpBitwiseAnd %uint %mixed %uint1""")
    return code

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

def run(vs, fs, label, combine, msaa, depth, isolate, extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('VKH_')}
    env['VKH_STORAGE_MASK'] = '1'
    env['PROBE_VALIDATION'] = '1'
    env['PROBE_SYNC_VALIDATION'] = '1'
    for key in ('VKH_COMBINE', 'VKH_PROBE_MSAA4', 'VKH_PROBE_DEPTH', 'VKH_ISOLATE_PREPASS', 'VKH_MASK_PS', 'VKH_BAD_MASK', 'VKH_SKIP_MASK', 'VKH_NO_MASK_RESET', 'VKH_PROBE_VERTICES'):
        env.pop(key, None)
    env['VKH_CONSERVATIVE'] = 'over'
    env['VKH_HEAP'] = '0=0'
    env['VKH_HEAP_DUMP'] = str(art / (label + '.heap'))
    if combine:
        env['VKH_COMBINE'] = '1'
    if msaa:
        env['VKH_PROBE_MSAA4'] = '1'
    if depth:
        env['VKH_PROBE_DEPTH'] = '1'
    if isolate:
        env.update(VKH_ISOLATE_PREPASS='1', VKH_MASK_PS=mask)
    if extra:
        env.update(extra)
    raw = art / (label + '.raw')
    r = subprocess.run([exe, vs, fs, str(raw), '8', '8'], env=env, capture_output=True, text=True, timeout=30)
    check_harness_result(r)
    assert 'PROBE validation enabled' in r.stderr
    assert 'PROBE sync validation enabled' in r.stderr
    words = p.pixels(str(raw), (8, 8))
    values = [struct.unpack('<4f', struct.pack('<4I', *v)) for v in words]
    hits = [v[1] == 123 for v in values]
    count = struct.unpack_from('<I', Path(env['VKH_HEAP_DUMP']).read_bytes())[0]
    return (hits, count, values)
