import os, re, struct, subprocess
from pathlib import Path
import coverage as b
p = b.p
root = b.root
compiler = b.compiler
b.art = root / 'scratch_reuse'
b.art.mkdir(parents=True, exist_ok=True)

def scaled_vertex(code):
    ft = '%float' if '%float = OpTypeFloat 32' in code else '%f'
    ut = '%uint' if '%uint = OpTypeInt 32 0' in code else '%bu'
    code = code.replace('OpEntryPoint Vertex %main "main"', 'OpEntryPoint Vertex %main "main" %scale_pc')
    code = code.replace('OpDecorate %pos BuiltIn Position', """OpDecorate %pos BuiltIn Position
OpDecorate %scale_struct Block
OpMemberDecorate %scale_struct 0 Offset 12""")
    code = code.replace('%main = OpFunction', f'%scale_struct = OpTypeStruct {ft}\n%scale_ps = OpTypePointer PushConstant %scale_struct\n%scale_pf = OpTypePointer PushConstant {ft}\n%scale_pc = OpVariable %scale_ps PushConstant\n%scale_zero = OpConstant {ut} 0\n%main = OpFunction')
    match = re.search('OpStore %pos (%\\w+)', code)
    assert match
    value = match.group(1)
    code = code.replace(match.group(0), f'%scale_ptr = OpAccessChain %scale_pf %scale_pc %scale_zero\n%scale = OpLoad {ft} %scale_ptr\n%old_x = OpCompositeExtract {ft} {value} 0\n%old_y = OpCompositeExtract {ft} {value} 1\n%new_x = OpFMul {ft} %old_x %scale\n%new_y = OpFMul {ft} %old_y %scale\n%scale_temp = OpCompositeInsert %v4 %new_x {value} 0\n%scale_pos = OpCompositeInsert %v4 %new_y %scale_temp 1\nOpStore %pos %scale_pos', 1)
    return code
if __name__ == '__main__':
    vs = p.spirv(scaled_vertex(b.batch_vertex(p.VS_TRI.replace('OpConstant %float -1', 'OpConstant %float -0.65').replace('OpConstant %float 3', 'OpConstant %float 0.65'))), str(b.art / 'vs.spv'), str(b.art))
    b.mask = p.spirv(b.batch_shader(b.storage_shader(True, False, mask_only=True), mask_only=True), str(b.art / 'mask.spv'), str(b.art))
    rf = p.spirv(b.reference_shader(b.legacy_shader(False, False)), str(b.art / 'ref.spv'), str(b.art))
    gf = p.spirv(b.batch_shader(b.storage_shader(False, False)), str(b.art / 'guest.spv'), str(b.art))
    for batches in (2, 3, 5):
        for msaa in (False, True):
            tag = f'b{batches}_m{int(msaa)}'
            opts = {'VKH_BATCH_MASK': '1', 'VKH_REPLAY_BATCHES': str(batches), 'VKH_SCALE_BATCHES': '1', 'VKH_HEAP': '0=0,256=0x12345678'}
            ref = b.run(vs, rf, tag + '_ref', False, msaa, False, False, opts)
            clean = b.run(vs, gf, tag + '_clean', True, msaa, False, True, opts)
            dirty = b.run(vs, gf, tag + '_dirty', True, msaa, False, True, opts | {'VKH_DIRTY_SCRATCH': '1'})
            assert clean[1] == dirty[1] == ref[1] and clean[0] == ref[0] and (clean[2] == dirty[2])
            assert struct.unpack_from('<I', (b.art / (tag + '_dirty.heap')).read_bytes(), 256)[0] == 305419896
            bad = b.run(vs, gf, tag + '_negative', True, msaa, False, True, opts | {'VKH_DIRTY_SCRATCH': '1', 'VKH_SKIP_SCRATCH_CLEAR': '1'})
            diff = sum((a != v for a, v in zip(clean[2], bad[2])))
            assert diff > 0
            print(tag, 'PASS effects=', clean[1], 'dirty/no-clear negative=', diff, 'pixels', flush=True)
    opts = {'VKH_BATCH_MASK': '1', 'VKH_REPLAY_BATCHES': '2', 'VKH_SCALE_BATCHES': '1'}
    clean = b.run(vs, gf, 'history_clean', True, False, False, True, opts)
    stale = b.run(vs, gf, 'history_negative', True, False, False, True, opts | {'VKH_SKIP_SCRATCH_CLEAR': '1'})
    diff = sum((a != v for a, v in zip(clean[2], stale[2])))
    assert diff > 0
    print('Stale prior-batch negative detected:', diff, 'pixels', flush=True)
    print('PASS 6 multi-batch reuse comparisons; GPU-only clear, preserved counter/sentinel, validation active', flush=True)
