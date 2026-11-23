import os, re, struct, subprocess
from pathlib import Path
import coverage as b
p = b.p
root = b.root
compiler = b.compiler
b.art = root / 'batch_regressions'
b.art.mkdir(parents=True, exist_ok=True)
b.mask = p.spirv(b.batch_shader(b.storage_shader(True, False, mask_only=True), mask_only=True), str(b.art / 'mask.spv'), str(b.art))
vs = p.spirv(b.batch_vertex(p.VS_TRI.replace('OpConstant %float -1', 'OpConstant %float -0.65').replace('OpConstant %float 3', 'OpConstant %float 0.65')), str(b.art / 'vs.spv'), str(b.art))
extra = {'VKH_BATCH_MASK': '1'}
checks = 0
for early in (False, True):
    for discard in (False, True):
        for depth in (False, True):
            tag = f'sample_e{int(early)}_d{int(discard)}_z{int(depth)}'
            rf = p.spirv(b.sample_shader(b.reference_shader(b.legacy_shader(early, discard)), discard), str(b.art / (tag + '_ref.spv')), str(b.art))
            gf = p.spirv(b.sample_shader(b.batch_shader(b.storage_shader(early, discard)), discard), str(b.art / (tag + '_guest.spv')), str(b.art))
            opts = extra | {'VKH_SAMPLE_SHADING': '1'}
            ref = b.run(vs, rf, tag + '_ref', False, True, depth, False, opts)
            actual = b.run(vs, gf, tag + '_batch', True, True, depth, True, opts)
            assert ref[1] == actual[1] == 104 and all((a[1:] == v[1:] for a, v in zip(ref[2], actual[2])))
            assert struct.unpack_from('<4I', (b.art / (tag + '_batch.heap')).read_bytes(), 64) == (26, 26, 26, 26)
            for i, (r, a) in enumerate(zip(ref[2], actual[2])):
                fraction = (r[3] + 1) / 2
                assert abs(a[0] - (-1 + (2 if i in (18, 19, 26) else 1) * fraction)) < 1e-06
            print(tag, 'PASS', flush=True)
            checks += 1
inspect = p.spirv(b.reference_shader(b.base), str(b.art / 'inspect.spv'), str(b.art))
for early in (False, True):
    for discard in (False, True):
        for msaa in (False, True):
            tag = f'stencil_e{int(early)}_d{int(discard)}_m{int(msaa)}'
            rf = p.spirv(b.reference_shader(b.legacy_shader(early, discard)), str(b.art / (tag + '_ref.spv')), str(b.art))
            gf = p.spirv(b.batch_shader(b.storage_shader(early, discard)), str(b.art / (tag + '_guest.spv')), str(b.art))
            opts = extra | {'VKH_USER_STENCIL': '1', 'VKH_STENCIL_PROBE_PS': inspect}
            ref = b.run(vs, rf, tag + '_ref', False, msaa, True, False, opts)
            actual = b.run(vs, gf, tag + '_batch', True, msaa, True, True, opts)
            assert actual[1] == ref[1] and actual[2] == ref[2]
            print(tag, 'PASS', flush=True)
            checks += 1
for helper in (False, True):
    for discard in (False, True):
        for msaa in (False, True):
            tag = f'deriv_h{int(helper)}_d{int(discard)}_m{int(msaa)}'
            rf = p.spirv(b.derivative(b.reference_shader(b.legacy_shader(False, discard)), helper), str(b.art / (tag + '_ref.spv')), str(b.art))
            gf = p.spirv(b.derivative(b.batch_shader(b.storage_shader(False, discard)), helper), str(b.art / (tag + '_guest.spv')), str(b.art))
            ref = b.run(vs, rf, tag + '_ref', False, msaa, False, False, extra)
            actual = b.run(vs, gf, tag + '_batch', True, msaa, False, True, extra)
            assert ref[1] == actual[1] and all((a[1:] == v[1:] for a, v in zip(ref[2], actual[2])))
            if helper:
                assert any((v[3] != -1 and (v[1] != 0 or v[2] != 0) for v in actual[2]))
            print(tag, 'PASS reference effects=', ref[1], flush=True)
            checks += 1
for early, write in ((True, False), (False, False), (False, True)):
    tag = f'far_e{int(early)}_w{int(write)}'
    far = p.spirv(b.batch_vertex(p.VS_TRI.replace('OpConstant %float -1', 'OpConstant %float -0.65').replace('OpConstant %float 3', 'OpConstant %float 0.65').replace('%f0 = OpConstant %float 0', '%f0 = OpConstant %float 0.75')), str(b.art / 'far.spv'), str(b.art))
    rc = b.reference_shader(b.legacy_shader(early, False, write))
    gc = b.batch_shader(b.storage_shader(early, False, write))
    if write:
        rc = rc.replace('%depthval = OpConstant %float 0.75', '%depthval = OpConstant %float 0.25')
        gc = gc.replace('%depthval = OpConstant %float 0.75', '%depthval = OpConstant %float 0.25')
    rf = p.spirv(rc, str(b.art / (tag + '_ref.spv')), str(b.art))
    gf = p.spirv(gc, str(b.art / (tag + '_guest.spv')), str(b.art))
    ref = b.run(far, rf, tag + '_ref', False, True, True, False, extra)
    actual = b.run(far, gf, tag + '_batch', True, True, True, True, extra)
    assert ref[0] == actual[0] and ref[1] == actual[1] == (0 if early else 26)
    assert sum(actual[0]) == (26 if write else 0)
    print(tag, 'PASS', flush=True)
    checks += 1
print('PASS', checks, 'batched regressions; validation active with no messages. Unguarded derivative/discard counter anomaly also occurs in reference.', flush=True)
