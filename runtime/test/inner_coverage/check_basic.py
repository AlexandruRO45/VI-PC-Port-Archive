import coverage as b
from coverage import p, batch_shader, storage_shader, batch_vertex, reference_shader, legacy_shader
run = b.run
b.mask = p.spirv(batch_shader(storage_shader(True, False, mask_only=True), mask_only=True), str(b.art / 'mask.spv'), str(b.art))
vs = p.spirv(batch_vertex(p.VS_TRI.replace('OpConstant %float -1', 'OpConstant %float -0.65').replace('OpConstant %float 3', 'OpConstant %float 0.65')), str(b.art / 'vs.spv'), str(b.art))
for early in (True, False):
    for discard in (False, True):
        for msaa in (False, True):
            tag = f'e{int(early)}_d{int(discard)}_m{int(msaa)}'
            rf = p.spirv(reference_shader(legacy_shader(early, discard)), str(b.art / (tag + '_ref.spv')), str(b.art))
            gf = p.spirv(batch_shader(storage_shader(early, discard)), str(b.art / (tag + '_guest.spv')), str(b.art))
            extra = {'VKH_BATCH_MASK': '1'}
            ref = run(vs, rf, tag + '_ref', False, msaa, True, False, extra)
            actual = run(vs, gf, tag + '_batch', True, msaa, True, True, extra)
            assert actual[0] == ref[0] and actual[1] == ref[1] == 26
            assert all((not hit or actual[2][i][0] == float(i in (18, 19, 26)) for i, hit in enumerate(actual[0])))
            print(tag, 'PASS', flush=True)
print('PASS 8 initial batched semantic comparisons; validation active', flush=True)
