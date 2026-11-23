import coverage as b
p = b.p
b.art = b.root / 'helper_counter'
b.art.mkdir(parents=True, exist_ok=True)
b.mask = p.spirv(b.batch_shader(b.storage_shader(True, False, mask_only=True), mask_only=True), str(b.art / 'mask.spv'), str(b.art))
vs = p.spirv(b.batch_vertex(p.VS_TRI.replace('OpConstant %float -1', 'OpConstant %float -0.65').replace('OpConstant %float 3', 'OpConstant %float 0.65')), str(b.art / 'vs.spv'), str(b.art))

def guarded(code):
    code = b.derivative(code, True)
    return code.replace('%old = OpAtomicIAdd %uint %counter %scope %relaxed %uint1', """%nonhelper_increment = OpSelect %uint %h %zero %uint1
%old = OpAtomicIAdd %uint %counter %scope %relaxed %nonhelper_increment""")
for early in (False, True):
    for msaa in (False, True):
        tag = f'e{int(early)}_m{int(msaa)}'
        rf = p.spirv(guarded(b.reference_shader(b.legacy_shader(early, True))), str(b.art / (tag + '_ref.spv')), str(b.art))
        gf = p.spirv(guarded(b.batch_shader(b.storage_shader(early, True))), str(b.art / (tag + '_guest.spv')), str(b.art))
        ref = b.run(vs, rf, tag + '_ref', False, msaa, False, False, {'VKH_BATCH_MASK': '1'})
        actual = b.run(vs, gf, tag + '_mask', True, msaa, False, True, {'VKH_BATCH_MASK': '1'})
        assert ref[1] == actual[1] == 26
        assert all((a[1:] == v[1:] for a, v in zip(ref[2], actual[2])))
        print(tag, 'PASS: explicit helper guard effects=26', flush=True)
