import os, re, struct, subprocess
from pathlib import Path
import coverage as b
p = b.p
root = b.root
compiler = b.compiler
b.art = root / 'batched_coverage'
b.art.mkdir(parents=True, exist_ok=True)
triangles = [[(-0.8, -0.8), (0.8, -0.8), (-0.8, 0.8)], [(-0.4, -0.4), (0.9, -0.4), (-0.4, 0.9)]]
serial_mask = p.spirv(b.reference_shader(b.storage_shader(True, False, mask_only=True)), str(b.art / 'serial_mask.spv'), str(b.art))
checks = 0
for order_index, order in enumerate((triangles, triangles[::-1])):
    vs = p.spirv(b.batch_vertex(b.vertex_shader(order)), str(b.art / f'vs{order_index}.spv'), str(b.art))
    for early in (False, True):
        for discard in (False, True):
            for msaa in (False, True):
                tag = f'o{order_index}_e{int(early)}_d{int(discard)}_m{int(msaa)}'
                sf = p.spirv(b.reference_shader(b.storage_shader(early, discard)), str(b.art / (tag + '_serial.spv')), str(b.art))
                bf = p.spirv(b.batch_shader(b.storage_shader(early, discard), primitives=2), str(b.art / (tag + '_batch.spv')), str(b.art))
                b.mask = serial_mask
                serial = b.run(vs, sf, tag + '_serial', True, msaa, False, True, {'VKH_PROBE_VERTICES': '6'})
                b.mask = p.spirv(b.batch_shader(b.storage_shader(True, False, mask_only=True), primitives=2, mask_only=True), str(b.art / 'batch_mask.spv'), str(b.art))
                actual = b.run(vs, bf, tag + '_batch', True, msaa, False, True, {'VKH_BATCH_MASK': '1', 'VKH_PROBE_VERTICES': '6'})
                assert actual[2] == serial[2] and actual[1] == serial[1] == 69, (tag, 'serial/batch mismatch')
                tail = (b.art / (tag + '_batch.heap')).read_bytes()[4096 + 16:]
                assert not any(tail), (tag, 'scratch index outside primitive range')
                print(tag, 'PASS effects=69', flush=True)
                checks += 1
    bad_mask = b.batch_shader(b.storage_shader(True, False, mask_only=True), primitives=2, mask_only=True, collapse=True)
    bad_fs = b.batch_shader(b.storage_shader(True, True), primitives=2, collapse=True)
    b.mask = p.spirv(bad_mask, str(b.art / 'bad_mask.spv'), str(b.art))
    bf = p.spirv(bad_fs, str(b.art / 'bad_guest.spv'), str(b.art))
    bad = b.run(vs, bf, f'o{order_index}_negative_collapsed', True, True, False, True, {'VKH_BATCH_MASK': '1', 'VKH_PROBE_VERTICES': '6'})
    differences = sum((a != v for a, v in zip(bad[2], actual[2])))
    assert differences > 0
    print('Collapsed primitive mask negative detected', differences, 'pixels', flush=True)
for order_index, order in enumerate((triangles, triangles[::-1])):
    for base in (0, 5, 37):
        tag = f'instance_o{order_index}_base{base}'
        vs = p.spirv(b.batch_vertex(b.instanced_vs(order, base)), str(b.art / (tag + '.spv')), str(b.art))
        sf = p.spirv(b.reference_shader(b.storage_shader(False, False)), str(b.art / 'instance_serial.spv'), str(b.art))
        b.mask = serial_mask
        serial = b.run(vs, sf, tag + '_serial', True, False, False, True, {'VKH_PROBE_INSTANCES': '2', 'VKH_PROBE_BASE_INSTANCE': str(base)})
        bf = p.spirv(b.batch_shader(b.storage_shader(False, False), base_instance=base), str(b.art / 'instance_batch.spv'), str(b.art))
        b.mask = p.spirv(b.batch_shader(b.storage_shader(True, False, mask_only=True), base_instance=base, mask_only=True), str(b.art / 'instance_mask.spv'), str(b.art))
        actual = b.run(vs, bf, tag + '_batch', True, False, False, True, {'VKH_BATCH_MASK': '1', 'VKH_PROBE_INSTANCES': '2', 'VKH_PROBE_BASE_INSTANCE': str(base)})
        assert actual[2] == serial[2] and actual[1] == serial[1] == 69, (tag, 'instance mismatch')
        assert not any((b.art / (tag + '_batch.heap')).read_bytes()[4112:]), (tag, 'instance/PrimitiveId range mismatch')
        print(tag, 'PASS', flush=True)
        checks += 1
print('PASS', checks, 'serial/batch comparisons, two negative controls; validation active, zero messages', flush=True)
