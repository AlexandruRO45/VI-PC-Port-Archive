from pathlib import Path
import re, statistics, struct, subprocess, os
import coverage as b
root = b.root
art = root / 'batch-timing'
art.mkdir(exist_ok=True)
p = b.p
exe = b.exe
for size in (int(os.environ['PROBE_BENCH_SIZE']),) if 'PROBE_BENCH_SIZE' in os.environ else (8, 32):
    v = p.VS_TRI.replace('OpConstant %float -1', 'OpConstant %float -0.65').replace('OpConstant %float 3', 'OpConstant %float 0.65')
    v = v.replace('%main = OpFunction', '%stride = OpConstant %int 3\n%main = OpFunction').replace('%i = OpLoad %int %vid', """%raw_index = OpLoad %int %vid
%i = OpSMod %int %raw_index %stride""")
    vs = p.spirv(b.batch_vertex(v), str(art / f'vs{size}.spv'), str(art))
    ref = p.spirv(b.reference_shader(b.legacy_shader(False, False)), str(art / f'ref{size}.spv'), str(art))
    serial = p.spirv(b.reference_shader(b.storage_shader(False, False).replace('%width = OpConstant %uint 8', f'%width = OpConstant %uint {size}')), str(art / f'serial{size}.spv'), str(art))
    sm = p.spirv(b.reference_shader(b.storage_shader(True, False, mask_only=True).replace('%width = OpConstant %uint 8', f'%width = OpConstant %uint {size}')), str(art / f'smask{size}.spv'), str(art))
    for triangles in (int(os.environ['PROBE_BENCH_TRIANGLES']),) if 'PROBE_BENCH_TRIANGLES' in os.environ else (1, 16, 64):
        batch = p.spirv(b.batch_shader(b.storage_shader(False, False), width=size, height=size, primitives=triangles), str(art / 'batch.spv'), str(art))
        bm = p.spirv(b.batch_shader(b.storage_shader(True, False, mask_only=True), width=size, height=size, primitives=triangles, mask_only=True), str(art / 'bmask.spv'), str(art))
        times = {mode: [] for mode in ('reference', 'serial', 'batch')}
        for repeat in range(6):
            for mode, fs, mask in (('reference', ref, None), ('serial', serial, sm), ('batch', batch, bm)):
                env = {k: v for k, v in os.environ.items() if not k.startswith('VKH_')}
                env.update(PROBE_VALIDATION='1', PROBE_SYNC_VALIDATION='1')
                env.update(VKH_CONSERVATIVE='over', VKH_STORAGE_MASK='1', VKH_HEAP='0=0', VKH_HEAP_DUMP=str(art / 'counter.heap'), VKH_VERTICES=str(triangles * 3), VKH_GPU_TIMING='1')
                for k in ('VKH_COMBINE', 'VKH_PROBE_VERTICES', 'VKH_ISOLATE_PREPASS', 'VKH_MASK_PS', 'VKH_BATCH_MASK'):
                    env.pop(k, None)
                if mode != 'reference':
                    env.update(VKH_COMBINE='1', VKH_PROBE_VERTICES=str(triangles * 3), VKH_ISOLATE_PREPASS='1', VKH_MASK_PS=mask)
                if mode == 'batch':
                    env['VKH_BATCH_MASK'] = '1'
                r = subprocess.run([exe, vs, fs, str(art / 'pixels.raw'), str(size), str(size)], env=env, capture_output=True, text=True, timeout=30)
                if r.returncode or 'PROBE_VALIDATION:' in r.stderr:
                    raise RuntimeError(r.stderr)
                assert 'PROBE validation enabled' in r.stderr
                count = struct.unpack_from('<I', (art / 'counter.heap').read_bytes())[0]
                assert count == (26 if size == 8 else 274) * triangles, (size, triangles, mode, count)
                ns = float(re.search('PROBE_GPU_NS ([0-9.]+)', r.stderr).group(1))
                if repeat:
                    times[mode].append(ns / 1000)
        med = {mode: statistics.median(v) for mode, v in times.items()}
        print(f"{size}x{size} N={triangles}: reference {med['reference']:.3f}us, serial {med['serial']:.3f}us, batch {med['batch']:.3f}us; speedup {med['serial'] / med['batch']:.2f}x, batch/reference {med['batch'] / med['reference']:.2f}x; bitmask {(size * size * triangles + 31) // 32 * 4} bytes", flush=True)
        print('  batch range', min(times['batch']), max(times['batch']), 'us', flush=True)
print('GPU render-pass timestamps; one warmup and five samples per mode, validation active; not a production performance claim.', flush=True)
