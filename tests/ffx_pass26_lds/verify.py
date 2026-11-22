"""Check the source-owned LDS fixture or the optional decoded package on one adapter."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=Path, required=True)
parser.add_argument("--backend", choices=("warp","hardware"), default="warp")
parser.add_argument("--package", nargs=2, type=Path, metavar=("PRODUCER","LLVM_DLL"))
args = parser.parse_args()
if not __debug__:
    raise RuntimeError("Run without -O to enable comparison checks.")

def fixture(cases, inclusive_x=False, stores=False):
    values = ",\n".join(f"uint4({w},{h},{x},{y})" for w,h,x,y in cases)
    template = Path(__file__).with_name("fixture.hlsl").read_text()
    return (f"#define INCLUSIVE_X {int(inclusive_x)}\n#define STORES {int(stores)}\n"
            f"static const uint4 inputs[{len(cases)}] = {{{values}}};\n" + template)

generate = fixture
if args.package:
    import package_gate
    package_gate.configure(Path(__file__).resolve().parents[2], *args.package)
    generate = package_gate.generate

backend = args.backend
def oracle(w,h,x,y):
    # Wave zero covers a 16x2 cell rectangle. When data exist in all 32
    # cells, the output endpoints are (0,0) and (31,3) before rescaling.
    count=min(16,w//2)*min(2,h//2)
    def extent(last,scale):
        return 3+round((last+0.5)*scale/256-0.5)-round(0.5*scale/256-0.5)
    ex,ey=extent(31,x),extent(3,y)
    return count,ex,ey,int(count==32 and ex<33 and ey<7)

cases=[(w,h,x,y) for w in (8,30,31,32) for h in (2,3,4)
       for x in (224,240,243,244,256) for y in (256,384)]
with tempfile.TemporaryDirectory() as tmp:
    shader=Path(tmp)/"gate.hlsl";shader.write_text(generate(cases))
    out=subprocess.check_output([args.host.resolve(),shader,
                                 backend,str(len(cases))],text=True)
    lines=out.splitlines();print(lines[0])
    assert len(lines[1:])==len(cases)
    for case,line in zip(cases,lines[1:]):
        actual=tuple(map(int,line.split()))
        expected=oracle(*case)
        # Inactive lane data on the real device is unspecified. Do not use
        # subgroup footprint observations as evidence on partial waves.
        assert actual[0]==expected[0] and actual[3]==expected[3], (case,actual,expected)
        if actual[0]==32: assert actual==expected,(case,actual,expected)
    shader.write_text(generate(cases, inclusive_x=True))
    run = subprocess.check_output([args.host.resolve(), shader,
                                   backend, str(len(cases))], text=True)
    observations = run.splitlines()[1:]
    assert len(observations) == len(cases)
    changed = [case for case, line in zip(cases, observations)
               if int(line.split()[3]) != oracle(*case)[3]]
    assert changed == [(32, 4, 244, 256)], (backend, changed)
    print(f"{backend}: inclusive-X control changed only {changed[0]}")
    store_cases = [(32,4,243,256), (32,4,244,256), (31,4,243,256)]
    shader.write_text(generate(store_cases, stores=True))
    run = subprocess.check_output([args.host.resolve(), shader,
                                  backend, str(len(store_cases)), "576"], text=True)
    observations = run.splitlines()[1:]
    assert len(observations) == len(store_cases)
    for case, line in zip(store_cases, observations):
        words = list(map(int,line.split()))
        expected = list(range(1,577)) if oracle(*case)[3] else [0]*576
        assert words == expected, (backend, case, "LDS word mismatch")
    print(f"{backend}: {len(cases)} gate cases and {len(store_cases)} tagged-store cases passed")
