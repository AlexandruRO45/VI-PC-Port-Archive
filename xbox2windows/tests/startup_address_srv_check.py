"""Publish two placed R8 textures by address, render their texels, and test rejected updates."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_stream_draw_check import gfx2hlsl, pixel, vertex


def draw_texture(directory, consumer, backend, hlsl, modes, view_kind="tex2d"):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_D3D12_DEBUG="0", X2W_D3D12_DEVICE_NOOP="1", X2W_SHADER_CACHE=str(cache), **modes)
    args = [consumer, vs, ps, root]
    subprocess.run(args, env=env, check=True)
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    manifest, = cache.glob("*.txt")
    with manifest.open("a", encoding="ascii") as file:
        file.write(f"\nsrv 0 {view_kind} table 0 0\n")
    source = manifest.with_suffix(".ps.hlsl")
    source.write_text(hlsl)
    subprocess.run([env["X2W_FXC"], "/nologo", "/T", "ps_5_0", "/E", "main", "/Fo",
                    manifest.with_suffix(".ps.cso"), source], check=True)
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if backend == "hardware" and result.returncode == 77:
        return None
    if result.returncode:
        raise RuntimeError(f"consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
    return output.read_bytes()


def check(directory, consumer, backend):
    hlsl = ("Texture2D<float> t : register(t0);\n"
            "float4 main(float4 p : SV_Position) : SV_Target {\n"
            "  return float4(t.Load(int3(uint2(p.xy) % 2, 0)), 0, 0, 1);\n}\n")
    actual = draw_texture(directory, consumer, backend, hlsl, {"X2W_TEST_ADDRESS_SRV": "1"})
    if actual is None:
        return 77
    plane_at_x = (0, 1, 0, 0, 1, 1, 1, 0)
    expected = bytes(c for y in range(4) for x, plane in enumerate(plane_at_x)
                     for c in (16 + plane * 128 + (y % 2) * 32 + (x % 2) * 7, 0, 0, 255))
    if actual != expected:
        raise AssertionError(f"actual {actual.hex()} expected {expected.hex()}")
    print(f"{backend}: ordinary view, alternating addresses, zero/interior/mismatched updates; all 32 RGBA pixels exact")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
