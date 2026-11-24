"""Check one frame's append requests and pitch stores against independent plane records."""
import ctypes as C
from pathlib import Path
import sys


class Plane(C.Structure):
    _fields_ = [("destination", C.c_uint64), ("enabled", C.c_uint32), ("pitch", C.c_uint32)]


class Frames(C.Structure):
    _fields_ = [("count", C.c_uint32), ("width", C.c_uint32), ("height", C.c_uint32),
                ("small_width", C.c_uint32), ("small_height", C.c_uint32),
                ("reserved", C.c_uint32), ("planes", (Plane * 5) * 3)]


Observe = C.CFUNCTYPE(None, C.c_void_p, C.c_uint32)


def check(path: Path) -> None:
    lib = C.CDLL(str(path.resolve()))
    lib.frame_planes.argtypes = [C.POINTER(Frames), C.c_uint32, C.POINTER(C.c_uint32), Observe]
    lib.frame_planes.restype = None
    pitches = (C.c_uint32 * 2)(96, 32)
    for index in range(3):
        for mask in range(32):
            frames = Frames(3, 79, 7, 29, 3)
            expected = Frames()
            frame = frames.planes[index]
            for n, plane in enumerate(frame):
                plane.enabled = 7 if mask & (1 << n) else 0
                plane.pitch = 0x12345678
            C.memmove(C.byref(expected), C.byref(frames), C.sizeof(frames))
            calls = []
            wanted = []
            for n in range(5):
                if frame[n].enabled:
                    small = n in (1, 2)
                    pitch = pitches[int(small)]
                    size = pitch * (frames.small_height if small else frames.height)
                    wanted.append((C.addressof(frame[n]), size, frame[n].pitch))
                    expected.planes[index][n].pitch = pitch
            def observe(destination: int, size: int) -> None:
                # Only dereference a slot from the buffer we supplied.
                slot = next((plane for plane in frame if C.addressof(plane) == destination), None)
                calls.append((destination, size, slot.pitch if slot is not None else None))

            callback = Observe(observe)
            lib.frame_planes(C.byref(frames), index, pitches, callback)
            assert calls == wanted, (index, mask, calls, wanted)
            assert bytes(frames) == bytes(expected), (index, mask)


if __name__ == "__main__":
    for image in sys.argv[1:]:
        check(Path(image))
