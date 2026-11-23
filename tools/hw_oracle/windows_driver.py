"""Private Windows oracle worker using the installed AMD COMGR and HIP DLLs.

The runner starts a worker for assembly or dispatch, keeping native driver failures in the child process.
COMGR types and actions follow amd/comgr/include/amd_comgr.h.in in ROCm/llvm-project.
HIP launches use the same one-pointer kernarg and output protocol as hw_oracle.c.
"""

import ctypes as ct
import os
import re
import sys
from contextlib import ExitStack
from pathlib import Path

SOURCE, LOG, EXECUTABLE = 1, 5, 8
LINK_RELOCATABLE, ASSEMBLE_SOURCE = 7, 8
HOST_TO_DEVICE, DEVICE_TO_HOST = 1, 2
BUFFER_POINTER, BUFFER_SIZE, EXTRA_END = 1, 2, 3
SENTINEL = 0xDEAD0000
MAX_WORDS = 16384


class DriverUnavailable(RuntimeError):
    """The installed driver does not expose the required API or device."""


class DriverError(RuntimeError):
    """Assembly, linking or a HIP operation failed."""


class ComgrHandle(ct.Structure):
    _fields_ = [("handle", ct.c_uint64)]


def _bind(dll, name, argtypes):
    try:
        fn = getattr(dll, name)
    except AttributeError:
        raise DriverUnavailable(f"{dll._name} does not export {name}") from None
    fn.argtypes = argtypes
    fn.restype = ct.c_int
    return fn


def _load_comgr():
    directory = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for name in ("amd_comgr_3.dll", "amd_comgr_2.dll"):
        path = directory / name
        if path.is_file():
            try:
                return ct.CDLL(str(path))
            except OSError as error:
                raise DriverUnavailable(f"cannot load {name}: {error}") from None
    raise DriverUnavailable("no AMD COMGR DLL in System32")


def _comgr_check(fn, *args):
    status = fn(*args)
    if status:
        raise DriverError(f"{fn.__name__}: COMGR status {status}")


def _comgr_create(stack, create, destroy, *args):
    handle = ComgrHandle()
    _comgr_check(create, *args, ct.byref(handle))
    stack.callback(destroy, handle)
    return handle


def _comgr_outputs(bind, data_set, kind):
    count, item = ct.c_size_t(), ComgrHandle()
    handle, size = ComgrHandle, ct.c_size_t
    _comgr_check(bind("action_data_count", handle, ct.c_int, ct.POINTER(size)), data_set, kind, ct.byref(count))
    get_item = bind("action_data_get_data", handle, ct.c_int, size, ct.POINTER(handle))
    get_bytes = bind("get_data", handle, ct.POINTER(size), ct.c_void_p)
    release = bind("release_data", handle)
    outputs = []
    for i in range(count.value):
        _comgr_check(get_item, data_set, kind, i, ct.byref(item))
        try:
            length = size()
            _comgr_check(get_bytes, item, ct.byref(length), None)
            buf = ct.create_string_buffer(length.value)
            _comgr_check(get_bytes, item, ct.byref(length), buf)
            outputs.append(buf.raw[:length.value])
        finally:
            release(item)
    return outputs


def assemble(text, arch, wave, code_object):
    """Assemble the oracle's synthetic kernel source to a loadable code object."""
    dll = _load_comgr()

    def bind(name, *args):
        return _bind(dll, "amd_comgr_" + name, args)

    handle, size = ComgrHandle, ct.c_size_t
    src = text.encode("utf-8")
    with ExitStack() as stack:
        create_set = bind("create_data_set", ct.POINTER(handle))
        destroy_set = bind("destroy_data_set", handle)
        source, reloc, output = (_comgr_create(stack, create_set, destroy_set) for _ in range(3))
        data = _comgr_create(stack, bind("create_data", ct.c_int, ct.POINTER(handle)),
                             bind("release_data", handle), SOURCE)
        _comgr_check(bind("set_data", handle, size, ct.c_char_p), data, len(src), src)
        _comgr_check(bind("set_data_name", handle, ct.c_char_p), data, b"k.s")
        _comgr_check(bind("data_set_add", handle, handle), source, data)
        info = _comgr_create(stack, bind("create_action_info", ct.POINTER(handle)),
                             bind("destroy_action_info", handle))
        target = f"amdgcn-amd-amdhsa--{arch}".encode("ascii")
        _comgr_check(bind("action_info_set_isa_name", handle, ct.c_char_p), info, target)
        _comgr_check(bind("action_info_set_logging", handle, ct.c_bool), info, True)
        options = [b"-mwavefrontsize64"] if wave == 64 else []
        if code_object:
            options.append(f"-mcode-object-version={code_object}".encode("ascii"))
        option_array = (ct.c_char_p * len(options))(*options)
        set_options = bind("action_info_set_option_list", handle, ct.POINTER(ct.c_char_p), size)
        _comgr_check(set_options, info, option_array, len(options))
        action = bind("do_action", ct.c_int, handle, handle, handle)
        for kind, incoming, outgoing in ((ASSEMBLE_SOURCE, source, reloc), (LINK_RELOCATABLE, reloc, output)):
            status = action(kind, info, incoming, outgoing)
            if status:
                logs = b"\n".join(_comgr_outputs(bind, outgoing, LOG)).replace(b"\0", b"")
                raise DriverError(logs.decode("utf-8", errors="replace").strip()
                                  or f"COMGR action {kind}: status {status}")
            _comgr_check(set_options, info, None, 0)
        code = _comgr_outputs(bind, output, EXECUTABLE)
        if len(code) != 1 or not code[0].startswith(b"\x7fELF"):
            raise DriverError("COMGR did not produce one ELF code object")
        return code[0]


def _hip_check(fn, *args):
    status = fn(*args)
    if status:
        raise DriverError(f"{fn.__name__}: hip error {status}")


class Hip:
    def __init__(self, path):
        try:
            self.dll = ct.WinDLL(path)
        except OSError as error:
            raise DriverUnavailable(f"cannot load HIP: {error}") from None

    def bind(self, name, *args):
        return _bind(self.dll, name, args)

    def device(self):
        devices = ct.c_int()
        if (self.bind("hipInit", ct.c_uint)(0)
                or self.bind("hipGetDeviceCount", ct.POINTER(ct.c_int))(ct.byref(devices)) or not devices.value):
            raise DriverUnavailable("no AMD HIP device")
        _hip_check(self.bind("hipSetDevice", ct.c_int), 0)
        name = ct.create_string_buffer(256)
        _hip_check(self.bind("hipDeviceGetName", ct.c_void_p, ct.c_int, ct.c_int), name, len(name), 0)
        # As in the C launcher, use an opaque, aligned buffer for the versioned properties ABI. Match only a
        # complete target ID, including optional features. The product name and arbitrary binary fields differ.
        props = (ct.c_uint64 * 2048)()
        _hip_check(self.bind("hipGetDevicePropertiesR0600", ct.c_void_p, ct.c_int), props, 0)
        targets = set(re.findall(rb"(?:^|\0)(gfx[0-9a-f]{3,4})(?::[a-z0-9_+\-]+)*\0", bytes(props)))
        if len(targets) != 1:
            raise DriverUnavailable("HIP did not report one GFX target ID for device 0")
        arch = targets.pop().decode("ascii")
        return f"# device={name.value.decode('utf-8', errors='replace')} arch={arch}"

    def launch(self, code, words, tokens):
        """Yield the existing launcher's lines for kernels in one code object."""
        if not 1 <= words <= MAX_WORDS:
            raise ValueError(f"words must be 1..{MAX_WORDS}")
        if not code.startswith(b"\x7fELF"):
            raise ValueError("expected an ELF code object")
        yield self.device()
        ptr, uint, size = ct.c_void_p, ct.c_uint, ct.c_size_t
        module, buffer = ptr(), ptr()
        image = ct.create_string_buffer(code)
        with ExitStack() as stack:
            load = self.bind("hipModuleLoadData", ct.POINTER(ptr), ptr)
            unload = self.bind("hipModuleUnload", ptr)
            _hip_check(load, ct.byref(module), image)
            stack.callback(unload, module)
            malloc = self.bind("hipMalloc", ct.POINTER(ptr), size)
            free = self.bind("hipFree", ptr)
            _hip_check(malloc, ct.byref(buffer), words * 4)
            stack.callback(free, buffer)
            copy = self.bind("hipMemcpy", ptr, ptr, size, ct.c_int)
            find = self.bind("hipModuleGetFunction", ct.POINTER(ptr), ptr, ct.c_char_p)
            launch = self.bind("hipModuleLaunchKernel", ptr, *([uint] * 7), ptr, ct.POINTER(ptr), ct.POINTER(ptr))
            sync = self.bind("hipDeviceSynchronize")
            argument_size = size(ct.sizeof(buffer))
            extra = (ptr * 5)(BUFFER_POINTER, ct.addressof(buffer), BUFFER_SIZE, ct.addressof(argument_size), EXTRA_END)
            result = (ct.c_uint32 * words)()
            for token in tokens:
                name, *bounds = token.split(":", 2)
                threads = int(bounds[0]) if bounds else 1
                wanted = int(bounds[1]) if len(bounds) == 2 else words
                if not name or not 1 <= threads <= 1024 or not 1 <= wanted <= words:
                    raise ValueError(f"bad kernel token: {token}")
                result[:wanted] = [SENTINEL] * wanted
                _hip_check(copy, buffer, result, wanted * 4, HOST_TO_DEVICE)
                kernel = ptr()
                _hip_check(find, ct.byref(kernel), module, name.encode("ascii"))
                _hip_check(launch, kernel, 1, 1, 1, threads, 1, 1, 0, None, None, extra)
                _hip_check(sync)
                _hip_check(copy, result, buffer, wanted * 4, DEVICE_TO_HOST)
                yield " ".join([name, *(f"{word:08x}" for word in result[:wanted])])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        action, *args = sys.argv[1:]
        if action == "assemble":
            source, output, arch, wave, version = args
            Path(output).write_bytes(assemble(Path(source).read_text(encoding="utf-8"), arch, int(wave), int(version)))
        elif action == "device":
            print(Hip(*args).device())
        elif action == "launch":
            image, words, dll = args
            for line in Hip(dll).launch(Path(image).read_bytes(), int(words), sys.stdin.read().split()):
                print(line)
        else:
            raise ValueError(f"unknown driver worker action: {action}")
    except DriverUnavailable as error:
        print(error, file=sys.stderr)
        return 77
    except (DriverError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
