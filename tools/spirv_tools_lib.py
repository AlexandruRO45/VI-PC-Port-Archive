"""SPIRV-Tools in process: what spirv-as and spirv-val do, through the SPIRV-Tools shared library.

rdna2spirv.py assembles and validates every module it writes. Launching spirv-as and spirv-val for each one cost
about half of the time per small shader (two process starts, ~12 ms of ~24 ms). The Vulkan SDK ships the library both
programs are built from (lib/libSPIRV-Tools-shared.so, Bin/SPIRV-Tools-shared.dll), and its C API has the same two
entry points: spvTextToBinaryWithOptions (spirv-as) and spvValidateWithOptions (spirv-val). Called with the same target
environment and options, they give the same bytes and the same verdict, without a process per module.

The library is used only where it is the very build of the spirv-as and spirv-val on PATH: its version string must
equal the first line both programs print for --version. Otherwise (no library, another version, a missing entry
point) load() returns None and the caller launches the programs as before, so validation is never laxer than the
spirv-val a user has installed.

LEONIDA_SPIRV_LIB=off always launches the programs; LEONIDA_SPIRV_LIB=<path> names the library to try first (a
distro that keeps it outside the SDK layout), still subject to the version check.
"""
import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys

ENV = "LEONIDA_SPIRV_LIB"
_NAMES = {"win32": ("SPIRV-Tools-shared.dll",), "darwin": ("libSPIRV-Tools-shared.dylib",)}
_LIB_NAMES = _NAMES.get(sys.platform, ("libSPIRV-Tools-shared.so",))


class _Position(ctypes.Structure):
    _fields_ = [("line", ctypes.c_size_t), ("column", ctypes.c_size_t), ("index", ctypes.c_size_t)]


class _Diagnostic(ctypes.Structure):
    _fields_ = [("position", _Position), ("error", ctypes.c_char_p), ("isTextSource", ctypes.c_bool)]


class _Binary(ctypes.Structure):
    _fields_ = [("code", ctypes.POINTER(ctypes.c_uint32)), ("wordCount", ctypes.c_size_t)]


_P = ctypes.c_void_p
_DIAG = ctypes.POINTER(_Diagnostic)
_BIN = ctypes.POINTER(_Binary)
_SIGNATURES = {
    "spvSoftwareVersionDetailsString": (ctypes.c_char_p, []),
    "spvParseTargetEnv": (ctypes.c_bool, [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]),
    "spvContextCreate": (_P, [ctypes.c_int]),
    "spvTextToBinaryWithOptions": (ctypes.c_int, [_P, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32,
                                                  ctypes.POINTER(_BIN), ctypes.POINTER(_DIAG)]),
    "spvBinaryDestroy": (None, [_BIN]),
    "spvDiagnosticDestroy": (None, [_DIAG]),
    "spvValidatorOptionsCreate": (_P, []),
    "spvValidatorOptionsSetAllowOffsetTextureOperand": (None, [_P, ctypes.c_bool]),
    "spvValidateWithOptions": (ctypes.c_int, [_P, _P, _BIN, ctypes.POINTER(_DIAG)]),
}


class SpirvTools:
    """One library, one context and one set of validator options for the whole process (never destroyed: the
    process ends first)."""

    def __init__(self, lib, target_env, allow_offset_texture_operand):
        self._lib = lib
        env = ctypes.c_int()
        if not lib.spvParseTargetEnv(target_env.encode(), ctypes.byref(env)):
            raise ValueError(f"SPIRV-Tools does not know the target environment {target_env}")
        self._context = lib.spvContextCreate(env.value)
        self._options = lib.spvValidatorOptionsCreate()
        if not self._context or not self._options:
            raise MemoryError("SPIRV-Tools could not create a context")
        lib.spvValidatorOptionsSetAllowOffsetTextureOperand(self._options, allow_offset_texture_operand)

    def _take_diagnostic(self, diagnostic):
        if not diagnostic:
            return None
        try:
            d = diagnostic.contents
            return d.position.line, d.position.column, d.position.index, (d.error or b"").decode("utf-8", "replace"), \
                d.isTextSource
        finally:
            self._lib.spvDiagnosticDestroy(diagnostic)

    def assemble(self, text):
        """(status, binary bytes, stderr) as `spirv-as - -o <file> --target-env <env>` gives them for `text`
        (bytes): status 0 and the module, or non-zero and the message spirv-as prints (spvDiagnosticPrint)."""
        binary, diagnostic = _BIN(), _DIAG()
        status = self._lib.spvTextToBinaryWithOptions(self._context, text, len(text), 0,
                                                      ctypes.byref(binary), ctypes.byref(diagnostic))
        found = self._take_diagnostic(diagnostic)
        try:
            if status == 0 and binary:
                b = binary.contents
                return 0, ctypes.string_at(b.code, b.wordCount * 4), ""
        finally:
            if binary:
                self._lib.spvBinaryDestroy(binary)
        return 1, b"", _print_diagnostic(found)

    def validate(self, module):
        """(status, stderr) as `spirv-val --target-env <env> [--allow-offset-texture-operand] <file>` gives them
        for `module` (bytes): 0 and "" for a valid module, else non-zero and spirv-val's "error: line N: ..."."""
        words = len(module) // 4
        if len(module) % 4:
            return 1, "error: input is not a sequence of 32-bit words\n"
        buf = (ctypes.c_uint32 * words).from_buffer_copy(module)
        binary = _Binary(ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32)), words)
        diagnostic = _DIAG()
        status = self._lib.spvValidateWithOptions(self._context, self._options, ctypes.byref(binary),
                                                  ctypes.byref(diagnostic))
        found = self._take_diagnostic(diagnostic)
        if status == 0:
            return 0, ""
        # spirv-val's console consumer prints each message as "error: line <index>: <message>".
        return 1, f"error: line {found[2]}: {found[3]}\n" if found else "error: validation failed\n"


def _print_diagnostic(found):
    """The text spvDiagnosticPrint writes to stderr, which is what spirv-as reports a refused line with."""
    if found is None:
        return "error: spirv-as failed\n"
    line, column, index, message, text_source = found
    if text_source:
        return f"error: {line + 1}: {column + 1}: {message}\n"
    return f"error: {index}: {message}\n" if index > 0 else f"error: {message}\n"


def _first_line(tool):
    try:
        r = subprocess.run([tool, "--version"], capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = r.stdout.splitlines()
    return lines[0].strip() if r.returncode == 0 and lines else None


def _candidates(tools):
    named = os.environ.get(ENV, "").strip()
    if named:
        yield named
    for tool in tools:
        here = os.path.dirname(os.path.realpath(tool))
        for folder in (here, os.path.join(here, "..", "lib"), os.path.join(here, "..", "lib64")):
            for name in _LIB_NAMES:
                path = os.path.join(folder, name)
                if os.path.isfile(path):
                    yield os.path.normpath(path)
    found = ctypes.util.find_library("SPIRV-Tools-shared")
    if found:
        yield found


def _open(path):
    lib = ctypes.CDLL(path)
    for name, (restype, argtypes) in _SIGNATURES.items():
        fn = getattr(lib, name)      # AttributeError: an older library without this entry point
        fn.restype, fn.argtypes = restype, argtypes
    return lib


def find(target_env="vulkan1.2", allow_offset_texture_operand=True):
    """A SpirvTools that matches the spirv-as and spirv-val on PATH, or None (then launch the programs)."""
    if os.environ.get(ENV, "").strip().lower() in ("off", "0", "no", "false"):
        return None
    tools = [shutil.which("spirv-as"), shutil.which("spirv-val")]
    if not all(tools):
        return None
    versions = {_first_line(t) for t in tools}
    if len(versions) != 1 or None in versions:
        return None
    want = versions.pop()
    for path in dict.fromkeys(_candidates(tools)):
        try:
            lib = _open(path)
            if (lib.spvSoftwareVersionDetailsString() or b"").decode("utf-8", "replace").strip() != want:
                continue
            return SpirvTools(lib, target_env, allow_offset_texture_operand)
        except (OSError, AttributeError, ValueError, MemoryError):
            continue
    return None


_cache = {}


def load(target_env="vulkan1.2", allow_offset_texture_operand=True):
    """find(), once per process for the same PATH and LEONIDA_SPIRV_LIB."""
    key = (target_env, allow_offset_texture_operand, os.environ.get("PATH", ""), os.environ.get(ENV, ""))
    if key not in _cache:
        _cache[key] = find(target_env, allow_offset_texture_operand)
    return _cache[key]
