# The retail executable's host probes (RT-5)

**Status: measured from the file, 2026-11-23** (Ghidra static analysis, and `tools/import_call_sites.py`
for the call counts). The image is the one [runtime_imports.md](runtime_imports.md) names, and its SHA-256
matches. Nothing was run, and no executable bytes or addresses are quoted: only RVAs, export names and
the constants a call passes.

## 1. `ConsoleMode`, under the OEM key

One helper (RVA `0x55deb88`) opens `HKCU\Software\Microsoft\Windows NT\CurrentVersion\OEM` with
`RegOpenKeyW`, queries the `ConsoleMode` value with `RegQueryValueExW` into a zeroed 8-byte buffer whose
size word starts at 4, and closes the key. A query that fails — the key or value missing on a plain PC host —
returns 1. A successful query feeds the value's first dword into a ladder:

| First dword of `ConsoleMode` | Returns |
|---:|---:|
| 16, 32, 33, 48 or 49 | 2 |
| 64 | 3 |
| 80 | 4 |
| 88 | 5 |
| 96 | 6 |
| 100 | 7 |
| 104 | 8 |
| any other value | 0 |

What the dword denotes is not known from the file, but the ladder is live: of the helper's three direct
calls (RVAs `0x75185`, `0x1e1cf84`, `0x55feaf6`) plus a stored function pointer (RVA `0xaea49b8`), the
first sets a flag byte to `class != 6`, the second proceeds only for classes 1, 5 and 6 (refusing anything
above 8), and the third passes the class to a follow-up call. Class 1 is the answer a host without the key
gives, so the second site runs on a PC and on a console whose dword is 88 or 96, and skips the other
values. The offline answer to this key comes from `LEONIDA_DEVICE_TYPE` (#6990); a value it writes must
carry the dword this build expects. One of the callers is the function that also calls
`D3D12ConfigureVirtualMemoryX`. `tools/import_call_sites.py` counts no direct call to `RegOpenKeyW`: the
helper reaches the imports through computed calls on the address table, which the tool does not follow.

## 2. Media Foundation, started LITE

The executable imports seven Media Foundation functions, all bound to the host: `MFPlat.DLL`'s `MFStartup`,
`MFShutdown`, `MFGetSystemTime`, `MFCreateAttributes`, `MFCreateMediaType` and `MFCreateSourceResolver`,
plus `MFReadWrite.dll`'s `MFCreateSourceReaderFromMediaSource`. `tools/import_call_sites.py` counts one
direct call each to `MFStartup`, `MFShutdown`, `MFCreateAttributes`, `MFCreateSourceResolver` and
`MFCreateSourceReaderFromMediaSource`, two to `MFCreateMediaType`, and `MFGetSystemTime` reached by one
tail jump instead of a call.

The `MFStartup` site is a small helper (RVA `0x33a41d0`; one caller at RVA `0x33a45e8`, one stored
function pointer at RVA `0xad774c4`): `CoInitializeEx(NULL, COINIT_MULTITHREADED)`, retried with
`COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE` when the first call answers `RPC_E_CHANGED_MODE`;
a COM that still fails answers 0. Otherwise the helper calls `MFStartup(0x00020070, 1)` — `MF_VERSION`
with `MFSTARTUP_LITE`, which the SDK's `mfapi.h` defines as `MFSTARTUP_NOSOCKET`, the lightest documented
startup — and answers 3 on success, 1 on failure. Whether the helper runs before the first frame, and
what the source resolver or reader then opens, is a `--trace` question ([boot.md](boot.md)); on a host
without the Media Foundation optional feature, `MFStartup` fails and the helper answers 1.
