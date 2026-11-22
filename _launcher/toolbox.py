"""Tool checks, downloads and installs for the Leonida Launcher (standard library only).

Each tool can be: found automatically, pointed to by the user ("Locate"), or downloaded. Installers are run with
their normal window so the user sees and accepts each one's licence and the Windows admin prompt themselves.
"""
import ctypes
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = os.path.join(HERE, "downloads")
PORTABLE = os.path.join(HERE, "tools")
UA = {"User-Agent": "LeonidaLauncher/1.0"}
NO_WINDOW = 0x08000000

# key: label, required, what it's for, how to get it, what Locate asks for
TOOLS = [
    {"key": "python", "label": "Python 3.12+", "required": True, "why": "Runs the build scripts",
     "get": ("installer", "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"),
     "locate": ("file", "python.exe")},
    {"key": "dotnet", "label": ".NET 8 SDK", "required": True, "why": "Exports the game's assets",
     "get": ("installer", "https://aka.ms/dotnet/8.0/dotnet-sdk-win-x64.exe"), "locate": ("file", "dotnet.exe")},
    {"key": "vulkan", "label": "Vulkan SDK", "required": True, "why": "spirv-as / spirv-val, pipeline checker",
     "get": ("installer", "https://sdk.lunarg.com/sdk/download/latest/windows/vulkan-sdk.exe"),
     "locate": ("dir", "the Vulkan SDK folder (the one with Bin\\spirv-val.exe)")},
    {"key": "git", "label": "Git", "required": True, "why": "Updates and the rpf9-explorer submodule",
     "get": ("installer", "github:git-for-windows/git:Git-*-64-bit.exe"), "locate": ("file", "git.exe")},
    {"key": "vs", "label": "Visual Studio C++ + Windows SDK", "required": True,
     "why": "C++ tools + Windows SDK 10.0.26100+",
     "get": ("installer", "https://aka.ms/vs/17/release/vs_community.exe",
             ["--add", "Microsoft.VisualStudio.Workload.NativeDesktop", "--add",
              "Microsoft.VisualStudio.Component.Windows11SDK.26100", "--includeRecommended"]),
     "locate": None},
    {"key": "llvm", "label": "LLVM llvm-mc (AMDGPU)", "required": True,
     "why": "Shader disassembler, built from source (20-40 min)",
     "get": ("script", "scripts\\build_llvm_mc.ps1"), "locate": ("file", "llvm-mc.exe")},
    {"key": "ffmpeg", "label": "ffmpeg", "required": False, "why": "Decodes XMA2 audio, needed for WAV",
     "get": ("zip", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", "ffmpeg.exe"),
     "locate": ("file", "ffmpeg.exe")},
]
BY_KEY = {t["key"]: t for t in TOOLS}


def run(args, env=None, timeout=60, cwd=None):
    try:
        r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, errors="replace",
                           timeout=timeout, creationflags=NO_WINDOW)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def apply_overrides(env, paths):
    """Put the tools the user located (or the launcher downloaded) first in the build's environment."""
    env = dict(env)
    front = []
    for key, p in (paths or {}).items():
        if not p or not os.path.exists(p):
            continue
        if key == "vulkan":
            env["VULKAN_SDK"] = p
            front.append(os.path.join(p, "Bin"))
        elif key == "llvm":
            env["LLVM_MC"] = p
        elif key == "dotnet":
            env["DOTNET_ROOT"] = os.path.dirname(p)
            front.append(os.path.dirname(p))
        elif key in ("git", "ffmpeg", "python"):
            front.append(os.path.dirname(p))
    if front:
        env["PATH"] = ";".join(front + [env.get("PATH", "")])
    return env


def which(name, env):
    return shutil.which(name, path=env.get("PATH"))


def check(key, env, py_version):
    """(ok, detail) for one tool."""
    if key == "python":
        return py_version >= (3, 12), "Python %d.%d.%d" % py_version[:3]
    if key == "dotnet":
        exe = which("dotnet", env)
        if not exe:
            return False, "Not found"
        rc, out = run([exe, "--list-sdks"], env=env)
        sdks = [line.split()[0] for line in out.splitlines() if line[:1].isdigit()]
        ok = any(s.split(".")[0].isdigit() and int(s.split(".")[0]) >= 8 for s in sdks)
        return ok, ("SDK " + ", ".join(sdks)) if sdks else "dotnet found, but no SDK installed"
    if key == "vulkan":
        sdk = env.get("VULKAN_SDK", "")
        sv = which("spirv-val", env) or (os.path.join(sdk, "Bin", "spirv-val.exe") if sdk else "")
        if sdk and os.path.isfile(sv):
            return True, sdk
        return False, "Not found" if not sdk else f"VULKAN_SDK={sdk}, but no Bin\\spirv-val.exe there"
    if key == "git":
        g = which("git", env)
        return bool(g), g or "Not found"
    if key == "ffmpeg":
        f = which("ffmpeg", env)
        return bool(f), f or "Not found (optional)"
    if key == "llvm":
        mc = env.get("LLVM_MC") or which("llvm-mc", env)
        if not mc:
            return False, "Not found"
        if not os.path.isfile(mc):
            return True, f"LLVM_MC={mc}"
        rc, out = run([mc, "--version"], env=env)
        if "amdgcn" in out.lower() or "amdgpu" in out.lower():
            return True, mc
        return False, f"{mc} has no AMDGPU target"
    if key == "vs":
        vw = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                          "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if not os.path.isfile(vw):
            return False, "Not found"
        rc, out = run([vw, "-latest", "-products", "*", "-requires",
                       "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "displayName"])
        name = out.strip().splitlines()[-1] if out.strip() else ""
        if not name:
            return False, "Visual Studio found, but without the C++ tools"
        kits = r"C:\Program Files (x86)\Windows Kits\10\Include"
        vers = sorted((v for v in os.listdir(kits) if v.startswith("10.")),
                      key=lambda v: int(v.split(".")[2])) if os.path.isdir(kits) else []
        if not any(int(v.split(".")[2]) >= 26100 for v in vers):
            return False, f"{name}, but Windows SDK {vers[-1] if vers else 'none'} (need 10.0.26100+)"
        return True, f"{name}, Windows SDK {vers[-1]}"
    return False, "?"


def validate_located(key, path):
    """Whether a path the user picked is the right thing; returns (ok, reason)."""
    t = BY_KEY[key]
    if not path or not os.path.exists(path):
        return False, "That path doesn't exist"
    if key == "vulkan":
        return (os.path.isfile(os.path.join(path, "Bin", "spirv-val.exe")),
                "No Bin\\spirv-val.exe in that folder: pick the SDK's version folder, e.g. C:\\VulkanSDK\\1.4.x")
    want = t["locate"][1].lower()
    if os.path.basename(path).lower() != want:
        return False, f"Pick {t['locate'][1]}"
    if key == "llvm":
        rc, out = run([path, "--version"])
        if "amdgcn" not in out.lower() and "amdgpu" not in out.lower():
            return False, "That llvm-mc has no AMDGPU target. Use the Build button to make one."
    return True, ""


class Cancelled(Exception):
    pass


def resolve(url):
    """github:owner/repo:pattern -> the newest release asset matching pattern (one * allowed)."""
    if not url.startswith("github:"):
        return url, os.path.basename(url.split("?")[0])
    _, repo, pattern = url.split(":", 2)
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/latest", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    pre, _, post = pattern.partition("*")
    for a in data.get("assets", []):
        n = a["name"]
        if n.startswith(pre) and n.endswith(post):
            return a["browser_download_url"], n
    raise RuntimeError(f"No {pattern} in the latest {repo} release")


def download(url, progress, cancel):
    """Download to DOWNLOADS. progress(done_bytes, total_bytes_or_0). Returns the file path."""
    url, name = resolve(url)
    os.makedirs(DOWNLOADS, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        final = os.path.basename(r.geturl().split("?")[0]) or name
        if os.path.splitext(final)[1].lower() in (".exe", ".zip", ".msi"):
            name = final
        total = int(r.headers.get("Content-Length") or 0)
        dest = os.path.join(DOWNLOADS, name)
        part = dest + ".part"
        done = 0
        with open(part, "wb") as f:
            while True:
                if cancel.is_set():
                    raise Cancelled()
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                progress(done, total)
    if total and done < total:
        raise RuntimeError(f"Download cut off at {done // 1048576} of {total // 1048576} MB")
    os.replace(part, dest)
    return dest


def explain(e):
    """A plain reason for a failed download/install."""
    if isinstance(e, Cancelled):
        return "Cancelled"
    if isinstance(e, urllib.error.HTTPError):
        return f"The server answered {e.code} {e.reason}"
    if isinstance(e, urllib.error.URLError):
        if "getaddrinfo" in str(e.reason) or "11001" in str(e.reason):
            return "No internet connection, or the download site couldn't be found"
        if "CERTIFICATE" in str(e.reason).upper():
            return "The download site's security certificate was rejected"
        return f"Couldn't connect to the download site ({e.reason})"
    if isinstance(e, TimeoutError):
        return "The download timed out"
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return "The disk is full"
    if isinstance(e, zipfile.BadZipFile):
        return "The downloaded archive is damaged"
    return str(e) or e.__class__.__name__


class _SEI(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("fMask", ctypes.c_ulong), ("hwnd", ctypes.c_void_p),
                ("lpVerb", ctypes.c_wchar_p), ("lpFile", ctypes.c_wchar_p), ("lpParameters", ctypes.c_wchar_p),
                ("lpDirectory", ctypes.c_wchar_p), ("nShow", ctypes.c_int), ("hInstApp", ctypes.c_void_p),
                ("lpIDList", ctypes.c_void_p), ("lpClass", ctypes.c_wchar_p), ("hkeyClass", ctypes.c_void_p),
                ("dwHotKey", ctypes.c_ulong), ("hIcon", ctypes.c_void_p), ("hProcess", ctypes.c_void_p)]


def run_installer(path, args=()):
    """Start an installer the way Explorer would (so Windows shows its admin prompt) and wait for it.
    Returns the exit code; raises with a reason when it can't start."""
    sei = _SEI()
    sei.cbSize = ctypes.sizeof(sei)
    sei.fMask = 0x00000040  # SEE_MASK_NOCLOSEPROCESS
    sei.lpVerb = "open"
    sei.lpFile = path
    sei.lpParameters = subprocess.list2cmdline(list(args)) if args else None
    sei.nShow = 1
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(sei)):
        err = ctypes.GetLastError()
        if err == 1223:
            raise RuntimeError("The Windows admin prompt was declined")
        raise RuntimeError(f"The installer couldn't start (Windows error {err})")
    if not sei.hProcess:
        return 0
    k32 = ctypes.windll.kernel32
    k32.WaitForSingleObject(ctypes.c_void_p(sei.hProcess), 0xFFFFFFFF)
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(ctypes.c_void_p(sei.hProcess), ctypes.byref(code))
    k32.CloseHandle(ctypes.c_void_p(sei.hProcess))
    return code.value


def unzip_tool(zpath, exe_name):
    """Extract a portable tool into tools/<name> and return the path of exe_name inside it."""
    base = os.path.join(PORTABLE, os.path.splitext(os.path.basename(zpath))[0])
    if os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(base)
    for root, _, files in os.walk(base):
        if exe_name in files:
            return os.path.join(root, exe_name)
    raise RuntimeError(f"{exe_name} wasn't in the archive")
