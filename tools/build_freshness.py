"""Is a built runtime tool older than the sources it was built from?

A build folder configured once keeps its binaries across `git pull`. make_game.py and first_light.py used to run
whatever vk_pipeline_check / vk_harness was there, so a pull that changed the harness gave wrong results (#7148).
Standard library only; nothing here touches a GPU.
"""
import os

SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".h", ".hpp", ".inc", ".in", ".cmake", ".txt")


def newest_source_mtime(root: str) -> float:
    """The newest modification time of a C/C++ or CMake file under root/runtime, build folders left out."""
    newest = 0.0
    for folder, subfolders, names in os.walk(os.path.join(root, "runtime")):
        subfolders[:] = [d for d in subfolders if d != "build"]
        for name in names:
            if name.endswith(SOURCE_SUFFIXES):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(folder, name)))
                except OSError:
                    pass  # removed while walking
    return newest


def binary_is_stale(exe: str, root: str) -> bool:
    """True when a runtime source file is newer than exe, so it needs a build before it is trusted."""
    try:
        built = os.path.getmtime(exe)
    except OSError:
        return False  # no binary at all is the caller's "not built" case, not a stale one
    return newest_source_mtime(root) > built
