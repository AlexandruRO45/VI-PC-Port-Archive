"""Publish a texture or its catalog only after the complete write succeeds."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import sys
import tempfile

from file_mode import published_mode


def _publish_new(tmp, target):
    if os.name == "nt":
        # Windows rename refuses an existing target, including on volumes without hard links.
        os.rename(tmp, target)
        return
    try:
        os.link(tmp, target)
        return
    except OSError as exc:
        if exc.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM, errno.ENOSYS):
            raise
        # Some removable and network volumes cannot create a hard link. Use the
        # OS's exclusive rename there so another writer still cannot be clobbered.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        source, dest = os.fsencode(tmp), os.fsencode(target)
        if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
            rc = libc.renamex_np(source, dest, 4)  # RENAME_EXCL
        elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
            rc = libc.renameat2(-100, source, -100, dest, 1)  # AT_FDCWD, RENAME_NOREPLACE
        else:
            raise
        if rc:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), target) from exc


@contextmanager
def staged_output(target, *, replace, preserve_mode=True):
    target = Path(target)
    output = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
    tmp = Path(output.name)
    try:
        with output:
            yield output
        if preserve_mode and os.name != "nt":
            tmp.chmod(published_mode(target))
        if replace:
            os.replace(tmp, target)
        else:
            _publish_new(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
