"""Interprocess exclusion for builds that publish several files into one directory."""
from contextlib import contextmanager
import hashlib
import os
import stat
import tempfile

try:
    import fcntl
except ImportError:                       # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:                       # POSIX
    msvcrt = None


class OutputLockError(RuntimeError):
    """The builder could not acquire the output directory's lock."""


@contextmanager
def exclusive_output(out_dir: str, builder: str, *, lock_path: str | None = None):
    """Hold a nonblocking OS lock, released even if the builder process dies."""
    local_lock = lock_path is not None
    if lock_path is None:
        # world_build leaves no output on failure. Retain its lock name across versions.
        key = hashlib.sha256(
            os.path.normcase(os.path.realpath(out_dir)).encode("utf-8", "surrogatepass")
        ).hexdigest()[:20]
        lock_path = os.path.join(tempfile.gettempdir(), f"leonida-{builder.replace('_', '-')}-{key}")
        lock_file = open(lock_path, "a+b")
    else:
        if os.path.islink(lock_path) or os.path.isjunction(lock_path):
            raise OutputLockError(f"refusing a redirected build lock: {lock_path}")
        try:
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            lock_file = os.fdopen(os.open(lock_path, flags, 0o666), "r+b")
        except OSError as exc:
            raise OutputLockError(f"cannot open the build lock {lock_path}: {exc}") from exc
    with lock_file as fh:
        fd = fh.fileno()
        info = os.fstat(fd)
        if local_lock and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise OutputLockError(f"expected a regular file with one link for the build lock: {lock_path}")
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                # The local lock can stay empty: the locked byte may extend past EOF.
                if not local_lock and info.st_size == 0:
                    fh.write(b"\0")
                    fh.flush()
                fh.seek(0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                raise OutputLockError("no file-lock implementation available")
        except OSError as exc:
            raise OutputLockError(f"another {builder} is using {out_dir}") from exc
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            else:
                fh.seek(0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
