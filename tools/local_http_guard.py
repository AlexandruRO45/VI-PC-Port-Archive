"""Shared loopback Host and relative subpath guards for local HTTP servers."""
from __future__ import annotations

import os
import urllib.parse

_WINDOWS_RESERVED_DEVICES = frozenset({
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    # Windows also treats superscript 1/2/3 as device numbers. Keep this fallback
    # complete for Python 3.12, which has no os.path.isreserved.
    *(f"COM{i}" for i in "123456789\u00b9\u00b2\u00b3"),
    *(f"LPT{i}" for i in "123456789\u00b9\u00b2\u00b3"),
})

# Binding to 127.0.0.1 keeps other machines out, but not other web pages: a page on any site can point its own
# domain at 127.0.0.1 (DNS rebinding) and read local files through the server as same-origin. Its requests still carry
# that domain in the Host header, so only requests that name this machine are answered.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def _is_reserved_segment(seg: str) -> bool:
    # Spaces before an extension do not make a device into a normal filename:
    # Windows treats e.g. "NUL .glb" as NUL (as does ntpath.isreserved in Python 3.13+).
    stem = seg.split(".", 1)[0].rstrip(" ").upper()
    if stem in _WINDOWS_RESERVED_DEVICES:
        return True
    isreserved = getattr(os.path, "isreserved", None)
    if callable(isreserved) and isreserved(seg):
        return True
    return False


def host_allowed(host: str | None) -> bool:
    """True when a request's Host header names the loopback address. Browsers always send Host; a request without
    one comes from a local tool, not a web page, and is allowed.

    The value must be `name` or `name:port` (RFC 7230 section 5.4): text after an IPv6 literal's `]` other than
    `:port`, a second colon, or a port that is not 1-5 ASCII digits up to 65535 is malformed and refused."""
    if host is None:
        return True
    host = host.strip(" \t").lower()
    if host.startswith("["):                        # [::1] or [::1]:8765
        end = host.find("]")
        if end < 0:
            return False
        name, rest = host[:end + 1], host[end + 1:]
        if rest and not rest.startswith(":"):
            return False                            # [::1]junk, [::1].
        port = rest[1:] if rest else None
    else:
        name, colon, port = host.partition(":")
        if not colon:
            port = None
        elif ":" in port:
            return False                            # a bare IPv6 address, or name:port:more
        if name.endswith("."):                      # one trailing dot: the fully qualified form
            name = name[:-1]
    # The length cap comes before int(): a port of thousands of digits would hit Python's int conversion limit.
    if port is not None and not (0 < len(port) <= 5 and port.isascii() and port.isdigit() and int(port) <= 65535):
        return False
    return name in _LOOPBACK_HOSTS


def safe_subpath(base_dir: str, rel: str) -> str | None:
    """Resolve a relative URL path safely inside base_dir, or return None."""
    if not base_dir or not isinstance(rel, str) or not rel or "\x00" in rel:
        return None
    if rel.startswith(("/", "\\")) or rel.endswith(("/", "\\")):
        return None
    norm = rel.replace("\\", "/")
    if os.path.isabs(rel) or os.path.isabs(norm):
        return None
    parts = norm.split("/")
    for seg in parts:
        if not seg or seg in (".", "..") or ":" in seg or _is_reserved_segment(seg):
            return None
    real_base = os.path.realpath(base_dir)
    real_target = os.path.realpath(os.path.join(real_base, *parts))
    try:
        common = os.path.commonpath([real_base, real_target])
    except ValueError:
        return None
    if os.path.normcase(common) != os.path.normcase(real_base):
        return None
    if os.path.normcase(real_target) == os.path.normcase(real_base):
        return None
    if os.path.isdir(real_target):
        return None
    return real_target


def safe_prefix_path(base_dir: str, req_path: str, prefix: str) -> str | None:
    """Resolve a request path starting with `prefix` inside base_dir, or return None."""
    if not base_dir or not isinstance(req_path, str):
        return None
    clean = req_path.split("#", 1)[0].split("?", 1)[0]
    if not clean.startswith(prefix):
        return None
    return safe_subpath(base_dir, urllib.parse.unquote(clean[len(prefix):]))
