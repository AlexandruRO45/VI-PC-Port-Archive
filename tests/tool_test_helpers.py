"""Helpers for test doubles that dispatch on executable names."""

import ntpath


def tool_stem(path):
    """Return a lowercase executable stem for Windows or POSIX paths on any host."""
    return ntpath.splitext(ntpath.basename(path))[0].lower()
