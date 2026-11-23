"""Write a one-line Windows batch file (a fake tool for a test) that cmd.exe can run from any user folder.

cmd.exe reads the batch file using the console output code page. This can differ from the system OEM code page
when a terminal uses UTF-8. Without an attached console, use the system OEM code page.
Standard library only.
"""


def batch_path(path):
    """A spelling of path that cmd.exe reads correctly from a batch file: as is when ASCII, else its 8.3 short name
    (ASCII). Falls back to the long path where the volume has 8.3 names turned off."""
    if path.isascii():
        return path
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
    return buf.value if 0 < n < len(buf) else path


def _encodes(char, code_page):
    import codecs

    try:
        codecs.code_page_encode(code_page, char)
        return True
    except UnicodeEncodeError:
        return False


def write_python_cmd(cmd_path, python, script):
    """Write cmd_path as a batch file that runs `python script` with the batch file's arguments.

    Raises unittest.SkipTest when a path has no usable 8.3 name and contains letters outside the code page."""
    import codecs
    import ctypes

    kernel32 = ctypes.windll.kernel32
    code_page = kernel32.GetConsoleOutputCP() or kernel32.GetOEMCP()
    line = f'@"{batch_path(python)}" "{batch_path(script)}" %*\n'
    try:
        data, _ = codecs.code_page_encode(code_page, line)
    except UnicodeEncodeError:
        import unittest
        # The Windows encoder doesn't always report a useful position.
        letters = "".join(sorted({c for c in line if not _encodes(c, code_page)}))
        raise unittest.SkipTest(f"{letters!r} in a path has no usable 8.3 name and is outside code page {code_page},"
                                " so cmd.exe can't run a batch file that names it; use an ASCII TEMP") from None
    with open(cmd_path, "wb") as f:
        f.write(data)
    return cmd_path
