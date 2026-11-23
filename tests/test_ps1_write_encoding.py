"""Every PowerShell script under scripts/ that writes a text file names its encoding.

    python -I tests/test_ps1_write_encoding.py

Windows PowerShell 5.1 writes a text file in the ANSI code page for `Set-Content`, `Add-Content` and
`Export-Csv`, and in UTF-16LE for `Out-File` and `>`; PowerShell 7 writes UTF-8 for all of them. The same
script therefore leaves byte-different files on the two hosts the build runs on, and a report with a non-ASCII
path or a localized Windows message is where it shows. Measured on this machine: `'aä' | Set-Content x` wrote
`61 e4 0d 0a` on Windows PowerShell 5.1 and `61 c3 a4 0d 0a` on 7; `-Encoding utf8` wrote `61 c3 a4 0d 0a` on
both (5.1 adds a BOM, 7 does not, but neither depends on the console's code page). This is the PowerShell
counterpart of `tests/test_text_write_line_endings.py`, which does this for the Python writers.

`Tee-Object` is not checked: Windows PowerShell 5.1 has no `-Encoding` on it
(`(Get-Command Tee-Object).Parameters.ContainsKey('Encoding')` is False there), so its file cannot be pinned.
`[IO.File]::WriteAllText` is not checked either: it writes UTF-8 without a BOM on every .NET.

Reads tracked scripts only; no game files.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = "scripts"     # the build helpers a user runs by hand; no game data is read to check them
WRITERS = ("Set-Content", "Add-Content", "Out-File", "Export-Csv")
# A cmdlet is a command when it starts the line or follows a pipe, semicolon, brace or parenthesis.
WRITER = re.compile(r"(?<![\w.-])(?:%s)\b" % "|".join(WRITERS))


def strip_comment(line):
    """`line` up to a `#` that is outside a quoted string."""
    single = double = False
    for i, ch in enumerate(line):
        if ch == "'" and not double:
            single = not single
        elif ch == '"' and not single:
            double = not double
        elif ch == "#" and not single and not double:
            return line[:i]
    return line


def mask_strings(text):
    """`text` with the contents of '...' and "..." blanked, so a cmdlet name inside a message is not a command."""
    out, quote, i = [], None, 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "`" and quote == '"':      # a backtick escapes the next character inside "..."
                out.append("  ")
                i += 2
                continue
            if ch == quote:
                quote = None
            out.append(" ")
        elif ch in "'\"":
            quote = ch
            out.append(" ")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def logical_lines(source):
    """[(first line number, the line with ` continuations joined)] and block comments dropped."""
    out, block, i = [], False, 0
    raw = source.split("\n")
    while i < len(raw):
        first, text = i + 1, raw[i]
        i += 1
        if block:
            if "#>" in text:
                block = False
                text = text.split("#>", 1)[1]
            else:
                continue
        text = re.sub(r"<#.*?#>", "", text)
        if "<#" in text:
            block = True
            text = text.split("<#", 1)[0]
        while text.rstrip().endswith("`") and i < len(raw):
            text = text.rstrip()[:-1] + " " + raw[i]
            i += 1
        out.append((first, strip_comment(text)))
    return out


def unpinned_writes(source):
    """Line numbers of writer cmdlets in `source` that pass no -Encoding."""
    found = []
    for number, text in logical_lines(source):
        masked = mask_strings(text)
        if not WRITER.search(masked):
            continue
        for match in WRITER.finditer(masked):
            if "-Encoding" not in masked[match.start():]:
                found.append(number)
    return found


def script_files():
    try:
        out = subprocess.run(["git", "ls-files", SCAN + "/*.ps1"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
        return [p for p in out.split() if p]
    except (OSError, subprocess.CalledProcessError):
        folder = os.path.join(ROOT, SCAN)
        return [SCAN + "/" + name for name in sorted(os.listdir(folder)) if name.endswith(".ps1")]


class Ps1WriteEncoding(unittest.TestCase):
    def test_detector_finds_the_unpinned_cmdlets(self):
        src = "\n".join([
            'Set-Content -LiteralPath $a -Value $lines',                       # 1 flagged
            'Set-Content -Encoding utf8 -LiteralPath $b -Value $lines',        # pinned
            '$rows | ConvertTo-Json | Set-Content -LiteralPath $c -Value $_',  # 3 flagged
            '# Set-Content -LiteralPath $d -Value $lines',                     # a comment
            '"a Set-Content inside a string" | Write-Host',                    # a string
            '$x | Out-File -LiteralPath $e',                                   # 6 flagged
            'Export-Csv -NoTypeInformation -LiteralPath $f',                   # 7 flagged
            '$lines | Set-Content -LiteralPath $g `',                          # 8 flagged (continuation)
            '    -Value $lines',
            'Tee-Object -LiteralPath $h',                                      # not a named writer
            '[IO.File]::WriteAllText($i, $text)',                              # .NET, always UTF-8
            '<#',
            'Set-Content -LiteralPath $j -Value $lines',                       # inside a block comment
            '#>',
            'Set-Content -Path $k -Value $l -Encoding ascii',                  # pinned
        ])
        self.assertEqual(unpinned_writes(src), [1, 3, 6, 7, 8])

    def test_every_script_writer_names_its_encoding(self):
        files = script_files()
        self.assertGreaterEqual(len(files), 4, "the scan found almost no scripts; is the layout different?")
        bad = []
        for path in files:
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                bad += [f"{path}:{n}" for n in unpinned_writes(f.read())]
        self.assertEqual(bad, [], "PowerShell text writes without -Encoding (ANSI on 5.1, UTF-8 on 7): "
                         + ", ".join(bad))


if __name__ == "__main__":
    unittest.main()
