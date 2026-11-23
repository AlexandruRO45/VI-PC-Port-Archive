#!/bin/sh
# What a game image needs from the macOS host, before running it: the imports nothing answers yet, and the x86
# instructions Rosetta 2 cannot run (isa_probe.c: ADX, SHA, RDSEED, CLWB, SSE4a, CLZERO, AVX-512 all SIGILL).
# Prints names and counts only. Usage: sh scan_exe.sh <game.exe>   (run from runtime/build/windows-mac-mt)
set -e
exe=${1:?usage: sh scan_exe.sh <game.exe>}
here=$(cd "$(dirname "$0")" && pwd)
objdump=${OBJDUMP:-/opt/homebrew/opt/llvm/bin/llvm-objdump}

echo "== the game's imports gta6rt answers with nothing (gta6rt --dry-run through the host, nothing is run)"
perl -e 'alarm 300; exec @ARGV' "$here/build/macos_host" gta6rt.exe --dry-run "$exe" 2>&1 |
    grep -E 'UNIMPLEMENTED|unimplemented|module\(s\) mapped|host:' | grep -v 'host: mapped' || true

echo "== instructions Rosetta 2 does not run (count, mnemonic)"
"$objdump" -d --no-show-raw-insn --no-leading-addr "$exe" 2>/dev/null |
    awk '{print $1, $0}' |
    grep -Ei '^((adcx|adox|rdseed)[wlq]?|sha1[a-z0-9]*|sha256[a-z0-9]*|clwb|clzero|extrq|insertq|movntsd|movntss) |zmm|%k[0-7]\b|\{%k' |
    awk '{print $1}' | sort | uniq -c | sort -rn
echo "== done (nothing listed above = nothing found)"
