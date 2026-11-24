#!/usr/bin/env bash
# Exercise the normal launcher and title entry point without any game files.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
title="$work/probe title.exe"
rustc --edition 2024 --target x86_64-pc-windows-gnullvm -C target-feature=+crt-static \
    -C panic=abort -C opt-level=2 \
    -C link-arg=-nostartfiles -C link-arg=-Wl,--entry=title_main \
    "$root/tests/fixtures/proton-smoke.rs" -o "$title"
"$root/scripts/victory-proton.sh" "$title" --execute 2>&1 | tee "$work/run.log"
# The CLI prints the title's return value independently of its own exit code.
grep -Eq '^title exited with 42[[:space:]]*$' "$work/run.log"
