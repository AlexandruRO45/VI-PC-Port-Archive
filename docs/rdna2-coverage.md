# RDNA2 opcode coverage: what the tests pin, and what a second backend would have to cover

Written for [#83](https://github.com/Zettrox/ztrx/issues/83) (an iOS / Metal target). The numbers
answer the question that issue actually turns on: how much of the shader backend is *proven*, and
how much of it a second backend would have to reproduce unproven.

Everything here comes from the tree. No device, no game files, no run.

## The three numbers

| | Count | Source |
|---|---|---|
| Opcodes the GFX10 tables name | **1,232** | `gfx10/opcodes.rs`, every `(0x…, "name")` entry |
| Opcodes `lower.rs` names explicitly | **333** | string literals in `emit/lower.rs` |
| Opcodes the test files name | **89** | the 5 `*_tests.rs` files |

Read as a funnel: the ISA has 1,232 names, 333 of them are reached by name in the lowering, and 89
are named in a test. **The gap between 333 and 89 is the interesting one** — 244 opcodes are lowered
by code no test mentions by name.

Two caveats before anyone quotes these:

- Naming an opcode in `lower.rs` is not the same as handling it. Many appear in `match` arms that
  pass through to a shared helper, and a few only in a `tracing` message. The 333 is an upper bound
  on "the lowering knows this name", not "this name works".
- A test naming an opcode is not the same as that opcode being pinned. Some appear in a comment.
  The 89 is an upper bound too.

So the funnel bounds the truth rather than stating it. Getting an exact number means making the
lowering fail on an unknown opcode and sweeping the corpus — which is what `victory translate …
--list` does on real data, and what I cannot do without game files.

## Why this matters for a Metal backend

The tests are the only thing that would carry over to a second backend unchanged. `diff_tests.rs`
compiles a program with the console SDK's `dxc.exe` and the PC one and compares the two; the
`valu`/`wave64`/`intersect` suites pin RDNA2 results the translations must reproduce. Those are
reference outputs, and a Metal emitter would be held to the same ones.

If 89 opcodes are pinned and 333 are lowered, then **a second backend can be validated on roughly a
quarter of the surface it has to cover** — and the rest has to be right by construction. That is not
a reason not to do it, but it is the reason the estimate is "replace the back half of `lower.rs`"
and not "add an emitter": you cannot lean on the tests for the other 244.

## What could be done about that, in order

1. **Make the unpinned part fail loudly.** `lower` already returns `Unsupported { what }` for
   constructs it refuses. Sweeping the 26,503-shader corpus with `--list` would turn the 244 into a
   list of "reached by name, never asserted". That is a run on a machine with game files — the
   maintainer or a tester has it, I do not.
2. **Generate a table from that sweep** (opcode → number of programs that reach it) and sort by
   frequency. The top of that list is the minimum viable Metal backend, and it is very likely far
   below 244 — game shaders use a small subset heavily.
3. **Write the reference outputs for the top N** before writing the backend. They are cheap: the
   `valu_tests` harness already builds a program, runs it on the card and compares; the same program
   through a Metal path would compare against the same array.

Step 3 is the one that makes the iOS decision safe, and it does not need iOS hardware — it needs a
macOS machine for the MSL leg, which I also do not have.

## The other two blockers, in one line each

- **x64 without JIT**: `victory-loader` maps the title's PE and lets its x86-64 run. iOS grants no
  JIT to a third-party app, so this becomes an ahead-of-time x86-64 → AArch64 recompiler over a
  shipping game's code — a different project, and one that must be correct rather than
  best-effort.
- **Windows above the GPU**: `victory-hle` answers the console's API by calling Windows. Each module
  in `victory-hle/src/modules/` needs an iOS equivalent, and the ones already marked `shortcut:`
  (audio reverb silent, speech silent, HDR absent) get harder rather than easier.

## Scope of this document

Analysis only. It proposes a spike (inventory → frequency table → reference outputs → backend) and
does not claim a port is feasible or infeasible; the funnel above is why the answer is not obvious
either way. If the maintainer wants step 1 or the frequency table, both need a run over the shader
corpus, which needs game files.
