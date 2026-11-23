# Translator and tiling performance

Measured cost of the two paths this note covers, and what was done about it. Both run without game data, so both
can be re-measured from a clean checkout with the commands below. Numbers are from one machine (2 x86-64 cores,
Python 3.13, GCC 14, `-O2`); the ratios are what to compare, not the absolute times.

## The translator

`make_game.py` translates 26 503 shaders with `tools/rdna2spirv.py` (#1904's corpus). At about 3 ms per shader on
the fixtures here that is over an hour of single-core work per full run, so the translator's per-instruction cost
is the build's cost. Its work is one Python interpreter over an instruction list: `shader_ir.parse_file` turns the
listing into operand dictionaries, `Gen.insns` walks them and emits SPIR-V text, `shader_meta.analyse_instructions`
runs a second pass for the build-time metadata, `lane_masks.analyse` a third for the register kinds, and `assemble`
stitches the module together.

### Measuring it

```
python -I tools/bench_translate.py 12          # every fixture in tests/semantic/fixtures/disasm, best and median pass
python -m pyinstrument tools/rdna2spirv.py ...  # where one pass's time goes, by pass rather than by call count
```

The fixtures are the regression suite, not the corpus: a few hundred to a few thousand instructions each, so the
per-shader fixed cost weighs more than it does on a real shader. A change that helps per instruction still shows.

### What the profile said

Sampling over one pass of the 97 fixtures (pyinstrument, 0.5 ms interval), before the work below:

| Pass | Share | What it was doing |
|---|---|---|
| `Gen.insns` (the interpreter) | 38% | ~11 emitted lines per instruction, and a dispatch chain that tests `op` against strings |
| `shader_ir.parse_file` | 12% | one `operand()` per operand, several regex matches for the ones that are not a bare register |
| `assemble` | 9% | `dedupe_types`, `no_contraction` and `split_big_switch`, each a pass over the whole module text |
| `shader_meta.analyse_instructions` | 9% | the operand split redone for every instruction on every worklist visit |
| `lane_masks.analyse` | 9% | `base_op` and a tuple of `startswith` prefixes per instruction, twice per instruction |

### What changed

1. **One `base_op`, memoised and interned** (`shader_ir.base_op`). Nine modules carried their own copy of the same
   four-`endswith` test and re-derived the same answer thousands of times per shader. The answer is now cached per
   mnemonic and `sys.intern`ed, which also turns the `op == "v_mov_b32"` tests in `Gen._insn`'s dispatch chain into
   pointer comparisons. `shader_meta`, `lane_masks`, `wave64_halves` and `wave64_emulate_opt` delegate to it.
2. **`operand()` memoised per spelling, with a copy on the way out** (`shader_ir.operand`). The same spellings recur
   constantly and the expensive ones cost several regex matches. The copy matters: the translator annotates the
   operand it is looking at (an SDWA select, an f64 literal) and a shared dictionary would leak that to every other
   instruction with the same spelling.
3. **`parse_file` counts on the text as read.** It joined every line back into one string to count
   `"invalid instruction encoding"`, copying the whole listing a second time.
4. **The construct stack `desc_cache_ok` reads is maintained by `emit()`** (`Gen.emit`, `Gen.desc_cache_ok`).
   The descriptor-lookup cache may only be used at the top level of a basic block, so the translator tracked the
   merge labels of the constructs still open by re-scanning the body: a `strip()` and a `split()` per emitted line.
   `emit()` has the line in hand, so it updates the stack there and `desc_cache_ok` became a lookup.
5. **The operand split is computed once per instruction** (`shader_meta._split_operands`, `analyse_instructions`).
   The worklist revisits a block once per change of its entry state, so a loop body was split several times and
   once more in the final pass; the result is a pure function of the instruction and is now cached for the call, and
   `_step_state` takes it instead of deriving it again.
6. **`_split_operands` classifies the mnemonic once** (`shader_meta._split_class`): the operand layout is a property
   of the opcode, not of the instruction, and the test was a run of `startswith` calls against a tuple of prefixes.
7. **The comparison tables are module constants** (`_CMP_F32` and friends in `rdna2spirv.py`): `compare()` built
   three dictionaries per call, for every `v_cmp` of every shader.
8. **`check_lane_local` keeps the hoisted-pointer set** (`Gen.check_lane_local`): it rebuilt a set over the whole
   register file for every instruction of every active run. `reg_ptrs` only grows, so its length is a version.
9. **`dedupe_types` splits only the header.** It examined the type declarations before the first function and
   appended the rest, but `text.split("\n")` split the whole module into lines first. A large shader's module is tens
   of MB.

### Result

`python -I tools/bench_translate.py 12`, 97 fixtures, 52 translated, 2 446 478 characters of SPIR-V per pass. The
two checkouts were run alternately, three rounds each, so drift hits both alike:

| | best pass | median pass |
|---|---|---|
| before | 342.1 ms | 361.0 ms |
| after | 294.2 ms | 312.5 ms |
| | **1.16x** | **1.16x** |

The same output: every semantic test that compares emitted text still passes (5 821 tests over 433 files), and the
`test_desc_cache` suite — which counts the descriptor lookups a shader makes — is unchanged, which is what says the
construct tracking in (4) is right.

## Tiling and detiling

`resource_detile.h` converts between the console's 64 KiB `SW_64KB_D` layout and linear row-major order, which is
what a texture upload from that layout pays per texel. The inner loop was one `memcpy` per element, and per element
a division (`x / tile_w`), a modulo, a table load and an OR before it.

### Measuring it

```
cmake -S runtime -B runtime/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build runtime/build --target resource_detile_bench
runtime/build/resource_detile_bench              # timings
runtime/build/resource_detile_bench --self-test  # the round trips, what ctest runs
```

### What changed

- The intra-tile offset of every element of one tile row is built once per row into a table and reused for every
  tile column in that row. The row is the same in every tile of the surface, so the per-element division, modulo,
  table load and OR are gone.
- The loop is unrolled four ways, and each element is a load and a store of its own width (`Bpe` is a compile-time
  constant) instead of a `memcpy` call site.
- The direction is a template parameter: detiling has the scattered side on the loads, tiling on the stores, and a
  16-byte element is one 128-bit load in the first case and two 64-bit stores in the second (measured each way
  round; the other pairing was slower).

### Result

Old and new interleaved in one process, three rounds of four repetitions each (so drift hits both sides alike),
and that run three times; the table is the average of the three. Noise between runs is about 10%, so read the
small cases as ratios and the 128 MB one as "unchanged".

| Surface | tile before | tile after | tile | detile before | detile after | detile |
|---|---|---|---|---|---|---|
| 1080p RGBA8 | 2.08 ms | 1.22 ms | 1.71x | 2.10 ms | 1.21 ms | 1.74x |
| 4K RGBA8 | 14.41 ms | 12.19 ms | 1.18x | 10.45 ms | 7.61 ms | 1.37x |
| 1024² R8 | 0.96 ms | 0.39 ms | 2.46x | 0.97 ms | 0.48 ms | 2.03x |
| 1024² RG8 | 0.96 ms | 0.41 ms | 2.36x | 0.98 ms | 0.51 ms | 1.92x |
| 1024² RGBA32F | 3.11 ms | 3.17 ms | 0.98x | 2.58 ms | 1.99 ms | 1.30x |
| 4096² RGBA16F (128 MB) | 44.45 ms | 43.57 ms | 1.02x | 43.71 ms | 42.95 ms | 1.02x |
| BC1 2048² | 0.24 ms | 0.14 ms | 1.66x | 0.24 ms | 0.14 ms | 1.75x |

The 16-byte element is the one case the rewrite does not help on the tiling side (0.98x here, 1.06x on another
run): a 16-byte move was already a single `memcpy`, so there was no per-element overhead left to remove, and the
two 64-bit stores it now uses are the same width. It is kept because detiling the same element is 1.30x and the
direction is one template parameter. The 128 MB case is past every cache and bound by memory, not by the loop;
the small-element cases are where the per-element overhead was most of the work.

Bit-exactness is the round trip, which `resource_detile_test` checks over every element size, sub-tile,
exact-tile, multi-tile, non-power-of-two and block-compressed shape, plus a deterministic randomized sweep of
240 shapes (a width ending inside a tile column, a height ending inside a tile row, both at once).

## Not measured here

- The corpus itself: 26 503 shaders need the game package, so the translator numbers above are per fixture, not per
  corpus shader. The per-instruction work is the same code path; the fixed per-shader cost is a larger share of a
  small shader than of a large one, so the corpus ratio should be at least the one measured here.
- The runtime's own hot paths (the D3D12 → Vulkan layer, the command list, descriptor binding) need Windows, a GPU
  and the game; they are outside what this note can measure.
