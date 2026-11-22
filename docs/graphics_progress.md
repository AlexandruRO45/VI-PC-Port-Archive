# Graphics path progress

`tools/graphics_progress.py` counts how far the graphics path has come: the Xbox D3D12 to Vulkan layer under
`runtime/` and the RDNA2 to SPIR-V translator `tools/rdna2spirv.py`. Every number is a count over committed text.
**It says nothing about whether the game runs.** Nothing is built, run or imported to get a number, and it needs no
game data, GPU or network.

```sh
python -I tools/graphics_progress.py                      # working tree -> docs/progress/graphics.{json,html,md}
python -I tools/graphics_progress.py --tree origin/main   # a git ref, read with git ls-tree / git show
python -I tools/graphics_progress.py --date 2026-11-22    # also append that day to graphics-history.json
python -I tools/graphics_progress.py --isa amdgpu_isa_rdna2.xml   # a fresh RDNA2 instruction total
```

The output is deterministic: sorted keys, LF line endings, no clock. A date appears only with `--date`.

## What is counted

| Section | Counted how | Limits |
|---|---|---|
| Milestones | Rows of the table in [d3d12_vulkan.md](d3d12_vulkan.md) section 6, each id once. A row's status is read from its own words: "Met by" or "**Met" is met, "Synthetic part met" and "Partly met" are what they say, anything else is open. | Open pull requests are not known offline. The words are the authors', not a verdict. |
| D3D12 interfaces | The interfaces handed to `tools/gen_d3d12_stubs.py` in `runtime/CMakeLists.txt`. Per interface: the methods of its chain against the overrides in the class that derives from its generated stub class. Real body, body that only returns `E_NOTIMPL`, or no override (a generated stub). | Matching is by method name, as in `gen_d3d12_stubs.py --report`. A real body can still be partial. The method names come from `tools/graphics_progress_data.json` (Windows SDK `d3d12.h`, names only). |
| RDNA2 ISA | `tools/isa_coverage.py` used as a library on the translator text. | The instruction total is AMD's XML, which is not in the repository, so the share is the one [isa_coverage.md](isa_coverage.md) states unless `--isa` is given. "Has a rule" is not "correct". |
| Formats | Keys of `Gen.FORMATS` in the translator (read with `ast`) out of the codes in `tools/gfx10_formats.json`; `DXGI_FORMAT` names in `vulkan_format()` and `vertex_format()` of the layer out of the DXGI list in the data file. | Image formats are not counted. Mapped is not "every usage works". |
| GPU tests | Distinct `add_test(NAME ...)` names in `runtime/CMakeLists.txt`, sorted into areas by name. | Registered is not passing. |
| Unsupported sites | `raise Unsupported(...)` in the translator, and `E_NOTIMPL` tokens and "unsupported" / "not implemented" strings in the graphics-path files of `runtime/`. | A site is a line of code, not one missing feature. |

`tools/graphics_progress_data.json` is the only input that is not in the code under measurement. Regenerate it with
`--refresh-data --d3d12-header <d3d12.h> --dxgi-header <dxgiformat.h>` when the layer starts to use a newer interface.

## Where the page lives

The outputs are not committed. `.github/workflows/graphics-progress.yml` regenerates them on pushes that touch the
graphics path, daily and on request, uploads them as the `graphics-progress` artifact and, when GitHub Pages is
enabled for the repository, publishes them there. The history file is carried between runs in the Actions cache.

## How it relates to other trackers

A tracker that counts the whole project answers a different question. This page counts only the graphics path and says how each number is made. It is meant to be read next to the plan in
[d3d12_vulkan.md](d3d12_vulkan.md) and issue [#313](https://github.com/ItsNotRexed/Leonida/issues/313).
