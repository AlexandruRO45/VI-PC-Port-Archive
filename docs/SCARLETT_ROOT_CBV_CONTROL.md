# Physical root CBV window control

Follow-up to #107 (5c73bee). A Scarlett Device11 guest binds a raw,
unregistered CPU allocation as a root CBV via slot 41, with a root UAV
at slot 45. Three committed 64 KiB pages contain distinct word patterns.
Four windows start at offsets 65280, 0, 65536 and 65792. Read actual
words at 0, 252, 32772 and 65532 bytes in each window, including the
last addressable word, and verify the two untouched output guards
per dispatch. This checks actual shader access, not just pipeline creation.

The opt-in regression case requires four imported buffer views with
widths 128/64/64/128 KiB respectively, no short-window diagnostic, no
stand-in and no null shader fallback. It explicitly disables null
fallback. It uses fresh lists and waits for each dispatch before release.
The exact import sequence is part of this synthetic control, not a
performance requirement for arbitrary applications. If resource reuse
changes legitimately, update the fixture and gate together.

Build with build.py d3d12 and run tools/check_d3d12_regressions.py
--case root-cbv --case tables with the usual artifacts. RTX 4080 / Proton
Experimental: four dispatches, 16/16 values and 8/8 guards, both cases
pass on provider built from 2e96986. The previous one-byte implementation
was not deliberately executed on NVIDIA because its short range was
reported to trigger a GPU fault. No AMD or retail claim; graphics-root
CBV and unmapped-window fallback are not exercised here.

```json
{
  "schema": 1,
  "scope": "Synthetic D3D12 provider controls; not retail execution",
  "adapter": "10de:2704",
  "proton": "1794497090 experimental-11.0-20261112-x86_64",
  "artifacts_sha256": {
    "d3d12_xs.dll": "0efe7d65f21a7b58bea17f67f46dfc8e4cc5eecbf91663d3109b446ffe13fa4b",
    "d3d12.dll": "fc2bbef02509fb11c334cc96dd211f64def94bb7b4eb82082a4808e6e0549543",
    "d3d12core.dll": "9356501320114773be279506644aabd9a1717ff451d480ac802a4fd4e94b2576",
    "dxgi.dll": "2814edc3350cae6d34a75b7900df19752889c310ef34255c7316c4a95e1b183f",
    "dxcompiler.dll": "0689e2fee0a1c9df8662a59680469f24e2a3ad7ce6bbbcc9284f7a83172bfc59",
    "dxil.dll": "64be2368211d257a038a7d4ea6bb3e24c878eb8d588c7b8414fccf759b69c2ac",
    "d3d12-root-cbv.exe": "a800703601fa833d80c16e4231e1cb4b2a6f0b53e8a012f73999419568bbe8b0",
    "d3d12-tables.exe": "3eca7b39d11f3d4a03cb2255348341ca55510fc9a5ecfe67b0135cadcbfa2866"
  },
  "unchanged": true,
  "cases": [
    {
      "case": "root-cbv",
      "passed": true,
      "exit": 0,
      "seconds": 13.336,
      "receipt": "d3d12-root-cbv PASS pci=10de:2704 slots=41,45 windows=4 words=16/16 guards=8"
    },
    {
      "case": "tables",
      "passed": true,
      "exit": 0,
      "seconds": 20.362,
      "receipt": "d3d12-tables PASS pci=10de:2704 words=16/16 guards=4 heaps=2 rejected=3 placed-srv=16/16 range-copies=4 descriptor-mode=1"
    }
  ],
  "passed": true
}
```
