# Capped-heap texture table reuse

When a table reaches beyond the native heap's mirrored title slots, VIctory
rebuilds it in the dynamic descriptor pool. Recordings can share that range
when all populated entries are explicit texture SRVs and match in width,
descriptor generation, occupied offset, and native resource identity.

A recording retains each shared range and its native heap and textures until
reset or release. On the current synchronous submission path, native work
completes before reset. Each recording still schedules its own texture use,
including state transitions and uploads where applicable. Regular mirrored
tables and the remapping paths for raw table CBVs, buffer views, UAVs, placed
views, and implicit views keep their existing behavior.

## Synthetic reproduction

Run on Windows with the x64 Agility SDK 1.620.x DLLs and a Shader Model 6.6
DXC. Set the adapter to `amd`, `nvidia`, or `warp`.

```powershell
$env:VICTORY_TABLE_TEST_AGILITY = 'C:\path\to\Agility\build\native\bin\x64'
$env:VICTORY_TABLE_TEST_DXC = 'C:\path\to\dxc.exe'
$env:VICTORY_TABLE_TEST_ADAPTER = 'nvidia'
cargo test --locked -p victory-gpu --features diagnostics --lib descriptor::host::reuse_tests::capped_pool_texture_table_reuse -- --exact --ignored --nocapture --test-threads=1
```

The fixture uses a 65,536-slot pool with 2,048-slot tables and an original
compute shader. It checks native readback, a second execution of the same
closed command lists, separate generations for changed views, independent
heap identity, and release of the range on the last owner. The default run
enables the D3D12 debug layer and GPU-based validation.

| Fixture | Allocation failures, baseline to shared | Peak slots, baseline to shared |
| --- | ---: | ---: |
| 41 recordings, two distinct tables | 9 to 0 | 65,536 to 4,096 |
| 24 recordings, two distinct tables | 0 to 0 | 49,152 to 4,096 |
| 24 recordings, distinct tables | 0 to 0 | 49,152 to 49,152 |

For CPU preparation, run the same command with `--release` before the test
filter and `VICTORY_TABLE_TEST_TIMING=1`. This mode disables validation, warms
both paths, alternates their order over five pairs, and reports the median.
It times allocation and descriptor creation from already resolved entries.
GPU execution, native fence waits, and title descriptor resolution are
outside that measurement. Compare the two 24-recording cases for equal
work. A unique workload pays cache bookkeeping for the same occupancy.

In-game demand and performance remain unmeasured at this revision. The
separate title-memory ordering issue that can make raw table CBVs observe
later CPU writes is outside this cache.
