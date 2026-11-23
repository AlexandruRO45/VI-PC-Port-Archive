# AMD Windows driver: a 16-bit arithmetic shift by a zero amount loses the high-half extract

A standalone reproducer for [#5533](https://github.com/ItsNotRexed/Leonida/issues/5533) (TEST-1296): plain Vulkan
and SPIR-V, no Leonida code, so it can go to AMD as it is. No game data: the values come from the f16 fuzzer
(`tools/valu_fuzz.py --family f16 --seed 10 --programs 56 --wave 32`, program 55).

## What goes wrong

The fuzzer program ends with these two instructions:

```
v_med3_f16_e64 v24, v20, v27, v34 op_sel:[1,1,1,1]           ; v24 = 0x4000b88d
v_pk_ashrrev_i16 v29, v30, v24 op_sel:[1,1] op_sel_hi:[1,1]  ; both halves: hi(v24) >> (hi(v30) & 15) = 0x4000 >> 0
```

The answer is `0x40004000`. On the AMD driver the high result half is `0xb88d`, the low half of `v24`, so the word is
`0xb88d4000`. The translator's module is right: `tests/test_driver_repro_pk_ashr_hi16.py` runs it on the CPU, and
NVIDIA computes the same module correctly.

`pk_ashr_hi16.spvasm` is that module reduced with `spirv-reduce` (from the translator's 8 KB module), then moved
from Leonida's push-constant buffer pointer to a plain storage buffer. The repository ignores `.spvasm` files, so
`modules.py` holds both modules as text and writes them out. Each thread computes:

```
v24  = 0x4000b88d                                  ; stored to and loaded from a Function-scope array
amt  = BitFieldUExtract(v30, 16, 4)                ; 0
hi16 = (BitFieldSExtract(v24, 16, 16) >>arith amt) & 0xFFFF
word = lo16 | (hi16 << 16)                         ; lo16 is the same with its own copy of the ops
```

What it takes, measured with the module's variants on the AMD driver below:

| Change | AMD result |
|---|---|
| none | `0xb88d4000` (wrong) |
| shift amount 1, 2, 3, 4, 7, 8 or 15 instead of 0 (bits 19:16 of `v30`) | right |
| the shift amount a constant 0 instead of `BitFieldUExtract` | right |
| the three register writes without the `OpSelect` on the per-thread flag (the translator's EXEC model) | right |
| `OpShiftRightArithmetic v24 16` instead of `BitFieldSExtract v24 16 16` (the #1506 spelling) | wrong |
| the extract done on `%int` with `OpBitcast`s around it | wrong |
| the high half placed with `OpBitFieldInsert` instead of `<< 16` and `\|` | wrong |
| one `BitFieldSExtract` shared by both halves | wrong |
| the high half computed in place: `((v24 & 0xFFFF0000) >>arith amt) & 0xFFFF0000` (`pk_ashr_hi16_in_place.spvasm`) | right |

So the compiler seems to turn `(SExtract(x, 16, 16) >> 0 & 0xFFFF) << 16` into `x << 16` when it can see the value
through the `OpSelect`, dropping the extract's offset. The in-place spelling never moves the half down and back up,
and gives the same word for every shift amount 0..15. It is a candidate translator workaround, not part of this
change.

## Run it

Build the runner (it needs only the Vulkan loader and headers):

```powershell
cl /nologo /O2 vk_repro.c /I $env:VULKAN_SDK\Include /link /LIBPATH:$env:VULKAN_SDK\Lib vulkan-1.lib
```

```sh
gcc -O1 vk_repro.c -o vk_repro -lvulkan
```

Write the modules (`modules.py` also assembles them when `spirv-as` from the Vulkan SDK is on `PATH`), then run
them on a GPU whose name contains the second argument, ignoring case. AMD's Windows driver names the device
`AMD Radeon(TM) Graphics`; RADV names it after the chip (for example `AMD Ryzen ... (RADV RAPHAEL_MENDOCINO)`), so use
`radv` there. With no second argument the first device is used.

```powershell
python -I modules.py .
.\vk_repro.exe pk_ashr_hi16.spv radeon
.\vk_repro.exe pk_ashr_hi16_in_place.spv radeon
```

The runner prints the device and driver, every distinct word with its thread count, and `PASS` or `MISMATCH`.
Exit 0 when all 64 threads wrote the expected word (`0x40004000`, or a third argument in hex), 1 when any differs.

## Results

| GPU | Driver | `pk_ashr_hi16` | `pk_ashr_hi16_in_place` |
|---|---|---|---|
| AMD Radeon iGPU of a Ryzen 5 7600X (gfx1036, `1002:164e`), Windows 11 | 32.0.21045.29003 (Vulkan driver 2.0.353) | `0xb88d4000` in 64 of 64 threads | `0x40004000` |
| AMD Radeon iGPU (gfx1036, `1002:13c0`), Windows 11 ([@Halfect](https://github.com/ItsNotRexed/Leonida/pull/5725#issuecomment-7100360775)) | 32.0.21030.2001 (Vulkan driver 2.0.353) | `0xb88d4000` in 64 of 64 threads | `0x40004000` |
| AMD iGPU of a Ryzen 9 9950X3D, RADV `RAPHAEL_MENDOCINO`, Linux ([@gertify](https://github.com/ItsNotRexed/Leonida/pull/5725#issuecomment-7100514254)) | Mesa RADV, Vulkan 1.4.354; runner built with gcc | `0x40004000` | `0x40004000` |
| NVIDIA GeForce RTX 4070 SUPER, Windows 11 | 621.24 | `0x40004000` | `0x40004000` |

The same with `VK_LAYER_KHRONOS_validation` on: no validation messages, same words. `XFlawless1` saw the original
program fail on the gfx1036 iGPU of a Ryzen 7 9800X3D with driver 32.0.21045.29003 (#5533). So far it fails on AMD's
Windows driver and passes on RADV, on the same kind of GPU. Not yet run: discrete RDNA2 GPUs and other AMD Windows
drivers.

Add a row to this table, or a comment on #5533, with your GPU and driver.
