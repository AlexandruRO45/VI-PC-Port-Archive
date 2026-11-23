# Linux setup without root

How to get the GPU tests running on a Linux machine where you cannot install packages: no `sudo`, or a distribution
whose `llvm` package is older than 21 or lacks the AMDGPU target. Everything below goes into one folder you choose and
can delete afterwards. Nothing is written to `/usr`, and your shell configuration is not touched.

[testing.md](testing.md) lists what the tests need and explains each one. This page only adds the part that page
leaves out: where to get `llvm-mc` and SPIRV-Tools when "install them from your distribution" is not an option.

Measured on Fedora 44 (kernel 7.3), Python 3.14, GCC 16.2, CMake 4.3, Ninja 1.13, NVIDIA GeForce GTX 1660 SUPER
(Turing) with driver 620.70, at `main` `86a7807dd`. No game files are involved.

## What you still need from the system

- A Vulkan driver for your GPU and a working loader: `vulkaninfo --summary` lists the GPU.
- The compiler, CMake, Python and Ninja versions in [testing.md](testing.md#what-you-need). They were already present
  here. conda-forge also has `cmake` and `ninja`; that route was not measured.
- `curl`, `tar` and `bzip2` (all present here), and internet access for the downloads below.

The Vulkan headers are not needed from the system, see step 3.

## 1. Install micromamba into your own folder

[micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) is a single-file conda client. Pick a
folder with a few GB free (the environment below is about 500 MB, the package cache about 1 GB more). It can be on
another disk, but not on a `noexec` mount (`findmnt -no OPTIONS <folder>`).

```sh
LEO=$HOME/leonida-tools
mkdir -p "$LEO/bin" "$LEO/mamba"
curl -fsSLo "$LEO/micromamba.tar.bz2" https://micro.mamba.pm/api/micromamba/linux-64/latest
tar -xjf "$LEO/micromamba.tar.bz2" -C "$LEO/bin" --strip-components=1 bin/micromamba
"$LEO/bin/micromamba" --version              # 2.10.0 here
```

The `latest` URL is micromamba's official download endpoint and is not pinned to a version. The archive is saved first so
you can inspect it before extracting. `linux-64` is the x86-64 glibc build, which is what was measured. On another
architecture use the matching archive (for example `linux-aarch64`) from the
[micromamba installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html);
that guide says micromamba does not run natively on musl systems such as Alpine. Nothing else on this page was
tested outside x86-64.

## 2. Create the environment

```sh
export MAMBA_ROOT_PREFIX=$LEO/mamba
"$LEO/bin/micromamba" create -y -n leonida -c conda-forge --override-channels "llvm-tools>=21" spirv-tools
```

This took 6 seconds on a fast connection and installed LLVM 23.1.6 and SPIRV-Tools 2026.4. Put the environment's `bin`
first on `PATH` in each shell where you build or test (no `micromamba activate` and no `.bashrc` edit needed):

```sh
export PATH=$LEO/mamba/envs/leonida/bin:$PATH
llvm-mc --version | head -3
```

Then run the `llvm-mc` check from [testing.md](testing.md#what-you-need) (the `v_lshlrev_b16_e64 ... op_sel` line). With
this environment it prints an `encoding:` line, where an older build says `not a valid operand`.

## 3. Vulkan headers

When this was measured, conda-forge's newest `vulkan-headers` was 1.3.231 (too old), and Fedora ships no headers by
default. Let the project fetch its own pinned, hash-checked copy at configure time instead. The download lands in the
build folder:

```sh
cmake -S runtime -B runtime/build/linux -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLEONIDA_VULKAN_HEADERS_FETCH=ON
```

`gta6rt` itself needs Windows; on Linux CMake reports "only Vulkan tools are available" and builds those.

## 4. Build and run

```sh
cmake --build runtime/build/linux --parallel $(nproc)
```

134 of 134 targets built in 1 minute 34 seconds on 24 cores. Then run the suite from the same checkout, with the
options described in [testing.md](testing.md):

```bash
LLVM_MC=llvm-mc LEONIDA_VK_DEVICE="GTX 1660" \
  python3 -I scripts/run_gpu_tests.py --build runtime/build/linux --work "$LEO/run" --require-pass | tee "$LEO/run.log"
echo "runner exit code: ${PIPESTATUS[0]}"   # 1 when any test fails; a plain `| tee` would report tee's 0 instead
grep 'device:' "$LEO/run.log" | head -3
```

Set `LEONIDA_VK_DEVICE` to part of your GPU's name ("Choosing the GPU" in [testing.md](testing.md)). Without it a
machine that also has Mesa's `llvmpipe` may test the software device; the `grep` shows which device the tests used.

Expected on the machine above, three runs in a row with identical results (a run takes about 90 seconds):

```
246 shaders: 246 translated, 246 pass spirv-val
191/240 passed, 48 skipped
  FAIL  ms_wave_info_s3   at (0,0): got (-1.0000,-1.0000,-1.0000,-1.0000), want (0.0078,0.0078,0.5,1.0)
```

The runner exits with 1 here because of that one failure (it does so whenever a test fails, with or without
`--require-pass`, which only adds an exit code of 1 when nothing passes). Each skip is printed with its reason; here they were tests that need a 64-wide subgroup (this GPU has 32) and tests that
another harness runs. `ms_wave_info_s3` is a known mesh-shader failure (#6172, see
[gpu_test_results.md](gpu_test_results.md)); other GPUs will show different counts. Post yours in
[#166](https://github.com/ItsNotRexed/Leonida/issues/166).

## Cleaning up

Delete `$LEO` and `runtime/build/linux`. micromamba also writes two small files outside that folder:
`~/.conda/environments.txt` (it lists the environment's path) and `~/.mamba/pkgs/urls.txt`. If you do not use conda
otherwise, remove them with `rm -r ~/.conda ~/.mamba`. If you do, delete only those two files, because `rm -r` would
also remove your own environment list.
