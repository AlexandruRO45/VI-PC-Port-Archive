# Leonida Shader Semantic Test Coverage: Master Directive

## 1. Project Overview & Mission

This directive guides the implementation of semantic test suites and regression tests for RDNA2 shader instruction translation in **ItsNotRexed/Leonida** (`tests/semantic/`).

Our goal is to build comprehensive, verified, translator-backed regression tests for RDNA2 instructions (VALU, SALU, LDS, Buffer Atomics) to validate that `tools/rdna2spirv.py` generates valid, semantically correct SPIR-V.

---

## 2. Hard Quality & Mergeability Rules (MANDATORY)

1. **Every Test MUST Exercise `tools/rdna2spirv.py`**:
   - Upstream enforces `tests/test_semantic_translator_use.py`: tests that do not reach `tools/rdna2spirv.py` are automatically rejected.
   - Every test file MUST import `support` (or `rdna2spirv`) and call `support.translate_lines(...)`.
   - DO NOT create private standalone AST or scalar interpreters that model instructions purely in Python without exercising the repository translator.
2. **Translate & Execute Pipeline**:
   - **VALU operations**: Construct small hand-written assembly listings, translate them via `support.translate_lines("ps_<topic>", prog)`, and verify the executed results via `support.mrt0_words(...)`.
   - **LDS / Memory operations**: Translate compute listings via `support.translate_lines("cs_<topic>", lines, fssm=FSSM)` and run via `spirv_mini_interp.run_fragment` checking memory readbacks.
   - Verify translated SPIR-V opcodes, bit manipulation, sign extensions, and IEEE 754 edge cases against reference mathematical oracles.
3. **Strictly Zero Game Data**:
   - Never commit `.rpf`, `.awc`, `.bik`, shader bytecode, assembly dumps, `.exe`, memory dumps, or decryption keys.
   - All tests must construct inputs purely in-memory using Python constants or procedural synthesis.
4. **Standard Library Python Only (`python -I`)**:
   - Test suites must run cleanly under `python -I` (isolated mode, standard library only).
   - Zero external pip dependencies (`numpy`, `pytest`, `torch` are forbidden). Use standard `unittest`, `struct`, `math`.
5. **Fresh Branching from `upstream/main`**:
   - Always branch directly from `upstream/main` to guarantee zero merge conflicts:
     ```powershell
     git fetch upstream main
     git worktree add -b test/semantic-<topic> work/<topic> upstream/main
     ```
   - Verify mergeability immediately after PR creation:
     ```powershell
     gh pr view <pr_number> --json mergeable
     ```
6. **Zero Meta-Jargon**:
   - Act as genuine, professional open-source software contributors.
   - Never use phrases like "atomic commits", "pipeline steps", arbitrary target metrics, or swarm directives in commit logs, PR titles, or PR descriptions.

---

## 3. High-Efficiency 1-Turn Programmatic Execution

To maximize output and minimize token consumption:
- Write a single Python or PowerShell staging script that programmatically writes the test file, stages it, and runs `git commit`.
- Run single-file pre-flight verification:
  ```powershell
  python -I tests/test_semantic_translator_use.py
  python -I tests/semantic/test_<topic>.py
  ```
- Push to origin and open the PR in the same single turn:
  ```powershell
  git push -u origin test/semantic-<topic>
  gh pr create --repo ItsNotRexed/Leonida --base main --title "test(semantic): add translator tests for <topic>" --body-file pr_body.md
  ```

---

## 4. Standard PR Description Template (UTF-8)

Save to `pr_body.md` using UTF-8 encoding (no BOM):

```markdown
### What
Adds translator-backed regression tests for `<topic>` in `tests/semantic/test_<topic>.py`.

Covers:
- Hand-written RDNA2 shader assembly translation through `tools/rdna2spirv.py`
- Execution of translated SPIR-V via `spirv_mini_interp`
- Reference mathematical verification across standard and edge-case inputs
- Boundary checks, IEEE 754 float behavior, and modifier validation

### Tested
- Tested with Python 3.9+ (`python -I`)
- `python -I tests/test_semantic_translator_use.py` (Passes translator reachability check)
- `python -I tests/semantic/test_<topic>.py` (All tests passed, OK)
- Standard library Python only (`unittest`, `struct`, `math`)
- Clean merge against `upstream/main`
- Strictly zero game data, binaries, or proprietary keys
```

---

## 5. Instruction Topics (VALU, SALU, LDS)

### VALU Instructions
- Double-Precision: `v_fma_f64`, `v_add_f64`, `v_mul_f64`, `v_min_f64`, `v_max_f64`, `v_ldexp_f64`, `v_frexp_exp_i32_f64`, `v_fract_f64`, `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64`, `v_div_scale_f64`, `v_div_fmas_f64`, `v_div_fixup_f64`
- 16-Bit / Packed Float: `v_fma_f16`, `v_add_f16`, `v_sub_f16`, `v_subrev_f16`, `v_mul_f16`, `v_rcp_f16`, `v_rsq_f16`, `v_sqrt_f16`, `v_fract_f16`, `v_dot2_f32_f16`, `v_pk_fma_f16`, `v_pk_add_f16`, `v_pk_mul_f16`, `v_pk_min_f16`, `v_pk_max_f16`
- Conversions & Bitwise: `v_cvt_f32_f64`, `v_cvt_f64_f32`, `v_cvt_f16_f32`, `v_cvt_f32_f16`, `v_cvt_i32_f32`, `v_cvt_f32_i32`, `v_cvt_u32_f32`, `v_cvt_f32_u32`, `v_and_or_b32`, `v_or3_b32`, `v_xor3_b32`, `v_xnor_b32`, `v_alignbyte_b32`, `v_alignbit_b32`, `v_perm_b32`

### SALU Instructions
- `s_and_saveexec_b64`, `s_or_saveexec_b64`, `s_xor_saveexec_b64`, `s_andn2_saveexec_b64`, `s_orn2_saveexec_b64`, `s_nand_saveexec_b64`, `s_nor_saveexec_b64`, `s_xnor_saveexec_b64`, `s_brev_b32`, `s_mulk_i32`, `s_absdiff_i32`

### LDS Instructions
- `ds_add_u32`, `ds_sub_u32`, `ds_min_u32`, `ds_max_u32`, `ds_min_i32`, `ds_max_i32`, `ds_and_b32`, `ds_or_b32`, `ds_xor_b32`, `ds_wrxchg_rtn_b32`, `ds_read_u16`, `ds_read_i16`, `ds_write_b16`, `ds_read_b32`, `ds_write_b32`
