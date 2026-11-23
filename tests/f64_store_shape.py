"""Where a translated VALU result lands, for tests that follow a value's data flow into its destination VGPR.

Since #4473 the translator writes a VGPR in one of two shapes, and neither is the `OpSelect %uint active new old`
these tests used to look for:

* a shader that never writes EXEC stores the result directly (the per-lane guard is statically true);
* a shader that writes EXEC promotes the register to its own variable (`%reg_vN`) and stores the result inside an
  `if (%active)` block.

`assert_stored_to_vgpr` accepts a store of `word` to VGPR `n` in either storage (`OpAccessChain %vgpr %cN` or
`%reg_vN`), or through an `OpSelect` that picks `word` (the pre-#4473 shape), so a test pins the data flow and not one
optimisation. `PREFIXES` runs a test both ways: as written, and after an EXEC write that depends on an input register.
"""
import re

# ("name", listing prefix): the second makes EXEC data-dependent, so the translator cannot treat the guard as true.
PREFIXES = (
    ("no EXEC write", ""),
    ("EXEC from v2 != 0", "v_cmp_ne_u32_e32 vcc_lo, 0, v2\ns_and_b32 exec_lo, exec_lo, vcc_lo\n"),
)


def assert_stored_to_vgpr(test, spv, word, n):
    """Fail `test` unless SPIR-V `spv` stores the 32-bit `word` (an SSA id) to VGPR `n`."""
    targets = set(re.findall(rf"(?m)^ *(%\w+) = OpAccessChain %_ptr_Function_uint %vgpr %c{n}$", spv))
    targets.add(f"%reg_v{n}")
    stores = [value for target, value in re.findall(r"(?m)^ *OpStore (%\w+) (%\w+)$", spv) if target in targets]
    for value in stores:
        if value == word or re.search(rf"(?m)^ *{value} = OpSelect %uint %\w+ {word} %\w+$", spv):
            return
    test.fail(f"{word} is not stored to v{n} (stores to v{n}: {stores or 'none'})")
