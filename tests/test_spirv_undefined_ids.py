"""Check the translator's SPIR-V text for ids it references but never defines, without SPIRV-Tools.

``tests/test_u16_high_half.py::_validate_spv`` assembles and validates the translator's output, but returns early
when ``spirv-as`` / ``spirv-val`` are not on PATH (``conftest.HAVE_SPIRV`` is false). The module can then reference
an id that no line defines and nothing notices. ``aa2988cc`` built ``%fmt_size_arr`` with ``gen.cu(128)`` after
``assemble()`` had already dumped the ``OpConstant %uint`` list, so a shader doing a dynamic-format buffer store
declared that array type from a ``%c80`` that was never defined. The module is invalid: SPIRV-Tools' assembler
encodes it and ``spirv-val`` rejects the binary (“Operand ... requires a previous definition”), but no test that
does not call the tools looks. That is the CI blind spot described in #4466.

This test needs no llvm-mc, spirv-as or spirv-val: ``rdna2spirv.translate()`` returns the module as text.

Two checks, both on the module text:

* ``undefined_ids``: every ``%name`` a line references must be defined by some ``%name = Op...`` line. Unlike
  ordering, a missing definition is invalid for *any* id (labels and function-local ids cannot be created by a
  forward reference either: the reference has to name a definition that exists), so this check is applied to
  every ``%name``, not only the ``%cXX`` constants.
* ``used_before_definition``: a type or constant used by another type/constant definition before the line that
  defines it. Those operands can never be forward references (SPIR-V allows ``OpTypeForwardPointer``, which the
  translator does not emit and which is skipped below), so this catches the other half of the class: an id that
  is defined, but only after the type that consumes it.

Limits: this is not a SPIR-V validator. It checks id existence and type/constant declaration order, not operand
types, result types, dominance, duplicate definitions, decorations or instruction form, so a module without
dangling ids can still be rejected by the tools. It is the floor for machines and CI jobs that have neither
``spirv-as`` nor ``spirv-val``, not a replacement for validating where they are installed.
"""
import os
import re
import struct
import tempfile

import pytest

import rdna2spirv
from listing import EMPTY_DXBC, translate_listing

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "semantic", "fixtures", "disasm")

# COMPUTE_PGM_RSRC2 = 0x8088 (4 user SGPRs, TGID_X, 512 B LDS) and a 64x1x1 group: the synthetic FSSM header
# tests/semantic/test_value_semantics.py gives the cs_fmt_* fixtures. Only its RSRC2/threads words are read.
CS_HEADER = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)

_ID = re.compile(r"%[A-Za-z0-9_]+")
_DEFINITION = re.compile(r"^\s*(%[A-Za-z0-9_]+)\s*=\s*(Op[A-Za-z0-9_]+)\b")
# OpTypeForwardPointer is the one type declaration that may reference a later id; the translator does not emit
# it, but leave it out of the order check so the helper stays correct if that changes.
_TYPE_OR_CONSTANT = ("OpType", "OpConstant", "OpSpecConstant")


def _definitions(module):
    """({id: opcode}, {id: first line}) for every "%id = Op..." definition in ``module``."""
    opcodes, lines = {}, {}
    for number, line in enumerate(module.splitlines(), 1):
        match = _DEFINITION.match(line)
        if match:
            opcodes.setdefault(match.group(1), match.group(2))
            lines.setdefault(match.group(1), number)
    return opcodes, lines


def _references(module):
    """{id: first line it is used on}: every %id the module references, not counting result ids.

    The result id of a definition (the "%id =" at the start of its line) is skipped; a ";" comment is dropped
    first so a comment cannot fake a reference.
    """
    references = {}
    for number, line in enumerate(module.splitlines(), 1):
        line = line.split(";", 1)[0]
        match = _DEFINITION.match(line)
        if match:
            line = line[match.end(1):]
        for token in _ID.findall(line):
            references.setdefault(token, number)
    return references


def undefined_ids(module):
    """Sorted ids the module references but no "%id = Op..." line defines (``[]`` for a self-consistent module)."""
    definitions, _ = _definitions(module)
    return sorted(token for token in _references(module) if token not in definitions)


def used_before_definition(module):
    """[(id, use_line, definition_line)] of type/constant ids used before the line that defines them.

    Only type and constant definition lines are checked: their operands are ids that must already be declared.
    Labels, branch/switch/merge targets, OpPhi operands, OpEntryPoint's %main and function ids are legal forward
    references and are deliberately not checked.
    """
    _, definition_lines = _definitions(module)
    declared, problems = set(), []
    for number, line in enumerate(module.splitlines(), 1):
        line = line.split(";", 1)[0]
        match = _DEFINITION.match(line)
        if not match:
            continue
        name, opcode = match.group(1), match.group(2)
        if opcode.startswith(_TYPE_OR_CONSTANT) and opcode != "OpTypeForwardPointer":
            for token in _ID.findall(line[match.end(1):]):
                if token != name and token not in declared:
                    problems.append((token, number, definition_lines.get(token)))
        declared.add(name)
    return problems


def translate_compute_fixture(name):
    """SPIR-V assembly for tests/semantic/fixtures/disasm/<name>.s with a synthetic FSSM header.

    Same input layout as tests/semantic/support.py's translate_fixture, reimplemented here so this test file
    depends only on the standard library and on the translator it guards.
    """
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "disasm"))
        with open(os.path.join(FIXTURE_DIR, name + ".s"), "rb") as f:
            listing = f.read()
        with open(os.path.join(tmp, "disasm", name + ".s"), "wb") as f:
            f.write(listing)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(EMPTY_DXBC)
        os.makedirs(os.path.join(tmp, "fssm"))
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(CS_HEADER)
        return rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))


def assert_self_consistent(name, module):
    """Assert the module text passes both checks, with the first dangling/forward id in the message."""
    undefined = undefined_ids(module)
    assert undefined == [], (
        f"{name}: the module references ids it never defines: {undefined}. spirv-val rejects such a module "
        "(\"Operand ... requires a previous definition\"); see #4466.")
    forward = used_before_definition(module)
    assert forward == [], (
        f"{name}: types/constants used before their definition (id, use line, definition line): {forward}")
    assert module.strip(), f"{name}: the translator produced an empty module"


# Store-only dynamic-format listings: the #4466 reproducer. The V#'s FORMAT is unknown at translation time, so
# assemble() emits %fmt_size_tab and builds %fmt_size_arr from gen.cu(128); on the parent commit that constant is
# registered after the OpConstant %uint dump, so %c80 is referenced and never defined. The first body is the
# minimized sequence from tests/test_u16_high_half.py::test_d16_buffer_store_no_redundant_extract_after_shr16,
# the second the buffer-store form of tests/test_translate.py::test_normalized_buffer_store_nan (the same class,
# no d16 modifier).
LISTINGS = [
    pytest.param(
        "ps_store_fmt_d16_xy",
        "v_mov_b32 v1, 0x9c004000\n"
        "v_mov_b32 v2, 0\n"
        "buffer_store_format_d16_xy v1, v2, s[0:3], 0 offen",
        ("%fmt_size_tab",),
        id="dynamic-format-store-d16-xy",
    ),
    pytest.param(
        "ps_store_fmt_xyzw",
        "v_mov_b32 v1, 0x11111111\n"
        "v_mov_b32 v2, 0\n"
        "buffer_store_format_xyzw v[1:4], v2, s[0:3], 0 offen",
        ("%fmt_size_tab",),
        id="dynamic-format-store-xyzw",
    ),
    # The load side of the same table. It does not reproduce #4466 on the parent commit (a dynamic load emits a
    # format-decode function whose constants include the 128 the array type needs, before assemble() runs), but
    # it exercises %fmt_size_tab from the other direction and must keep passing.
    pytest.param(
        "ps_load_fmt_d16_xyzw",
        "v_mov_b32 v1, 0\n"
        "v_mov_b32 v2, 0\n"
        "buffer_load_format_d16_xyzw v[4:5], v1, s[0:3], 0 offen\n"
        "s_waitcnt vmcnt(0)",
        ("%fmt_size_tab",),
        id="dynamic-format-load-d16-xyzw",
    ),
    # Controls: modules that use no format table at all. They keep the check from being tested only on the
    # table path and show a plain module (and the fixed-format MTBUF path, whose size is an instruction literal)
    # passes it.
    pytest.param(
        "ps_plain",
        "v_mov_b32 v8, 0x3f800000",
        (),
        id="control-plain-pixel-shader",
    ),
    pytest.param(
        "ps_tbuffer_fixed_fmt",
        "v_mov_b32 v1, 0\n"
        "v_mov_b32 v2, 0\n"
        "tbuffer_store_format_d16_xy v1, v2, s[0:3], 0 offen",
        (),
        id="control-fixed-format-tbuffer-store",
    ),
]


@pytest.mark.parametrize("name,body,required", LISTINGS)
def test_listing_module_has_no_undefined_ids(name, body, required, tmp_path):
    module = translate_listing(tmp_path, body, name)
    for token in required:
        # If the fixture stops building the table it would no longer cover #4466; fail instead of passing quietly.
        assert token in module, f"{name}: the module no longer uses {token}; the listing stopped exercising it"
    assert_self_consistent(name, module)


# Real semantic fixtures that use typed buffer loads and stores. Both contain a dynamic-format *load* as well as
# stores, so on the parent commit their decode function defines every table constant before the dump and they do
# not reproduce #4466 -- the store-only listings above are the reproducer. They are kept because a 7-9k line
# module from the existing fixture set is a stronger false-positive control than the hand-written listings.
COMPUTE_FIXTURES = ["cs_fmt_1010102", "cs_fmt_d16_xyz"]


@pytest.mark.parametrize("name", COMPUTE_FIXTURES)
def test_existing_compute_fixture_has_no_undefined_ids(name):
    module = translate_compute_fixture(name)
    assert "%fmt_size_tab" in module, f"{name}: the fixture no longer exercises the dynamic-format table"
    assert_self_consistent(name, module)


# Minimal modules for the checks themselves: a forward reference to %main and to a label is legal SPIR-V, while
# the generated module in #4466 referenced a constant that was never defined.
_FORWARD_REFERENCES = """\
OpCapability Shader
OpEntryPoint Fragment %main "main" %out_mrt0
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%fn = OpTypeFunction %void
%_ptr_Output_v4 = OpTypePointer Output %v4float
%out_mrt0 = OpVariable %_ptr_Output_v4 Output
%f0 = OpConstant %float 0
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %tail
%tail = OpLabel
%value = OpCompositeConstruct %v4float %f0 %f0 %f0 %f0
OpStore %out_mrt0 %value
OpReturn
OpFunctionEnd
"""


def test_checker_accepts_legal_forward_references():
    """%main and the branch target are used before their definition: legal, so neither check may flag them."""
    assert undefined_ids(_FORWARD_REFERENCES) == []
    assert used_before_definition(_FORWARD_REFERENCES) == []


# A type definition that references a constant: with the constant missing (the #4466 shape), or with the
# constant defined after the type (what registering it "too late" looks like).
_TYPE_BEFORE_CONSTANT = """\
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%fmt_size_arr = OpTypeArray %uint {length}
{definition}"""


@pytest.mark.parametrize("token", ["%c80", "%fmt_size_len", "%t9000"])
def test_checker_reports_any_undefined_id(token):
    """The check is generic: any %name, not only the %c<hex> constants, is reported when undefined."""
    module = _TYPE_BEFORE_CONSTANT.format(length=token, definition="")
    assert undefined_ids(module) == [token]


def test_checker_reports_a_constant_defined_after_its_type():
    """An id that is defined too late still breaks the module; the definedness check alone would miss it."""
    module = _TYPE_BEFORE_CONSTANT.format(length="%c80", definition="%c80 = OpConstant %uint 128\n")
    assert undefined_ids(module) == []
    problems = used_before_definition(module)
    assert [problem[0] for problem in problems] == ["%c80"]
    _, use_line, definition_line = problems[0]
    assert use_line < definition_line, "the type must be reported as using the constant before its definition"
