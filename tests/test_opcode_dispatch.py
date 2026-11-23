"""Reject repeated literal opcode cases in Gen._insn's if/elif chains.

    python -I tests/test_opcode_dispatch.py

A repeated opcode makes its later branch unreachable, as happened when two class-compare implementations both
handled v_cmp_class_f64 / v_cmpx_class_f64. Read the AST without importing the translator or requiring game files,
LLVM, SPIRV-Tools or a GPU. Only bare ``op == '...'`` and ``op in (...)`` conditions with literal strings count:
operand/stage guards can intentionally handle the same opcode, and separate chains are independent.
"""
import ast
import os
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def literal_opcodes(test):
    """The opcodes an entire condition matches, or an empty set for a guarded/dynamic condition."""
    if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "op"
            and len(test.ops) == 1):
        return set()
    value = test.comparators[0]
    if isinstance(test.ops[0], ast.Eq) and isinstance(value, ast.Constant) and isinstance(value.value, str):
        return {value.value}
    if (isinstance(test.ops[0], ast.In) and isinstance(value, (ast.Tuple, ast.List, ast.Set))
            and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts)):
        return {item.value for item in value.elts}
    return set()


def repeated_opcodes(function):
    """[(opcode, first line, repeated line)] for bare literal cases in the same if/elif chain."""
    branches = [node for node in ast.walk(function) if isinstance(node, ast.If)]
    # ast represents an elif as the only If in the previous branch's orelse. Scan each chain once; an If inside
    # a branch's body starts a separate chain and must not inherit its parent's cases.
    elif_nodes = {id(node.orelse[0]) for node in branches
                  if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)}
    repeated = []
    for head in branches:
        if id(head) in elif_nodes:
            continue
        seen = {}
        branch = head
        while True:
            for opcode in sorted(literal_opcodes(branch.test)):
                if opcode in seen:
                    repeated.append((opcode, seen[opcode], branch.lineno))
                else:
                    seen[opcode] = branch.lineno
            if len(branch.orelse) != 1 or not isinstance(branch.orelse[0], ast.If):
                break
            branch = branch.orelse[0]
    return repeated


def sample(body):
    """A hand-written dispatcher for the checker's positive and negative controls."""
    source = "def dispatch(op, stage=None):\n" + textwrap.indent(textwrap.dedent(body), "    ")
    return ast.parse(source).body[0]


class OpcodeDispatchTests(unittest.TestCase):
    def test_repeated_equality_reports_both_lines(self):
        function = sample("""\
            if op == 'v_alpha_b32': pass
            elif op == 'v_beta_b32': pass
            elif op == 'v_alpha_b32': pass
            elif op == 'v_alpha_b32': pass
        """)
        self.assertEqual(repeated_opcodes(function), [("v_alpha_b32", 2, 4), ("v_alpha_b32", 2, 5)])

    def test_membership_intersections_and_equality_in_both_orders(self):
        conditions = [
            ("op == 'v_alpha_b32'", "op in ('v_beta_b32', 'v_alpha_b32')"),
            ("op in ('v_alpha_b32', 'v_beta_b32')", "op == 'v_beta_b32'"),
            ("op in ['v_alpha_b32', 'v_beta_b32']", "op in {'v_beta_b32', 'v_gamma_b32'}"),
        ]
        for first, second in conditions:
            with self.subTest(first=first, second=second):
                function = sample(f"if {first}: pass\nelif {second}: pass\n")
                expected = "v_alpha_b32" if first.startswith("op ==") else "v_beta_b32"
                self.assertEqual(repeated_opcodes(function), [(expected, 2, 3)])

    def test_distinct_cases_pass(self):
        function = sample("""\
            if op == 'v_alpha_b32': pass
            elif op in ('v_beta_b32', 'v_gamma_b32'): pass
            else: raise ValueError(op)
        """)
        self.assertEqual(repeated_opcodes(function), [])

    def test_one_branch_can_repeat_multiple_opcodes(self):
        function = sample("""\
            if op in ('v_beta_b32', 'v_alpha_b32'): pass
            elif op in ('v_alpha_b32', 'v_beta_b32'): pass
        """)
        self.assertEqual(repeated_opcodes(function), [("v_alpha_b32", 2, 3), ("v_beta_b32", 2, 3)])

    def test_stage_and_operand_guards_are_not_bare_cases(self):
        for guard in ("op == 'v_alpha_b32' and stage == 'ps'",
                      "op in ('v_alpha_b32',) and operands",
                      "op == 'v_alpha_b32' or stage == 'ps'"):
            for first, second in ((guard, "op == 'v_alpha_b32'"), ("op == 'v_alpha_b32'", guard)):
                with self.subTest(first=first, second=second):
                    self.assertEqual(repeated_opcodes(sample(f"if {first}: pass\nelif {second}: pass\n")), [])

    def test_guarded_branch_does_not_hide_a_later_duplicate(self):
        function = sample("""\
            if op == 'v_alpha_b32': pass
            elif op == 'v_beta_b32' and stage == 'ps': pass
            elif op == 'v_alpha_b32': pass
        """)
        self.assertEqual(repeated_opcodes(function), [("v_alpha_b32", 2, 4)])

    def test_separate_and_nested_chains_do_not_share_cases(self):
        function = sample("""\
            if op == 'v_alpha_b32':
                if op == 'v_alpha_b32': pass
            if op == 'v_alpha_b32': pass
        """)
        self.assertEqual(repeated_opcodes(function), [])

    def test_nested_chain_is_checked_once(self):
        function = sample("""\
            if stage == 'ps':
                if op == 'v_alpha_b32': pass
                elif op == 'v_beta_b32': pass
                elif op == 'v_beta_b32': pass
        """)
        self.assertEqual(repeated_opcodes(function), [("v_beta_b32", 4, 5)])

    def test_dynamic_and_unrelated_conditions_are_ignored(self):
        for condition in ("op in KNOWN_OPS", "op in ('v_alpha_b32', other)", "other == 'v_alpha_b32'",
                          "'v_alpha_b32' in op", "op.startswith('v_alpha')", "op != 'v_alpha_b32'",
                          "op == other == 'v_alpha_b32'", "op in (1, 2)"):
            with self.subTest(condition=condition):
                function = sample(f"if {condition}: pass\nelif op == 'v_alpha_b32': pass\n")
                self.assertEqual(repeated_opcodes(function), [])

    def test_repeated_entries_within_one_membership_are_one_case(self):
        self.assertEqual(repeated_opcodes(sample("if op in ('v_alpha_b32', 'v_alpha_b32'): pass\n")), [])

    def test_translator_has_no_repeated_literal_dispatch_cases(self):
        path = os.path.join(ROOT, "tools", "rdna2spirv.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        gen = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Gen")
        function = next(node for node in gen.body if isinstance(node, ast.FunctionDef) and node.name == "_insn")
        self.assertTrue(any(literal_opcodes(node.test) for node in ast.walk(function) if isinstance(node, ast.If)),
                        "Gen._insn has no bare literal opcode cases; update the dispatcher check")
        repeated = repeated_opcodes(function)
        self.assertEqual(repeated, [], "Repeated opcode cases in tools/rdna2spirv.py Gen._insn: " +
                         "; ".join(f"{op} at lines {first} and {again}" for op, first, again in repeated))


if __name__ == "__main__":
    unittest.main()
