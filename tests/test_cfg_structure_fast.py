"""Parity and invariant tests for the RPO-indexed fast paths in tools/cfg_structure.py.

    python -I tests/test_cfg_structure_fast.py

Checks _order_rpo, _dominators_rpo, _order, _dominators, and structure against a verbatim copy of the
previous dict-based implementation across random reducible/irreducible CFGs, multi-level nested natural
loops, unreachable blocks, and multi-thousand-block diamond/loop/exit shader CFGs. Standard library only.
"""
from __future__ import annotations

import collections
import os
import random
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cfg_structure as S  # noqa: E402


def _ref_order(succ):
    state, order, back = {0: 1}, [], []
    stack = [(0, iter(succ[0]))]
    while stack:
        v, it = stack[-1]
        for w in it:
            if w not in state:
                state[w] = 1
                stack.append((w, iter(succ[w])))
                break
            if state[w] == 1:
                back.append((v, w))
        else:
            state[v] = 2
            order.append(v)
            stack.pop()
    rpo = order[::-1]
    preds = collections.defaultdict(list)
    for v in rpo:
        for w in succ[v]:
            preds[w].append(v)
    return rpo, back, preds


def _ref_dominators(rpo, preds):
    pos = {v: i for i, v in enumerate(rpo)}
    idom = {rpo[0]: rpo[0]}
    changed = True
    while changed:
        changed = False
        for v in rpo[1:]:
            new = None
            for p in preds[v]:
                if p not in idom:
                    continue
                if new is None:
                    new = p
                    continue
                a, b = p, new
                while a != b:
                    while pos[a] > pos[b]:
                        a = idom[a]
                    while pos[b] > pos[a]:
                        b = idom[b]
                new = a
            if idom.get(v) != new:
                idom[v] = new
                changed = True
    return idom, pos


def _ref_structure(succ):
    rpo, back, preds = _ref_order(succ)
    idom, pos = _ref_dominators(rpo, preds)

    def dominates(a, b):
        while pos[b] > pos[a]:
            b = idom[b]
        return a == b

    if not all(dominates(w, v) for v, w in back):
        return None

    bodies = {}
    for v, h in back:
        body = bodies.setdefault(h, {h})
        work = [v]
        while work:
            x = work.pop()
            if x not in body:
                body.add(x)
                work.extend(preds[x])
    inner = {}
    for h in sorted(bodies, key=lambda h: len(bodies[h])):
        for x in bodies[h]:
            inner.setdefault(x, h)
    outer = {}
    for h in bodies:
        around = [g for g in bodies if g != h and h in bodies[g]]
        outer[h] = min(around, key=lambda g: len(bodies[g])) if around else None

    def member(x, region):
        loop = inner.get(x)
        if loop == region:
            return ("b", x)
        while outer[loop] != region:
            loop = outer[loop]
        return ("l", loop)

    children = collections.defaultdict(list)
    for x in rpo[1:]:
        region = inner.get(x)
        if region == x:
            children[member(idom[x], outer[x])].append(("l", x))
        else:
            children[member(idom[x], region)].append(("b", x))

    max_guard_depth = 16
    root = []
    start = ("l", 0) if 0 in bodies else ("b", 0)
    stack = [("emit", root, None, start, 0)]
    while stack:
        task = stack.pop()
        if task[0] == "open_guard":
            _, parent_list, m, depth = task
            gbody = []
            parent_list.append(("guard", m[1], gbody))
            stack.append(("emit", gbody, parent_list, m, depth))
            continue
        _, out_list, parent_list, m, depth = task
        kind, x = m
        below = sorted(children[m], key=lambda c: pos[c[1]])
        if kind == "l":
            lbody = []
            out_list.append(("loop", x, lbody))
        else:
            out_list.append(("block", x))
        if kind == "b" and len(below) == 1 and list(succ[x]) == [below[0][1]]:
            stack.append(("emit", out_list, parent_list, below[0], depth))
        else:
            hoist = (parent_list is not None and below
                     and (len(below) >= 2 or kind == "l" or depth >= max_guard_depth))
            if hoist:
                stack.append(("open_guard", parent_list, below[-1], depth))
                inner_b = below[:-1]
            else:
                inner_b = below
            for c in reversed(inner_b):
                stack.append(("open_guard", out_list, c, depth + 1))
        if kind == "l":
            stack.append(("emit", lbody, None, ("b", x), 0))
    return root


def _random_graph(rng):
    n = rng.randint(1, 16)
    kind = rng.random()
    succ = []
    for v in range(n):
        k = rng.choice((0, 1, 1, 2, 2, 2))
        if kind < 0.5:
            succ.append([rng.randrange(n) for _ in range(k)])
        else:
            succ.append([rng.randrange(v + 1, n) if v + 1 < n and rng.random() < 0.75 else rng.randrange(v + 1)
                         for _ in range(k)])
    return succ


class CfgStructureFastTests(unittest.TestCase):
    def test_order_rpo_and_order_parity(self):
        rng = random.Random(20261123)
        shapes = [
            [[]],
            [[1, 2], [3], [3], [], [1, 2]],  # block 4 is unreachable
            [[0, 1], [1, 2], []],
            [[1], [2, 3], [1, 4], [2, 4], []],
        ] + [_random_graph(rng) for _ in range(500)]
        for idx, g in enumerate(shapes):
            with self.subTest(idx=idx):
                rpo, pos, back, preds_i = S._order_rpo(g)
                ref_rpo, ref_back, ref_preds = _ref_order(g)
                self.assertEqual(rpo, ref_rpo)
                self.assertEqual(back, ref_back)
                for vi, v in enumerate(rpo):
                    self.assertEqual(pos[v], vi)
                    self.assertEqual([rpo[pi] for pi in preds_i[vi]], ref_preds.get(v, []))
                for v in range(len(g)):
                    if v not in ref_rpo:
                        self.assertEqual(pos[v], -1)
                got_rpo, got_back, got_preds = S._order(g)
                self.assertEqual((got_rpo, got_back, dict(got_preds)), (ref_rpo, ref_back, dict(ref_preds)))

    def test_dominators_rpo_single_pass_dag_and_cyclic_parity(self):
        # 1. Early-exit DAG where common exit block has 400 predecessors (all dominated by root 0).
        with self.subTest(case="early_exit_dag_400"):
            n_exits = 400
            exits = [[k + 1, n_exits] for k in range(n_exits)] + [[]]
            rpo, pos, back, preds_i = S._order_rpo(exits)
            self.assertEqual(back, [])
            idom_dag = S._dominators_rpo(len(rpo), preds_i, has_back=False)
            idom_gen = S._dominators_rpo(len(rpo), preds_i, has_back=True)
            self.assertEqual(idom_dag, idom_gen)
            ref_rpo, _, ref_preds = _ref_order(exits)
            ref_idom, ref_pos = _ref_dominators(ref_rpo, ref_preds)
            self.assertEqual({rpo[vi]: rpo[idom_dag[vi]] for vi in range(len(rpo))}, ref_idom)
            self.assertEqual(S._dominators(ref_rpo, ref_preds), (ref_idom, ref_pos))

        # 2. Random DAGs and cyclic graphs, plus unreachable-predecessor tolerance in _dominators.
        rng = random.Random(20261124)
        for idx in range(500):
            with self.subTest(idx=idx):
                g = _random_graph(rng)
                rpo_g, _, back_g, preds_ig = S._order_rpo(g)
                idom_ig = S._dominators_rpo(len(rpo_g), preds_ig, bool(back_g))
                ref_rpo_g, _, ref_preds_g = _ref_order(g)
                ref_idom_g, ref_pos_g = _ref_dominators(ref_rpo_g, ref_preds_g)
                self.assertEqual({rpo_g[vi]: rpo_g[idom_ig[vi]] for vi in range(len(rpo_g))}, ref_idom_g)
                ref_preds_with_dead = collections.defaultdict(list, {k: list(v) + [len(g) + 9]
                                                                     for k, v in ref_preds_g.items()})
                self.assertEqual(S._dominators(ref_rpo_g, ref_preds_with_dead), (ref_idom_g, ref_pos_g))

    def test_laminar_nested_loops_and_presorted_children(self):
        # 1. Three-level nested natural loop with sibling inner loops and loop header at block 0:
        #    Loop 0 -> Loop 1 -> (Loop 2, Loop 4) -> exit 6.
        nested_loops = [
            [1],        # 0: outer loop header
            [2, 3],     # 1: middle loop header
            [2, 4],     # 2: inner sibling loop A
            [4],        # 3: branch inside middle loop
            [4, 5],     # 4: inner sibling loop B
            [1, 0, 6],  # 5: back-edges to 1 and 0, or exit to 6
            [],         # 6: exit
        ]
        loops = [[k, k + 1] for k in range(300)] + [[]]
        diamonds = []
        for k in range(300):
            b = 3 * k
            diamonds.extend([[b + 1, b + 2], [b + 3], [b + 3]])
        diamonds.append([])

        orig_order, orig_dom = S._order, S._dominators
        try:
            def _forbid_legacy_dict_pass(*_a, **_kw):
                raise AssertionError("structure() should use RPO-indexed _order_rpo/_dominators_rpo directly")

            S._order = _forbid_legacy_dict_pass
            S._dominators = _forbid_legacy_dict_pass
            for name, g in (("nested_loops", nested_loops), ("loops", loops), ("diamonds", diamonds)):
                with self.subTest(name=name):
                    self.assertEqual(S.structure(g), _ref_structure(g))
        finally:
            S._order, S._dominators = orig_order, orig_dom

    def test_structure_parity_across_5000_random_graphs(self):
        rng = random.Random(20261125)
        structured = refused = 0
        orig_order, orig_dom = S._order, S._dominators
        try:
            def _forbid_legacy_dict_pass(*_a, **_kw):
                raise AssertionError("structure() should use RPO-indexed _order_rpo/_dominators_rpo directly")

            S._order = _forbid_legacy_dict_pass
            S._dominators = _forbid_legacy_dict_pass
            for idx in range(5000):
                g = _random_graph(rng)
                got = S.structure(g)
                want = _ref_structure(g)
                if got != want:
                    self.fail(f"graph {idx} ({g}): got {got}, want {want}")
                if got is None:
                    refused += 1
                else:
                    structured += 1
        finally:
            S._order, S._dominators = orig_order, orig_dom
        self.assertGreater(structured, 1000)
        self.assertGreater(refused, 200)


if __name__ == "__main__":
    unittest.main()
