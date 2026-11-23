"""tools/mesh_v5_liveness.py on hand-written blocks (no game data): back edges and live-at-entry VGPRs."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import mesh_v5_liveness as ml  # noqa: E402


def v(r):
    return {"k": "vgpr", "r": r}


def ins(op, *args):
    return {"op": op, "args": list(args), "mods": []}


END = ins("s_endpgm")
BR = ins("s_cbranch_scc0", {"k": "imm", "v": 0})   # the edges come from succ, not from the operand


class MeshV5Liveness(unittest.TestCase):
    def test_straight_line_reads_v5_only(self):
        blocks = [[ins("v_readfirstlane_b32", {"k": "sgpr", "r": 0}, v(5)), ins("v_mov_b32", v(1), v(5)),
                   ins("v_add_f32", v(2), v(1), v(1)), END]]
        self.assertEqual(ml.analyse(blocks, [[]]), (False, {5}, {5}, {5}))

    def test_loop_carried_register_written_before_the_loop_is_not_an_input(self):
        # B0: v1 = v5, v3 = 0; B1 (loop): v2 = v3 + v1, v3 = v2, branch back to B1; B2: end
        blocks = [[ins("v_mov_b32", v(1), v(5)), ins("v_mov_b32", v(3), {"k": "imm", "v": 0})],
                  [ins("v_add_u32", v(2), v(3), v(1)), ins("v_mov_b32", v(3), v(2)), BR],
                  [END]]
        succ = [[1], [1, 2], []]
        self.assertEqual(ml.back_edges(succ), {(1, 1)})
        self.assertEqual(ml.analyse(blocks, succ), (True, {5}, {5}, {5}))

    def test_loop_register_never_written_before_the_loop_is_an_input(self):
        blocks = [[ins("v_mov_b32", v(1), v(5))],
                  [ins("v_add_u32", v(2), v(7), v(1)), ins("v_mov_b32", v(7), v(2)), BR],
                  [END]]
        self.assertEqual(ml.analyse(blocks, [[1], [1, 2], []]), (True, {5, 7}, {5, 7}, {5, 7}))

    def test_write_skipped_by_a_branch_is_seen_unlike_the_linear_scan(self):
        # B0 branches over B1's write of v4 to B2, which reads v4: v4 is live at entry, though in listing order
        # its first write comes before its first read.
        blocks = [[ins("v_mov_b32", v(1), v(5)), BR], [ins("v_mov_b32", v(4), v(1))], [ins("v_mov_b32", v(6), v(4)), END]]
        self.assertEqual(ml.rdna2spirv.read_before_write_vgprs([i for b in blocks for i in b]), {5})
        self.assertEqual(ml.analyse(blocks, [[1, 2], [2], []]), (False, {4, 5}, {4, 5}, {4, 5}))

    def test_retreating_edge_of_an_irreducible_graph_changes_the_answer(self):
        # E -> W -> A -> B -> A (retreating), E -> B: A reads v9, W writes it; only E -> B -> A skips the write.
        blocks = [[BR], [ins("v_mov_b32", v(9), v(5))], [ins("v_mov_b32", v(2), v(9))], [BR], ]
        succ = [[1, 3], [2], [3], [2]]
        self.assertEqual(ml.back_edges(succ), {(3, 2)})
        self.assertEqual(ml.analyse(blocks, succ), (True, {5, 9}, {5}, {5, 9}))

    def test_instructions_after_s_endpgm_are_ignored_and_stores_only_read(self):
        blocks = [[ins("global_store_dword", v(0), v(8), {"k": "sgpr", "r": 4}), END, ins("v_mov_b32", v(1), v(3))]]
        self.assertEqual(ml.analyse(blocks, [[]]), (False, {0, 8}, {0, 8}, {0, 8}))

    def test_execz_skip_edge_is_left_out_of_the_last_count(self):
        # pc 0: v1 = v5; pc 4: s_cbranch_execz to pc 12 (simm16 1); pc 8: v4 = v1; pc 12: v6 = v4, end
        def at(pc, i, size=4):
            return dict(i, pc=pc, enc="00" * size)
        blocks = [[at(0, ins("v_mov_b32", v(1), v(5))), at(4, ins("s_cbranch_execz", {"k": "imm", "v": 1}))],
                  [at(8, ins("v_mov_b32", v(4), v(1)))],
                  [at(12, ins("v_mov_b32", v(6), v(4))), at(16, END)]]
        succ = [[2, 1], [2], []]
        self.assertEqual(ml.execz_skip_edges(blocks, succ), {(0, 2)})
        self.assertEqual(ml.analyse(blocks, succ), (False, {4, 5}, {4, 5}, {5}))


if __name__ == "__main__":
    unittest.main()
