"""tools/gs_compute_stages.py: the scatter and the fused pass, the generated stages of the capture path (SHADER-1092)."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gs_compute_stages as stages  # noqa: E402


# gs_tri's index.json gs_compute layout, the shape the runner's fixture publishes.
LAY = {
    "verts_in": 3,
    "max_verts": 3,
    "num_inputs": 2,
    "num_outputs": 2,
    "inputs": ["position", "location 1"],
    "outputs": ["position", "location 1"],
    "topology": "triangle_strip",
}


def defs(text):
    """id -> its operands, for '%id = OpX a b ...' lines."""
    out = {}
    for m in re.finditer(r"^(%\w+) = (Op\w+)(?: ([^\n]*))?$", text, re.M):
        out[m.group(1)] = (m.group(2), m.group(3) or "")
    return out


def hops(d, id_, count=6):
    """Every definition reachable from id_ within `count` steps, breadth-first through all operands."""
    chain, frontier, seen = [], [id_], set()
    while count and frontier:
        nxt = []
        for current in frontier:
            if current in seen or current not in d:
                continue
            seen.add(current)
            op, operands = d[current]
            chain.append((current, op, operands))
            nxt.extend(re.findall(r"%\w+", operands))
        frontier, count = nxt, count - 1
    return chain


def test_scatter_reads_one_index_of_the_selected_width():
    """The index read loads only the width the kind selects, at that width's own scale: the other width's
    bytes can lie past the index buffer the draw bound (found by @tupacfan04 on #7696), and the u32 element
    must be scaled once -- the first version scaled it twice."""
    text = stages.scatter_module(LAY)
    d = defs(text)

    # The one index load: a PSB uint pointer whose address chain holds both width offsets and the select.
    index_loads = []
    for m in re.finditer(r"^(%\w+) = OpConvertUToPtr %_ptr_PSB_uint (%\w+)$", text, re.M):
        joined = " ; ".join(op + " " + operands for _, op, operands in hops(d, m.group(2)))
        if "OpBitwiseAnd" in joined and "%c4294967292" in joined and "OpSelect" in joined:
            index_loads.append((m.group(1), m.group(2)))
    assert len(index_loads) == 1, f"the scatter must load one index, width-selected; found {len(index_loads)}"

    # The select's two branches: the u32 element at its own scale (x*4, not x*16), the u16's containing
    # dword (x*2 masked to the dword). Both scale the same per-vertex element, and the select tests the
    # index kind, so a non-indexed draw never reads the index address at all.
    select_id = next(i for i, op, _ in hops(d, index_loads[0][1]) if op == "OpSelect")
    _, sel_operands = d[select_id]
    cond, u32, u16 = re.findall(r"%\w+", sel_operands)[-3:]
    assert d[cond][0] == "OpIEqual" and d[cond][1].endswith("%c1"), \
        "the width select must test the index kind"
    assert d[u32][0] == "OpIMul" and d[u32][1].endswith("%c4"), \
        "the u32 index offset lost its element scale"
    element = re.findall(r"%\w+", d[u32][1])[1]
    assert not (d[element][0] == "OpIMul" and d[element][1].endswith("%c4")), \
        "the u32 element is scaled twice on its way to the load"
    assert d[u16][0] == "OpBitwiseAnd" and "%c4294967292" in d[u16][1], \
        "the u16 offset lost its dword mask"
    byte = re.findall(r"%\w+", d[u16][1])[1]
    assert d[byte][0] == "OpIMul" and d[byte][1].endswith("%c2"), "the u16 offset lost its byte scale"
    assert re.findall(r"%\w+", d[byte][1])[1] == element, \
        "the two widths must scale the same per-vertex element"

    # The loaded word is picked whole for u32 and shifted out of its dword for u16: one more select on
    # the same condition, with the load's result on the u32 side.
    load_val = re.search(r"^(%\w+) = OpLoad %uint " + re.escape(index_loads[0][0]) + r"\b",
                         text, re.M).group(1)
    value_sel = [i for i, (op, ops) in d.items()
                 if op == "OpSelect" and cond in re.findall(r"%\w+", ops)
                 and load_val in re.findall(r"%\w+", ops)]
    assert len(value_sel) == 1, "the loaded index word lost its width select"
    v_u32, v_u16 = re.findall(r"%\w+", d[value_sel[0]][1])[-2:]
    assert v_u32 == load_val, "the u32 read must take the loaded word whole"
    v16_chain = " ; ".join(op + " " + o for _, op, o in hops(d, v_u16, 3))
    assert "OpShiftRightLogical" in v16_chain and "%c65535" in v16_chain, \
        "the u16 read lost its dword shift and mask"


def test_draw_vs_rejects_what_it_cannot_decode():
    """The corpus's strips only: another output topology, or a non-position-first output order, is refused
    rather than mis-decoded."""
    for bad in ({"topology": "triangle_list"}, {"outputs": ["location 1", "position"]}):
        lay = dict(LAY)
        lay.update(bad)
        try:
            stages.draw_vs_module(lay)
        except stages.Unsupported:
            continue
        raise AssertionError(f"draw_vs_module accepted {bad}")
