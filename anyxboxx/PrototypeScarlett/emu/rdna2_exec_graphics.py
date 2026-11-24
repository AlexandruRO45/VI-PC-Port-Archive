"""Instructions graphiques de l'interpréteur RDNA 2 : interpolation (VINTRP) et exports (EXP).

Conventions (zones mémoire, entrées v0/v1) : emu/rdna2_graphics.py. Sémantique (AMD, « RDNA 2
Instruction Set Architecture ») : v_interp_p1_f32 D = P10 × S + P0 ; v_interp_p2_f32
D = P20 × S + D ; v_interp_mov_f32 D = P10, P20 ou P0. Produit et somme sont arrondis
séparément en float32, comme le HLSL généré (`precise`).
"""
from emu import rdna2_graphics as graphics
from emu.rdna2_exec_ops import bits_float, float_bits

FAMILIES = frozenset({"VINTRP", "EXP"})
SUPPORTED = frozenset({*graphics.INTERP_OPS.values(), "exp"})


def supports(family, name):
    return family in FAMILIES and name in SUPPORTED


def _multiply_add(a, b, c):
    return float_bits(bits_float(float_bits(bits_float(a) * bits_float(b))) + bits_float(c))


def _interpolate(wave, name, fields):
    from emu.rdna2_exec import Untranslated

    if fields["attr"] >= graphics.PARAMETER_COUNT:
        raise Untranslated(f"{name} : attribut {fields['attr']} au-delà de param{graphics.PARAMETER_COUNT - 1}")
    address = graphics.parameter_address(wave.primitive, fields["attr"], fields["chan"])
    p0, p10, p20 = wave.memory.read_words(address, 3)
    destination, source = wave.v[fields["vdst"]], fields["vsrc"]
    for lane in wave.active():
        if name == "v_interp_p1_f32":
            destination[lane] = _multiply_add(p10, wave.v[source][lane], p0)
        elif name == "v_interp_p2_f32":
            destination[lane] = _multiply_add(p20, wave.v[source][lane], destination[lane])
        else:
            destination[lane] = {0: p10, 1: p20, 2: p0}.get(source, 0)


def _export(wave, fields):
    from emu.rdna2_exec import Untranslated

    if fields["compr"]:
        raise Untranslated(f"exp {fields['name']} compr : export compressé (16 bits) non pris en charge")
    for lane in wave.active():
        values = [wave.v[register][lane] if fields["en"] >> channel & 1 else 0
                  for channel, register in enumerate(fields["sources"])]
        wave.exports.setdefault(fields["target"], {})[lane] = values


def execute(wave, family, name, word):
    """Exécute une instruction VINTRP (`word` 32 bits utiles) ou EXP (`word` 64 bits)."""
    if family == "VINTRP":
        _interpolate(wave, name, graphics.vintrp_fields(word))
    else:
        _export(wave, graphics.exp_fields(word))


def _single_lane_wave(sgprs, memory):
    from emu import rdna2_exec

    wave = rdna2_exec.Wave()
    wave.memory.write(0, memory)
    wave.s[rdna2_exec.EXEC_LO] = 1
    for number, value in (sgprs or {}).items():
        wave.s[number] = value
    wave.exports, wave.primitive = {}, 0
    return wave


def run_vertex(data, vertex, sgprs, memory):
    """Exports {cible: [4 mots]} du vertex shader pour un sommet (v0 = numéro du sommet)."""
    from emu import rdna2_exec

    wave = _single_lane_wave(sgprs, memory)
    wave.v[0][0] = vertex
    rdna2_exec.run(data, wave)
    return {target: lanes[0] for target, lanes in wave.exports.items()}


def run_pixel(data, i_bits, j_bits, primitive, sgprs, memory):
    """Exports {cible: [4 mots]} du pixel shader pour un pixel (v0 = i, v1 = j)."""
    from emu import rdna2_exec

    wave = _single_lane_wave(sgprs, memory)
    wave.v[0][0], wave.v[1][0], wave.primitive = i_bits, j_bits, primitive
    rdna2_exec.run(data, wave)
    return {target: lanes[0] for target, lanes in wave.exports.items()}
