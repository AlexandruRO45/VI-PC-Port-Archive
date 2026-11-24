"""Instructions MIMG dans l'interpréteur RDNA 2 : lecture/écriture et échantillonnage 2D.

Descripteurs et cas pris en charge : emu/rdna2_image.py. Couvre image_load, image_store,
image_get_resinfo, image_sample[_lz], image_sample_c[_lz] et image_gather4[_lz] avec filtrage
au plus proche ou bilinéaire et adressage wrap, mirror, clamp-to-edge, mirror-once ou bordure.
Le HLSL généré (emu/rdna2_hlsl_image.py) fait exactement le même calcul, au bit près.
"""
import math

from emu import rdna2_image as images
from emu.rdna2_exec_ops import MASK32, bits_float, float_bits

FAMILIES = frozenset({"MIMG"})
INT_MIN, INT_MAX_FLOAT = -2147483648.0, 2147483520.0   # bornes float32 d'un entier signé 32 bits
BORDER_COLORS = {
    images.BORDER_TRANS_BLACK: [0, 0, 0, 0],
    images.BORDER_OPAQUE_BLACK: [0, 0, 0, images.ONE_BITS],
    images.BORDER_OPAQUE_WHITE: [images.ONE_BITS, images.ONE_BITS, images.ONE_BITS, images.ONE_BITS],
}


def supports(family, name):
    return family in FAMILIES and name in images.SUPPORTED_OPS


def _f32(value):
    return bits_float(float_bits(value))


def clamp_index(index, size, mode):
    """Indice de texel après adressage, ou -1 si hors image en mode bordure."""
    if mode == images.CLAMP_WRAP:
        return index % size
    if mode == images.CLAMP_MIRROR:
        period = 2 * size
        pos = index % period
        return pos if pos < size else period - 1 - pos
    if mode == images.CLAMP_LAST_TEXEL:
        return min(max(index, 0), size - 1)
    if mode == images.CLAMP_MIRROR_ONCE_LAST_TEXEL:
        if index < 0:
            index = -1 - index
        return min(index, size - 1)
    if mode == images.CLAMP_MIRROR_ONCE_BORDER:
        if index < 0:
            index = -1 - index
        return index if index < size else -1
    return index if 0 <= index < size else -1


def texel_index(bits, size, unnormalized, mode):
    """Indice de texel (filtrage au plus proche) sur un axe ; -1 si hors image en mode bordure."""
    value = bits_float(bits)
    if not unnormalized:
        value = _f32(value * size)
    if math.isnan(value):
        return clamp_index(0, size, mode)
    index = int(math.floor(min(max(value, INT_MIN), INT_MAX_FLOAT)))
    return clamp_index(index, size, mode)


def bilinear_axis(bits, size, unnormalized):
    """(i0, i1, frac) sur un axe pour le filtrage bilinéaire ou gather4 ; NaN → (0, 0, 0.0)."""
    value = bits_float(bits)
    if not unnormalized:
        value = _f32(value * size)
    if math.isnan(value):
        return 0, 0, 0.0
    shifted = _f32(value - 0.5)
    clamped = min(max(shifted, INT_MIN), INT_MAX_FLOAT)
    fl = math.floor(clamped)
    frac = _f32(clamped - fl)
    i0 = int(fl)
    return i0, i0 + 1, frac


def texel(memory, image, x, y):
    """Les 4 canaux (bits float32) du texel (x, y)."""
    address = image["address"] + (y * image["width"] + x) * images.TEXEL_BYTES[image["format"]]
    if image["format"] == images.FORMAT_32_32_32_32_FLOAT:
        return list(memory.read_words(address, 4))
    word = memory.read_words(address, 1)[0]
    return [images.UNORM8[(word >> (8 * channel)) & 0xFF] for channel in range(4)]


def fetch_texel(memory, image, x, y, border_color=images.BORDER_TRANS_BLACK):
    """Texel (x, y) ou couleur de bordure si un indice vaut -1."""
    if x < 0 or y < 0:
        return list(BORDER_COLORS[border_color])
    return texel(memory, image, x, y)


def swizzled(channels, swizzle):
    selected = {images.SELECT_ZERO: 0, images.SELECT_ONE: images.ONE_BITS}
    return [channels[sel - images.SELECT_X] if sel >= images.SELECT_X else selected.get(sel, 0) for sel in swizzle]


def _select(channels, sel):
    if sel >= images.SELECT_X:
        return channels[sel - images.SELECT_X]
    return images.ONE_BITS if sel == images.SELECT_ONE else 0


def _lerp_bits(a_bits, b_bits, frac):
    a, b = bits_float(a_bits), bits_float(b_bits)
    diff = _f32(b - a)
    prod = _f32(diff * frac)
    return float_bits(_f32(a + prod))


def _bilerp_bits(c00, c10, c01, c11, fx, fy):
    return _lerp_bits(_lerp_bits(c00, c10, fx), _lerp_bits(c01, c11, fx), fy)


def _compare_depth(z_bits, texel_bits, func):
    z, t = bits_float(z_bits), bits_float(texel_bits)
    if func == images.COMPARE_NEVER:
        passed = False
    elif func == images.COMPARE_LESS:
        passed = z < t
    elif func == images.COMPARE_EQUAL:
        passed = z == t
    elif func == images.COMPARE_LEQUAL:
        passed = z <= t
    elif func == images.COMPARE_GREATER:
        passed = z > t
    elif func == images.COMPARE_NOTEQUAL:
        passed = z != t
    elif func == images.COMPARE_GEQUAL:
        passed = z >= t
    else:
        passed = True
    return images.ONE_BITS if passed else 0


def float_to_unorm8(bits):
    """Convertit un float32 (bits) en entier UNORM 8 bits [0..255], identique au HLSL."""
    value = bits_float(bits)
    if math.isnan(value) or value <= 0.0:
        return 0
    if value >= 1.0:
        return 255
    scaled = _f32(value * 255.0)
    return int(math.floor(_f32(scaled + 0.5)))


def _store_texel(memory, image, x, y, dmask, values):
    if x >= image["width"] or y >= image["height"]:
        return
    address = image["address"] + (y * image["width"] + x) * images.TEXEL_BYTES[image["format"]]
    if image["format"] == images.FORMAT_32_32_32_32_FLOAT:
        idx = 0
        for channel in range(4):
            if dmask >> channel & 1:
                memory.write_words(address + channel * images.DWORD, [values[idx]])
                idx += 1
        return
    word = 0 if dmask == 0xF else memory.read_words(address, 1)[0]
    idx = 0
    for channel in range(4):
        if dmask >> channel & 1:
            byte = float_to_unorm8(values[idx])
            shift = 8 * channel
            word = (word & ~(0xFF << shift)) | (byte << shift)
            idx += 1
    memory.write_words(address, [word & MASK32])


def _sample_color(memory, image, sampler, s_bits, t_bits, unnormalized):
    border = sampler["border_color"]
    if sampler["mag_filter"] == images.FILTER_LINEAR:
        x0, x1, fx = bilinear_axis(s_bits, image["width"], unnormalized)
        y0, y1, fy = bilinear_axis(t_bits, image["height"], unnormalized)
        cx0 = clamp_index(x0, image["width"], sampler["clamp_x"])
        cx1 = clamp_index(x1, image["width"], sampler["clamp_x"])
        cy0 = clamp_index(y0, image["height"], sampler["clamp_y"])
        cy1 = clamp_index(y1, image["height"], sampler["clamp_y"])
        c00 = fetch_texel(memory, image, cx0, cy0, border)
        c10 = fetch_texel(memory, image, cx1, cy0, border)
        c01 = fetch_texel(memory, image, cx0, cy1, border)
        c11 = fetch_texel(memory, image, cx1, cy1, border)
        raw = [_bilerp_bits(c00[k], c10[k], c01[k], c11[k], fx, fy) for k in range(4)]
    else:
        x = texel_index(s_bits, image["width"], unnormalized, sampler["clamp_x"])
        y = texel_index(t_bits, image["height"], unnormalized, sampler["clamp_y"])
        raw = fetch_texel(memory, image, x, y, border)
    return swizzled(raw, image["swizzle"])


def _sample_compare(memory, image, sampler, z_bits, s_bits, t_bits, unnormalized):
    border = sampler["border_color"]
    func = sampler["depth_compare"]
    if sampler["mag_filter"] == images.FILTER_LINEAR:
        x0, x1, fx = bilinear_axis(s_bits, image["width"], unnormalized)
        y0, y1, fy = bilinear_axis(t_bits, image["height"], unnormalized)
        cx0 = clamp_index(x0, image["width"], sampler["clamp_x"])
        cx1 = clamp_index(x1, image["width"], sampler["clamp_x"])
        cy0 = clamp_index(y0, image["height"], sampler["clamp_y"])
        cy1 = clamp_index(y1, image["height"], sampler["clamp_y"])
        r00 = _compare_depth(z_bits, fetch_texel(memory, image, cx0, cy0, border)[0], func)
        r10 = _compare_depth(z_bits, fetch_texel(memory, image, cx1, cy0, border)[0], func)
        r01 = _compare_depth(z_bits, fetch_texel(memory, image, cx0, cy1, border)[0], func)
        r11 = _compare_depth(z_bits, fetch_texel(memory, image, cx1, cy1, border)[0], func)
        pcf = _bilerp_bits(r00, r10, r01, r11, fx, fy)
    else:
        x = texel_index(s_bits, image["width"], unnormalized, sampler["clamp_x"])
        y = texel_index(t_bits, image["height"], unnormalized, sampler["clamp_y"])
        pcf = _compare_depth(z_bits, fetch_texel(memory, image, x, y, border)[0], func)
    return swizzled([pcf, pcf, pcf, images.ONE_BITS], image["swizzle"])


def _gather4(memory, image, sampler, s_bits, t_bits, unnormalized, dmask):
    border = sampler["border_color"]
    x0, x1, _ = bilinear_axis(s_bits, image["width"], unnormalized)
    y0, y1, _ = bilinear_axis(t_bits, image["height"], unnormalized)
    cx0 = clamp_index(x0, image["width"], sampler["clamp_x"])
    cx1 = clamp_index(x1, image["width"], sampler["clamp_x"])
    cy0 = clamp_index(y0, image["height"], sampler["clamp_y"])
    cy1 = clamp_index(y1, image["height"], sampler["clamp_y"])
    sel = image["swizzle"][images.gather_channel(dmask)]
    # Ordre matériel RDNA 2 / D3D12 / Vulkan : (x0, y1), (x1, y1), (x1, y0), (x0, y0)
    coords = ((cx0, cy1), (cx1, cy1), (cx1, cy0), (cx0, cy0))
    return [_select(fetch_texel(memory, image, cx, cy, border), sel) for cx, cy in coords]


def _write_masked(wave, vdata, lane, dmask, result):
    written = 0
    for component in range(4):
        if dmask >> component & 1:
            wave.v[vdata + written][lane] = result[component]
            written += 1


def execute(wave, name, word):
    """Exécute une instruction MIMG ; `word` : l'instruction entière."""
    from emu.rdna2_exec import Untranslated

    fields = images.instruction_fields(word)
    try:
        images.check_instruction(name, fields)
        image = images.image([wave.s[fields["srsrc"] + i] for i in range(8)])
        sampler = images.sampler([wave.s[fields["ssamp"] + i] for i in range(4)]) if name in images.SAMPLER_OPS else None
    except images.ImageError as error:
        raise Untranslated(str(error)) from None
    addrs = images.address_registers(name, fields)
    dmask, vdata = fields["dmask"], fields["vdata"]
    if name == "image_load":
        for lane in wave.active():
            x, y = wave.v[addrs[0]][lane], wave.v[addrs[1]][lane]
            if x >= image["width"] or y >= image["height"]:
                result = [0, 0, 0, 0]
            else:
                result = swizzled(texel(wave.memory, image, x, y), image["swizzle"])
            _write_masked(wave, vdata, lane, dmask, result)
        return
    if name == "image_store":
        count = bin(dmask).count("1")
        for lane in wave.active():
            x, y = wave.v[addrs[0]][lane], wave.v[addrs[1]][lane]
            values = [wave.v[vdata + i][lane] for i in range(count)]
            _store_texel(wave.memory, image, x, y, dmask, values)
        return
    if name == "image_get_resinfo":
        for lane in wave.active():
            mip = wave.v[addrs[0]][lane]
            result = [image["width"], image["height"], 1, 1] if mip == 0 else [0, 0, 0, 1]
            _write_masked(wave, vdata, lane, dmask, result)
        return
    unnormalized = fields["unorm"] or sampler["unnormalized"]
    if name in images.GATHER_OPS:
        for lane in wave.active():
            s_bits, t_bits = wave.v[addrs[0]][lane], wave.v[addrs[1]][lane]
            result = _gather4(wave.memory, image, sampler, s_bits, t_bits, unnormalized, dmask)
            for idx in range(4):
                wave.v[vdata + idx][lane] = result[idx]
        return
    if name in images.COMPARE_OPS:
        for lane in wave.active():
            z_bits, s_bits, t_bits = wave.v[addrs[0]][lane], wave.v[addrs[1]][lane], wave.v[addrs[2]][lane]
            result = _sample_compare(wave.memory, image, sampler, z_bits, s_bits, t_bits, unnormalized)
            _write_masked(wave, vdata, lane, dmask, result)
        return
    for lane in wave.active():
        s_bits, t_bits = wave.v[addrs[0]][lane], wave.v[addrs[1]][lane]
        result = _sample_color(wave.memory, image, sampler, s_bits, t_bits, unnormalized)
        _write_masked(wave, vdata, lane, dmask, result)
