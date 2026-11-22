"""The Home hero: a painted dusk over the water, generated (needs Pillow; the launcher uses flat navy without it).

Kept deliberately quiet: no sun stripes, palms or neon. Colour lives here so the rest of the UI can stay flat.
"""
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter

NAVY = (12, 13, 27)
SKY = [(0.00, (12, 13, 27)), (0.30, (34, 30, 70)), (0.55, (92, 56, 104)), (0.78, (204, 124, 128)),
       (1.00, (246, 196, 158))]
SEA = [(0.00, (70, 52, 92)), (0.25, (36, 30, 64)), (1.00, (12, 13, 27))]


def _grad(stops, t):
    t = min(1.0, max(0.0, t))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if t <= p1:
            k = (t - p0) / max(1e-6, p1 - p0)
            return tuple(int(a + (b - a) * k) for a, b in zip(c0, c1))
    return stops[-1][1]


def render_hero(w, h, seed=11):
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    hz = int(h * 0.70)
    for y in range(hz):
        d.line((0, y, w, y), fill=_grad(SKY, y / hz))
    for y in range(hz, h):
        d.line((0, y, w, y), fill=_grad(SEA, (y - hz) / max(1, h - hz)))

    # soft cloud banks in the low sky: blurred streaks, lit from below
    clouds = Image.new("L", (w, h))
    cd = ImageDraw.Draw(clouds)
    for _ in range(26):
        cx = rnd.uniform(-0.1, 1.1) * w
        cy = rnd.uniform(0.28, 0.66) * h
        cw = rnd.uniform(0.12, 0.38) * w
        ch = rnd.uniform(0.012, 0.035) * h
        cd.ellipse((cx - cw, cy - ch, cx + cw, cy + ch), fill=rnd.randint(40, 95))
    clouds = clouds.filter(ImageFilter.GaussianBlur(h * 0.02))
    tint = Image.new("RGB", (w, h), (255, 186, 196))
    img = Image.composite(tint, img, clouds.point(lambda v: int(v * 0.55)))

    # glow low on the horizon, right of centre
    glow = Image.new("L", (w, h))
    gx = int(w * 0.68)
    ImageDraw.Draw(glow).ellipse((gx - w * 0.35, hz - h * 0.30, gx + w * 0.35, hz + h * 0.10), fill=120)
    glow = glow.filter(ImageFilter.GaussianBlur(h * 0.18))
    img = ImageChops.screen(img, Image.composite(Image.new("RGB", (w, h), (255, 170, 150)),
                                                 Image.new("RGB", (w, h)), glow))

    # distant coastline: low, flat, low contrast
    city = Image.new("L", (w, h))
    cdraw = ImageDraw.Draw(city)
    x = int(w * 0.34)
    while x < int(w * 0.97):
        bw = rnd.randint(max(2, int(w * 0.004)), int(w * 0.014))
        r = rnd.random()
        bh = int(h * (rnd.uniform(0.005, 0.02) if r < 0.55 else rnd.uniform(0.02, 0.05) if r < 0.9
                      else rnd.uniform(0.05, 0.09)))
        cdraw.rectangle((x, hz - bh, x + bw, hz), fill=255)
        x += bw + rnd.randint(0, int(w * 0.004))
    city = city.filter(ImageFilter.GaussianBlur(0.7))
    img = Image.composite(Image.new("RGB", (w, h), (78, 52, 92)), img, city.point(lambda v: int(v * 0.85)))
    d = ImageDraw.Draw(img)
    d.rectangle((0, hz, w, hz), fill=(150, 104, 128))

    # reflection: soft horizontal light on the water under the glow
    refl = Image.new("L", (w, h))
    rd = ImageDraw.Draw(refl)
    for y in range(hz + 3, h, 3):
        t = (y - hz) / (h - hz)
        half = w * (0.16 - 0.10 * t) * rnd.uniform(0.5, 1.0)
        off = rnd.uniform(-w * 0.02, w * 0.02)
        rd.line((gx - half + off, y, gx + half + off, y), fill=int(150 * (1 - t) ** 1.5))
    refl = refl.filter(ImageFilter.GaussianBlur(1.2))
    img = ImageChops.screen(img, Image.composite(Image.new("RGB", (w, h), (255, 196, 170)),
                                                 Image.new("RGB", (w, h)), refl))

    # scrims: left (for the title) and bottom (into the page)
    sc = Image.new("L", (w, h))
    sd = ImageDraw.Draw(sc)
    for x in range(int(w * 0.62)):
        sd.line((x, 0, x, h), fill=int(225 * (1 - x / (w * 0.62)) ** 1.4))
    bottom = Image.new("L", (w, h))
    bd = ImageDraw.Draw(bottom)
    for y in range(int(h * 0.55), h):
        bd.line((0, y, w, y), fill=int(255 * max(0.0, (y - h * 0.55) / (h * 0.45)) ** 1.6))
    sc = ImageChops.lighter(sc, bottom)
    img = Image.composite(Image.new("RGB", (w, h), NAVY), img, sc)

    # fine grain so the gradients don't band
    noise = Image.effect_noise((w, h), 7).convert("L")
    img = Image.blend(img, ImageChops.add(img, Image.merge("RGB", [noise] * 3), 1, -128), 0.35)
    return img
