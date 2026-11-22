"""Leonida Launcher UI kit: tokens, type, and the few controls every page uses.

Direction (from the live rockstargames.com/VI styles): deep navy base, flat surfaces, cream uppercase condensed
headings, a soft pastel-pink primary button with dark ink, white medium-weight text. Colour belongs to the one hero
image; the chrome stays quiet. Rules: spacing only from SP; two families (Big Shoulders Display for headings,
Segoe UI for everything else); one accent per screen (pink); no gradients, glows, skew or italics in the chrome.

Run this file directly for the style-test screen.
"""
import ctypes
import os
import tkinter as tk
import tkinter.font as tkfont

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- tokens
NAVY = "#0C0D1B"        # page
RAIL = "#090A14"        # side rail
SURFACE = "#14152A"     # cards, inputs
SURFACE_HI = "#1C1D36"  # hover
LINE = "#26274A"        # hairlines
INK = "#FFFFFF"
INK_SOFT = "#A7A6C5"
INK_FAINT = "#6E6D8F"
CREAM = "#FFF9CB"       # section labels / headings
PINK = "#FFB0C4"        # the accent: primary button, active marker, progress
PINK_HI = "#FFC4D3"
PINK_LO = "#F297B0"
PINK_INK = "#2D2B4F"    # text on pink
INDIGO = "#38365D"      # secondary button
INDIGO_HI = "#45436F"
OK = "#86D9AE"
WARN = "#F3CD84"
ERR = "#FF9494"

SP = {"xs": 4, "s": 8, "m": 16, "l": 24, "xl": 32, "xxl": 48}

DISPLAY = "Big Shoulders Display ExtraBold"
DISPLAY_SEMI = "Big Shoulders Display SemiBold"
UI = "Segoe UI"
UI_SEMI = "Segoe UI Semibold"
MONO = "Cascadia Mono"

_scale = 1.0


def load_fonts():
    """Make the bundled fonts visible to this process only (no install)."""
    if os.name != "nt":
        return
    d = os.path.join(HERE, "fonts")
    for f in os.listdir(d) if os.path.isdir(d) else []:
        if f.lower().endswith(".ttf"):
            ctypes.windll.gdi32.AddFontResourceExW(os.path.join(d, f), 0x10, 0)  # FR_PRIVATE


def init(root):
    """Call once after the Tk root exists: picks the scale and builds the named fonts."""
    global _scale, DISPLAY, DISPLAY_SEMI
    _scale = max(1.0, root.winfo_fpixels("1i") / 96.0)
    fams = set(tkfont.families(root))
    if DISPLAY not in fams:  # bundled font missing: fall back to what Windows has
        DISPLAY = DISPLAY_SEMI = "Bahnschrift SemiBold Condensed"
    T.update({
        "hero": tkfont.Font(root, family=DISPLAY, size=-px(88)),
        "title": tkfont.Font(root, family=DISPLAY, size=-px(44)),
        "stat": tkfont.Font(root, family=DISPLAY_SEMI, size=-px(30)),
        "section": tkfont.Font(root, family=UI_SEMI, size=-px(12)),
        "nav": tkfont.Font(root, family=UI_SEMI, size=-px(14)),
        "button": tkfont.Font(root, family=UI_SEMI, size=-px(14)),
        "button_l": tkfont.Font(root, family=UI_SEMI, size=-px(16)),
        "heading": tkfont.Font(root, family=UI_SEMI, size=-px(15)),
        "body": tkfont.Font(root, family=UI, size=-px(14)),
        "small": tkfont.Font(root, family=UI, size=-px(12)),
        "mono": tkfont.Font(root, family=MONO, size=-px(12)),
    })


T = {}


def px(v):
    return int(round(v * _scale))


def spaced(text):
    """Letter-spaced caps for section labels (Tk has no tracking)."""
    return "\u2009".join(text.upper())


def mix(a, b, t):
    a = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * t) for x, y in zip(a, b))


# ---------------------------------------------------------------- text helpers
def label(parent, text, style="body", color=INK, bg=None, **kw):
    return tk.Label(parent, text=text, font=T[style], fg=color, bg=bg or parent["bg"], anchor="w",
                    justify="left", **kw)


def section(parent, text, bg=None):
    return label(parent, spaced(text), "section", CREAM, bg)


def title(parent, text, bg=None):
    return label(parent, text.upper(), "title", INK, bg)


def card(parent, bg=SURFACE):
    return tk.Frame(parent, bg=bg, highlightthickness=1, highlightbackground=LINE, highlightcolor=LINE)


def rule(parent, bg=LINE):
    return tk.Frame(parent, bg=bg, height=1)


def _round_rect(c, x0, y0, x1, y1, r, **kw):
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1, x0, y1,
           x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(pts, smooth=True, **kw)


# ---------------------------------------------------------------- button
TONES = {
    # tone: (fill, hover, pressed, ink, border)
    "primary": (PINK, PINK_HI, PINK_LO, PINK_INK, None),
    "secondary": (INDIGO, INDIGO_HI, mix(INDIGO, NAVY, 0.3), INK, None),
    "ghost": (None, SURFACE_HI, SURFACE, INK, LINE),
    "danger": (None, mix(ERR, NAVY, 0.85), mix(ERR, NAVY, 0.75), ERR, mix(ERR, NAVY, 0.45)),
}


class Button(tk.Canvas):
    """Flat rounded button. Same API the launcher uses: .command, set_text(), set_enabled(), pack()."""

    def __init__(self, master, text, command=None, tone="secondary", size="m", width=None):
        self.f = T["button_l" if size == "l" else "button"]
        self.h = px(48 if size == "l" else 36)
        self.padx = px(28 if size == "l" else 16)
        self.min_w = px(width) if width else 0
        self.text, self.tone, self.command = text, tone, command
        self.enabled, self.hover, self.down = True, False, False
        super().__init__(master, height=self.h, bg=master["bg"], highlightthickness=0, cursor="hand2")
        self._size()
        self.bind("<Enter>", lambda e: self._state(hover=True))
        self.bind("<Leave>", lambda e: self._state(hover=False, down=False))
        self.bind("<ButtonPress-1>", lambda e: self._state(down=True))
        self.bind("<ButtonRelease-1>", self._release)
        self.draw()

    def _size(self):
        self.w = max(self.min_w, self.f.measure(self.text) + 2 * self.padx)
        self.config(width=self.w)

    def _state(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.draw()

    def _release(self, e):
        was = self.down
        self._state(down=False)
        if was and self.enabled and self.hover and self.command:
            self.command()

    def draw(self):
        self.delete("all")
        fill, hov, prs, ink, border = TONES[self.tone]
        if not self.enabled:
            fill, ink, border = (SURFACE if fill else None), INK_FAINT, (LINE if border or not fill else None)
        elif self.down:
            fill = prs
        elif self.hover:
            fill = hov
        r = px(6)
        _round_rect(self, 1, 1, self.w - 1, self.h - 1, r, fill=fill or self["bg"],
                    outline=border or fill or self["bg"], width=1)
        self.create_text(self.w / 2, self.h / 2, text=self.text, font=self.f, fill=ink)
        self.config(cursor="hand2" if self.enabled else "arrow")

    def set_text(self, text):
        if text != self.text:
            self.text = text
            self._size()
            self.draw()

    def set_enabled(self, on):
        if on != self.enabled:
            self.enabled = on
            self.draw()

    def set_tone(self, tone):
        if tone != self.tone:
            self.tone = tone
            self.draw()


# ---------------------------------------------------------------- status dot, progress, checkbox, stepper
STATUS = {"ok": OK, "warn": WARN, "err": ERR, "busy": PINK, "idle": INK_FAINT}


class Dot(tk.Canvas):
    def __init__(self, master, state="idle", size=10):
        s = px(size)
        super().__init__(master, width=s, height=s, bg=master["bg"], highlightthickness=0)
        self.s = s
        self.set(state)

    def set(self, state):
        self.delete("all")
        self.create_oval(1, 1, self.s - 1, self.s - 1, fill=STATUS.get(state, INK_FAINT), outline="")


class Progress(tk.Canvas):
    """Thin track with a pink fill. value None = indeterminate (call step() on a timer)."""

    def __init__(self, master, width=220, height=4, color=PINK):
        super().__init__(master, width=px(width), height=px(height), bg=master["bg"], highlightthickness=0)
        self.wd, self.ht, self.value, self.phase, self.color = px(width), px(height), 0.0, 0, color
        self.bind("<Configure>", lambda e: self._resize(e.width))
        self.draw()

    def _resize(self, w):
        self.wd = w
        self.draw()

    def set(self, value):
        self.value = value
        self.draw()

    def step(self):
        self.phase = (self.phase + px(6)) % (self.wd + px(80))
        self.draw()

    def draw(self):
        self.delete("all")
        self.create_rectangle(0, 0, self.wd, self.ht, fill=LINE, outline="")
        if self.value is None:
            x = self.phase - px(80)
            self.create_rectangle(max(0, x), 0, min(self.wd, x + px(80)), self.ht, fill=self.color, outline="")
        elif self.value > 0:
            self.create_rectangle(0, 0, int(self.wd * min(1, self.value)), self.ht, fill=self.color, outline="")


class Check(tk.Frame):
    """Checkbox + label bound to a BooleanVar."""

    def __init__(self, master, text, var, command=None):
        super().__init__(master, bg=master["bg"], cursor="hand2")
        self.var, self.command = var, command
        s = px(18)
        self.box = tk.Canvas(self, width=s, height=s, bg=master["bg"], highlightthickness=0)
        self.box.pack(side="left")
        self.lbl = label(self, text, "body", INK)
        self.lbl.pack(side="left", padx=(px(SP["s"] + 2), 0))
        for w in (self, self.box, self.lbl):
            w.bind("<Button-1>", self.toggle)
        var.trace_add("write", lambda *_: self.draw())
        self.draw()

    def toggle(self, e=None):
        self.var.set(not self.var.get())
        if self.command:
            self.command()

    def draw(self):
        c, s = self.box, px(18)
        c.delete("all")
        on = bool(self.var.get())
        _round_rect(c, 1, 1, s - 1, s - 1, px(4), fill=PINK if on else SURFACE, outline=PINK if on else INK_FAINT)
        if on:
            c.create_line(s * 0.26, s * 0.52, s * 0.44, s * 0.7, s * 0.76, s * 0.32, fill=PINK_INK,
                          width=max(2, px(2)), capstyle="round", joinstyle="round")


class Stepper(tk.Frame):
    """[-] value [+] bound to an IntVar."""

    def __init__(self, master, var, lo, hi, step=1, command=None, unit=""):
        super().__init__(master, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        self.var, self.lo, self.hi, self.st, self.command, self.unit = var, lo, hi, step, command, unit
        for txt, d in (("\u2212", -1), (None, 0), ("+", 1)):
            if txt is None:
                self.val = label(self, "", "body", INK, SURFACE, width=6)
                self.val.config(anchor="center", justify="center")
                self.val.pack(side="left")
                continue
            b = tk.Label(self, text=txt, font=T["heading"], fg=INK_SOFT, bg=SURFACE, width=3, cursor="hand2")
            b.pack(side="left", ipady=px(4))
            b.bind("<Button-1>", lambda e, d=d: self.bump(d))
            b.bind("<Enter>", lambda e, b=b: b.config(bg=SURFACE_HI, fg=INK))
            b.bind("<Leave>", lambda e, b=b: b.config(bg=SURFACE, fg=INK_SOFT))
        var.trace_add("write", lambda *_: self.show())
        self.show()

    def bump(self, d):
        try:
            v = int(self.var.get())
        except (ValueError, tk.TclError):
            v = self.lo
        self.var.set(max(self.lo, min(self.hi, v + d * self.st)))
        if self.command:
            self.command()

    def show(self):
        try:
            self.val.config(text=f"{int(self.var.get())}{self.unit}")
        except (ValueError, tk.TclError):
            pass


def entry(parent, var):
    e = tk.Entry(parent, textvariable=var, bg=SURFACE, fg=INK, insertbackground=INK, relief="flat",
                 font=T["body"], highlightthickness=1, highlightbackground=LINE, highlightcolor=PINK,
                 disabledbackground=SURFACE, readonlybackground=SURFACE)
    return e


class Scroll(tk.Frame):
    """A vertically scrolling area (mouse wheel works anywhere inside it). Put content in .inner."""

    def __init__(self, master, bg=NAVY):
        super().__init__(master, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.canvas.pack(fill="both", expand=True)
        self.inner.bind("<Configure>", lambda e: self.canvas.config(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        self.bind_all("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, e):
        w = self.winfo_containing(e.x_root, e.y_root)
        while w is not None and w is not self:
            w = w.master
        if w is self and self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-e.delta / 120), "units")


# ---------------------------------------------------------------- style test
def style_test():
    load_fonts()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    root.configure(bg=NAVY)
    init(root)
    root.title("Leonida kit")
    pad = px(SP["xl"])
    f = tk.Frame(root, bg=NAVY)
    f.pack(fill="both", expand=True, padx=pad, pady=pad)
    label(f, "LEONIDA", "hero", INK).pack(anchor="w")
    title(f, "Tools").pack(anchor="w", pady=(px(SP["l"]), 0))
    section(f, "Section label").pack(anchor="w", pady=(px(SP["l"]), px(SP["s"])))
    label(f, "Body text: the build continues where it left off.", "body").pack(anchor="w")
    label(f, "Secondary text, 12 px, soft ink", "small", INK_SOFT).pack(anchor="w")
    row = tk.Frame(f, bg=NAVY)
    row.pack(anchor="w", pady=px(SP["l"]))
    Button(row, "Build", tone="primary", size="l").pack(side="left")
    Button(row, "Play", tone="secondary", size="l").pack(side="left", padx=px(SP["s"]))
    Button(row, "Locate", tone="ghost").pack(side="left", padx=px(SP["s"]))
    Button(row, "Stop", tone="danger").pack(side="left", padx=px(SP["s"]))
    d = Button(row, "Installed", tone="secondary")
    d.set_enabled(False)
    d.pack(side="left", padx=px(SP["s"]))
    c = card(f)
    c.pack(fill="x", pady=px(SP["s"]))
    inner = tk.Frame(c, bg=SURFACE)
    inner.pack(fill="x", padx=px(SP["m"]), pady=px(SP["m"]))
    for st, txt in (("ok", "Vulkan SDK  ·  C:\\VulkanSDK\\1.4.321.1"), ("err", "llvm-mc  ·  Not found"),
                    ("warn", "ffmpeg  ·  Optional"), ("busy", "Downloading  ·  143 of 268 MB")):
        r = tk.Frame(inner, bg=SURFACE)
        r.pack(fill="x", pady=px(SP["xs"]))
        Dot(r, st).pack(side="left", padx=(0, px(SP["s"] + 2)))
        label(r, txt, "body", INK, SURFACE).pack(side="left")
    p = Progress(f, 420)
    p.pack(anchor="w", pady=px(SP["m"]))
    p.set(0.53)
    v = tk.BooleanVar(value=True)
    Check(f, "Also make WAV audio", v).pack(anchor="w", pady=px(SP["s"]))
    iv = tk.IntVar(value=6)
    Stepper(f, iv, 1, 12).pack(anchor="w", pady=px(SP["s"]))
    sv = tk.StringVar(value="D:\\Games\\GTA VI Package")
    entry(f, sv).pack(fill="x", ipady=px(8), pady=px(SP["s"]))
    root.geometry(f"{px(900)}x{px(860)}")
    return root


if __name__ == "__main__":
    style_test().mainloop()
