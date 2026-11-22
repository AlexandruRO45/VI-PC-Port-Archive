"""Leonida Launcher - a GUI over make_game.py: folders, tools, build, play and GitHub updates.

Look and controls come from kit.py (see DESIGN.md); the hero image from art.py (optional, needs Pillow).
Start it with "Leonida Launcher.bat" in the repo folder.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kit  # noqa: E402
import toolbox  # noqa: E402
from kit import (CREAM, INDIGO, INK, INK_FAINT, INK_SOFT, LINE, NAVY, OK, ERR, PINK, RAIL, SURFACE,  # noqa: E402
                 WARN, Button, Check, Dot, Progress, Stepper, px)

try:  # optional: the painted hero needs Pillow
    from PIL import ImageTk
    import art
except Exception:  # noqa: BLE001 - any import problem just means a flat hero
    art = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETTINGS = os.path.join(HERE, "settings.json")
STAGES = ["check", "shaders", "assets", "runtime", "pipelines"]
STAGE_INFO = {
    "check": "Check game files",
    "shaders": "Translate shaders",
    "assets": "Export models, textures and audio",
    "runtime": "Build the game loader",
    "pipelines": "Pre-compile shaders for your GPU",
}
NEED_GB = 350
REPO_URL = "https://github.com/ItsNotRexed/Leonida"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# ---------------------------------------------------------------- helpers

def load_settings():
    d = {"game": "", "out": os.path.join(os.path.expanduser("~"), "Documents", "Leonida"), "work": "",
         "stages": {s: True for s in STAGES}, "wav": False,
         "jobs": max(1, min(6, (os.cpu_count() or 2) // 2)),
         "auto_check": True, "auto_update": True, "interval_min": 30}
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            d.update(json.load(f))
    except (OSError, ValueError):
        pass
    return d


def save_settings(s):
    try:
        with open(SETTINGS, "w", encoding="utf-8", newline="\n") as f:
            json.dump(s, f, indent=2)
    except OSError:
        pass


def fresh_env():
    """This process's environment plus the user/system variables as they are now in the registry, so tools
    installed after the launcher started (Vulkan SDK, LLVM_MC) are seen without a restart."""
    env = dict(os.environ)
    if os.name != "nt":
        return env
    import winreg
    paths = []
    for hive, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, "Environment")):
        try:
            with winreg.OpenKey(hive, key) as k:
                i = 0
                while True:
                    try:
                        name, val, _ = winreg.EnumValue(k, i)
                    except OSError:
                        break
                    i += 1
                    val = os.path.expandvars(str(val))
                    if name.lower() == "path":
                        paths.append(val)
                    else:
                        env[name] = val
        except OSError:
            pass
    if paths:
        merged = []
        for p in ";".join(paths + [os.environ.get("PATH", "")]).split(";"):
            if p and p not in merged:
                merged.append(p)
        env["PATH"] = ";".join(merged)
    return env


def which(name, env):
    return shutil.which(name, path=env.get("PATH"))


def python_exe():
    exe = sys.executable
    if os.path.basename(exe).lower() == "pythonw.exe":
        alt = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.isfile(alt):
            return alt
    return exe


def run(args, env=None, timeout=60, cwd=ROOT):
    try:
        r = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, errors="replace",
                           timeout=timeout, creationflags=NO_WINDOW)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def free_gb(path):
    p = os.path.abspath(path or ROOT)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    try:
        return shutil.disk_usage(p).free / 1024 ** 3
    except OSError:
        return None


def looks_like_game(d):
    if not d or not os.path.isdir(d):
        return False
    try:
        names = os.listdir(d)
    except OSError:
        return False
    return "MicrosoftGame.Config" in names and any(n.lower().endswith(".rpf") for n in names)


S = kit.SP  # spacing scale


# ---------------------------------------------------------------- app

class App(tk.Tk):
    def __init__(self):
        kit.load_fonts()
        super().__init__()
        kit.init(self)
        self.title("Leonida Launcher")
        try:
            self.iconbitmap(default=os.path.join(HERE, "icon.ico"))
            self._icon = tk.PhotoImage(file=os.path.join(HERE, "icon.png"))
            self.iconphoto(True, self._icon)
        except tk.TclError:
            pass
        self.configure(bg=NAVY)
        self.geometry(f"{px(1200)}x{px(780)}")
        self.minsize(px(1040), px(680))
        self.s = load_settings()
        self.q = queue.Queue()
        self.proc = None
        self.job_name = None
        self.user_stopped = False
        self.req_results = {}
        self.update_info = {}
        self.env = toolbox.apply_overrides(fresh_env(), self.s.get("tools", {}))
        self.spinners = set()      # progress bars currently animating
        self.ticking = False

        self.rail = tk.Frame(self, bg=RAIL, width=px(232))
        self.rail.pack(side="left", fill="y")
        self.rail.pack_propagate(False)
        tk.Frame(self, bg=LINE, width=1).pack(side="left", fill="y")
        self.main = tk.Frame(self, bg=NAVY)
        self.main.pack(side="left", fill="both", expand=True)
        self.build_rail()

        self.pages = {}
        self.build_home()
        self.build_folders()
        self.build_reqs()
        self.build_build()
        self.build_updates()
        self.show("home")
        self.ensure_git_exclude()
        self.after(100, self.pump)
        self.after(300, self.check_reqs)
        if self.s.get("auto_check", True):
            self.after(1500, lambda: self.check_updates(auto=True))
        self.after(60_000, self.update_timer)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------ shell
    def build_rail(self):
        r = self.rail
        top = tk.Frame(r, bg=RAIL)
        top.pack(fill="x", padx=px(S["l"]), pady=(px(S["xl"]), px(S["xl"])))
        kit.label(top, "LEONIDA", "title", INK).pack(anchor="w")
        kit.section(top, "Launcher").pack(anchor="w", pady=(px(2), 0))
        self.nav = {}
        for key, text in (("home", "Home"), ("folders", "Folders"), ("reqs", "Tools"), ("build", "Build"),
                          ("updates", "Updates")):
            row = tk.Frame(r, bg=RAIL, height=px(44), cursor="hand2")
            row.pack(fill="x")
            row.pack_propagate(False)
            mark = tk.Frame(row, bg=RAIL, width=px(3))
            mark.pack(side="left", fill="y", pady=px(10))
            lbl = kit.label(row, text, "nav", INK_SOFT, RAIL)
            lbl.pack(side="left", padx=(px(S["l"] - 3), 0))
            for w in (row, lbl, mark):
                w.bind("<Button-1>", lambda e, k=key: self.show(k))
                w.bind("<Enter>", lambda e, k=key: self.nav_hover(k, True))
                w.bind("<Leave>", lambda e, k=key: self.nav_hover(k, False))
            self.nav[key] = (row, mark, lbl)
        tk.Frame(r, bg=RAIL).pack(fill="both", expand=True)
        foot = tk.Frame(r, bg=RAIL)
        foot.pack(fill="x", padx=px(S["l"]), pady=px(S["l"]))
        line = tk.Frame(foot, bg=RAIL)
        line.pack(fill="x")
        self.rail_dot = Dot(line, "idle", 8)
        self.rail_dot.pack(side="left", padx=(0, px(S["s"])))
        self.rail_status = kit.label(line, "Starting", "small", INK_SOFT, RAIL)
        self.rail_status.pack(side="left")
        self.rail_version = kit.label(foot, "", "small", INK_FAINT, RAIL)
        self.rail_version.pack(anchor="w", pady=(px(S["xs"]), 0))
        self.current = None

    def nav_hover(self, key, on):
        if key != self.current:
            self.nav[key][2].config(fg=INK if on else INK_SOFT)

    def show(self, key):
        self.current = key
        for k, f in self.pages.items():
            f.pack_forget()
            row, mark, lbl = self.nav[k]
            mark.config(bg=PINK if k == key else RAIL)
            lbl.config(fg=INK if k == key else INK_SOFT)
        self.pages[key].pack(fill="both", expand=True)
        if key == "home":
            self.refresh_home()

    def page(self, key, title=None):
        f = tk.Frame(self.main, bg=NAVY)
        self.pages[key] = f
        if title:
            kit.title(f, title).pack(anchor="w", padx=px(S["xxl"]), pady=(px(S["xl"]), px(S["l"])))
        return f

    def set_status(self, text, state="idle"):
        self.rail_status.config(text=text)
        self.rail_dot.set(state)

    def refresh_status(self):
        if self.proc:
            self.set_status({"build": "Building", "check": "Checking files"}.get(self.job_name, "Working"), "busy")
            return
        need = [v[3] for v in self.req_results.values() if v[2] and not v[0]]
        u = self.update_info
        if need:
            self.set_status(f"{len(need)} tool{'s' if len(need) != 1 else ''} missing", "err")
        elif u.get("behind"):
            self.set_status(f"{u['behind']} update{'s' if u['behind'] != 1 else ''} available", "busy")
        elif not looks_like_game(self.s.get("game", "")):
            self.set_status("Game folder not set", "warn")
        else:
            self.set_status("Ready", "ok")

    # spinner ticker: runs only while something is animating
    def spin(self, bar, on):
        (self.spinners.add if on else self.spinners.discard)(bar)
        if on:
            bar.set(None)
        if self.spinners and not self.ticking:
            self.ticking = True
            self.after(40, self._tick)

    def _tick(self):
        for b in list(self.spinners):
            if b.winfo_exists():
                b.step()
            else:
                self.spinners.discard(b)
        if self.spinners:
            self.after(40, self._tick)
        else:
            self.ticking = False

    # ------------------------------------------------ HOME
    def build_home(self):
        f = self.page("home")
        self.hero = tk.Canvas(f, bg=NAVY, highlightthickness=0, height=px(400))
        self.hero.pack(fill="both", expand=True)
        self.hero_img, self.hero_size, self.hero_src = None, (0, 0), None
        self.hero.bind("<Configure>", self.draw_hero)
        h = self.hero
        self.hero_eyebrow = h.create_text(0, 0, text=kit.spaced("For Grand Theft Auto VI"), anchor="w",
                                          font=kit.T["section"], fill=CREAM)
        self.hero_title = h.create_text(0, 0, text="LEONIDA", anchor="w", font=kit.T["hero"], fill=INK)
        self.home_build = Button(h, "Build the game", lambda: (self.show("build"), self.start_build()),
                                 tone="primary", size="l")
        self.home_play = Button(h, "Play", self.play, tone="secondary", size="l", width=120)
        self.hero_btn1 = h.create_window(0, 0, window=self.home_build, anchor="w")
        self.hero_btn2 = h.create_window(0, 0, window=self.home_play, anchor="w")
        self.home_note = h.create_text(0, 0, text="", anchor="w", font=kit.T["small"], fill=INK_SOFT)

        stats = tk.Frame(f, bg=NAVY)
        stats.pack(fill="x", padx=px(S["xxl"]), pady=(px(S["m"]), px(S["xxl"])))
        self.stats = {}
        for i, (key, name, page) in enumerate((("game", "Game files", "folders"), ("tools", "Tools", "reqs"),
                                               ("space", "Free space", "folders"), ("update", "Version", "updates"))):
            if i:
                tk.Frame(stats, bg=LINE, width=1).pack(side="left", fill="y", padx=px(S["l"]))
            col = tk.Frame(stats, bg=NAVY, cursor="hand2")
            col.pack(side="left", fill="x", expand=True)
            sec = kit.section(col, name)
            sec.pack(anchor="w")
            val = kit.label(col, "—", "stat", INK)
            val.pack(anchor="w", pady=(px(S["xs"]), 0))
            det = kit.label(col, "", "small", INK_SOFT)
            det.pack(anchor="w")
            for w in (col, sec, val, det):
                w.bind("<Button-1>", lambda e, p=page: self.show(p))
            self.stats[key] = (val, det)

    def draw_hero(self, e=None):
        h = self.hero
        w, ht = h.winfo_width(), h.winfo_height()
        if w < 50 or ht < 50 or (w, ht) == self.hero_size:
            return
        self.hero_size = (w, ht)
        # text block sits on the lower-left of the sky, its buttons on the horizon line
        x0 = px(S["xxl"])
        base = int(ht * 0.66)
        h.coords(self.hero_eyebrow, x0, base - px(150))
        h.coords(self.hero_title, x0 - px(4), base - px(86))
        h.coords(self.hero_btn1, x0, base)
        h.coords(self.hero_btn2, x0 + self.home_build.w + px(S["s"] + 4), base)
        h.coords(self.home_note, x0, base + px(52))
        h.delete("bg")
        if art is not None:
            try:
                img = art.render_hero(w, ht)
                self.hero_src = img
                self.hero_img = ImageTk.PhotoImage(img)
                h.create_image(0, 0, image=self.hero_img, anchor="nw", tags="bg")
                h.tag_lower("bg")
            except Exception:  # noqa: BLE001 - flat navy hero instead
                self.hero_src = None
        self.match_button_bg()

    def match_button_bg(self):
        """Buttons sit on the picture: give their corners the colour of the pixels behind them."""
        for btn, item in ((self.home_build, self.hero_btn1), (self.home_play, self.hero_btn2)):
            col = NAVY
            if self.hero_src is not None:
                x, y = self.hero.coords(item)
                try:
                    col = "#%02x%02x%02x" % self.hero_src.getpixel((int(x) + 2, int(y)))[:3]
                except (IndexError, ValueError):
                    pass
            btn.config(bg=col)
            btn.draw()

    def exe_path(self):
        return os.path.join(self.s.get("out") or "", "gta6rt.exe")

    def refresh_home(self):
        g = self.s.get("game", "")
        val, det = self.stats["game"]
        if looks_like_game(g):
            val.config(text="Found", fg=INK)
            det.config(text=self.short(g))
        else:
            val.config(text="Not set", fg=ERR)
            det.config(text="Choose it in Folders")
        val, det = self.stats["tools"]
        if self.req_results:
            ready = sum(1 for v in self.req_results.values() if v[0])
            need = [v[3] for v in self.req_results.values() if v[2] and not v[0]]
            val.config(text=f"{ready} / {len(toolbox.TOOLS)}", fg=INK if not need else ERR)
            det.config(text="All required tools ready" if not need else f"Missing: {', '.join(need)}"[:46])
        val, det = self.stats["space"]
        work = self.s.get("work") or os.path.join(self.s.get("out") or ROOT, "work")
        gb = free_gb(work)
        if gb is not None:
            val.config(text=f"{gb:,.0f} GB", fg=INK if gb >= NEED_GB else WARN)
            drive = os.path.splitdrive(os.path.abspath(work))[0] or work
            det.config(text=f"On {drive}  ·  a build needs {NEED_GB} GB")
        val, det = self.stats["update"]
        u = self.update_info
        if not u:
            val.config(text="—", fg=INK)
            det.config(text="Not checked yet")
        elif u.get("nogit"):
            val.config(text="ZIP copy", fg=WARN)
            det.config(text="Connect it in Updates")
        elif u.get("error"):
            val.config(text="Offline", fg=WARN)
            det.config(text="Couldn't reach GitHub")
        elif u.get("behind"):
            val.config(text=u.get("head", "")[:7], fg=INK)
            det.config(text=f"{u['behind']} update{'s' if u['behind'] != 1 else ''} available")
        else:
            val.config(text=u.get("head", "")[:7], fg=INK)
            det.config(text=f"Up to date  ·  checked {u.get('when', '')}")
        has_exe = os.path.isfile(self.exe_path())
        self.home_play.set_enabled(has_exe and self.proc is None)
        self.home_build.set_enabled(self.proc is None)
        self.home_build.set_text("Building…" if self.proc else "Build the game")
        self.hero.itemconfigure(self.home_note, text="" if has_exe else
                                "Play unlocks once the build makes gta6rt.exe. Leonida can't boot the game yet.")
        self.refresh_status()

    @staticmethod
    def short(p, n=40):
        return p if len(p) <= n else "…" + p[-(n - 1):]

    # ------------------------------------------------ FOLDERS
    def folder_row(self, parent, key, name, browse_title, find=False):
        box = tk.Frame(parent, bg=NAVY)
        box.pack(fill="x", pady=(0, px(S["l"])))
        kit.section(box, name).pack(anchor="w", pady=(0, px(S["s"])))
        row = tk.Frame(box, bg=NAVY)
        row.pack(fill="x")
        var = tk.StringVar(value=self.s.get(key, ""))
        wrap = tk.Frame(row, bg=SURFACE, highlightthickness=1, highlightbackground=LINE)
        wrap.pack(side="left", fill="x", expand=True)
        e = tk.Entry(wrap, textvariable=var, bg=SURFACE, fg=INK, insertbackground=INK, relief="flat",
                     font=kit.T["body"], highlightthickness=0)
        e.pack(fill="x", padx=px(12), ipady=px(7))
        e.bind("<FocusIn>", lambda ev: wrap.config(highlightbackground=PINK))
        e.bind("<FocusOut>", lambda ev: wrap.config(highlightbackground=LINE))

        def browse():
            d = filedialog.askdirectory(title=browse_title, initialdir=var.get() or "C:\\", parent=self)
            if d:
                var.set(os.path.normpath(d))

        if find:
            Button(row, "Find", self.auto_find, tone="secondary").pack(side="left", padx=(px(S["s"]), 0))
        Button(row, "Browse", browse, tone="ghost").pack(side="left", padx=(px(S["s"]), 0))
        st = tk.Frame(box, bg=NAVY)
        st.pack(fill="x", pady=(px(S["s"]), 0))
        dot = Dot(st, "idle", 8)
        dot.pack(side="left", padx=(0, px(S["s"])))
        msg = kit.label(st, "", "small", INK_SOFT)
        msg.pack(side="left")

        def changed(*_):
            self.s[key] = var.get().strip().strip('"')
            save_settings(self.s)
            self.validate_folders()

        var.trace_add("write", changed)
        return var, (dot, msg)

    def build_folders(self):
        f = self.page("folders", "Folders")
        body = tk.Frame(f, bg=NAVY)
        body.pack(fill="both", expand=True, padx=px(S["xxl"]))
        self.game_var, self.game_state = self.folder_row(body, "game", "Game package",
                                                         "Choose your game package folder", find=True)
        self.out_var, self.out_state = self.folder_row(body, "out", "Build output", "Choose the output folder")
        self.work_var, self.work_state = self.folder_row(body, "work", "Work files", "Choose the work folder")
        kit.section(body, "Drives").pack(anchor="w", pady=(px(S["s"]), px(S["s"])))
        self.drives = tk.Frame(body, bg=NAVY)
        self.drives.pack(fill="x")
        self.validate_folders()

    def _state(self, pair, state, text):
        pair[0].set(state)
        pair[1].config(text=text, fg={"ok": INK_SOFT, "err": ERR, "warn": WARN}.get(state, INK_SOFT))

    def validate_folders(self):
        if not hasattr(self, "drives"):
            return
        g = self.s.get("game", "")
        if not g:
            self._state(self.game_state, "err", "Not set. Pick the folder with MicrosoftGame.Config and the .rpf files.")
        elif looks_like_game(g):
            self._state(self.game_state, "ok", "Game package found")
        else:
            self._state(self.game_state, "err", "No MicrosoftGame.Config and .rpf files in this folder")
        o = self.s.get("out", "")
        gb = free_gb(o) if o else None
        self._state(self.out_state, "ok" if o else "err", f"{gb:,.0f} GB free" if gb is not None else "Not set")
        w = self.s.get("work") or os.path.join(o or ROOT, "work")
        gb = free_gb(w)
        if gb is None:
            self._state(self.work_state, "warn", "?")
        else:
            note = "" if self.s.get("work") else "Empty uses <output>\\work.  "
            self._state(self.work_state, "ok" if gb >= NEED_GB else "warn",
                        f"{note}{gb:,.0f} GB free; a full build needs about {NEED_GB} GB")
        for c in self.drives.winfo_children():
            c.destroy()
        if os.name == "nt":
            for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                p = f"{letter}:\\"
                if not os.path.exists(p):
                    continue
                try:
                    u = shutil.disk_usage(p)
                except OSError:
                    continue
                free, total = u.free / 1024 ** 3, u.total / 1024 ** 3
                r = tk.Frame(self.drives, bg=NAVY)
                r.pack(fill="x", pady=px(S["xs"]))
                kit.label(r, f"{letter}:", "heading", INK, width=3).pack(side="left")
                bar = Progress(r, 240, 4, color="#6F6C9E")
                bar.pack(side="left", padx=(0, px(S["m"])))
                bar.set(free / total if total else 0)
                kit.label(r, f"{free:,.0f} GB free of {total:,.0f} GB", "small",
                          INK if free >= NEED_GB else INK_SOFT).pack(side="left")
                if free >= NEED_GB:
                    kit.label(r, "  ·  enough for a full build", "small", OK).pack(side="left")
        if self.current == "home":
            self.refresh_home()

    def auto_find(self):
        self._state(self.game_state, "busy", "Searching Downloads, Desktop, Documents, Games and each drive…")
        code = ("import sys, json; sys.path.insert(0, sys.argv[1]); import make_game; "
                "print('@@' + json.dumps([[d, v.get('version', '?')] for d, v in make_game.find_dumps()]))")

        def work():
            rc, out = run([self.build_python(), "-I", "-c", code, ROOT], env=self.env, timeout=300)
            found = []
            for line in out.splitlines():
                if line.startswith("@@"):
                    try:
                        found = json.loads(line[2:])
                    except ValueError:
                        pass
            self.q.put(("found", found, out if rc else ""))

        threading.Thread(target=work, daemon=True).start()

    def on_found(self, found, err):
        if not found:
            self.validate_folders()
            self._state(self.game_state, "err", "No copy found automatically. Use Browse.")
            if err:
                self.write_log(f"\n>>> find game\n{err[-600:]}\n", "err")
            return
        if len(found) == 1:
            self.game_var.set(found[0][0])
            return
        top = tk.Toplevel(self, bg=NAVY)
        top.title("Choose a copy")
        top.transient(self)
        kit.title(top, "Choose a copy").pack(anchor="w", padx=px(S["xl"]), pady=(px(S["l"]), px(S["m"])))
        for d, ver in found:
            r = tk.Frame(top, bg=NAVY)
            r.pack(fill="x", padx=px(S["xl"]), pady=px(S["xs"]))
            kit.label(r, f"{d}   ({ver})", "body").pack(side="left")
            Button(r, "Use", lambda d=d: (self.game_var.set(d), top.destroy()), tone="secondary").pack(
                side="right", padx=(px(S["m"]), 0))
        tk.Frame(top, bg=NAVY, height=px(S["l"])).pack()

    # ------------------------------------------------ TOOLS
    def tool_env(self):
        return toolbox.apply_overrides(fresh_env(), self.s.get("tools", {}))

    def build_python(self):
        p = self.s.get("tools", {}).get("python")
        return p if p and os.path.isfile(p) else python_exe()

    def build_reqs(self):
        f = self.page("reqs")
        head = tk.Frame(f, bg=NAVY)
        head.pack(fill="x", padx=px(S["xxl"]), pady=(px(S["xl"]), px(S["m"])))
        kit.title(head, "Tools").pack(side="left")
        Button(head, "Recheck", self.check_reqs, tone="ghost").pack(side="right")
        self.req_summary = kit.label(head, "", "heading", INK_SOFT)
        self.req_summary.pack(side="right", padx=px(S["m"]))
        sc = kit.Scroll(f)
        sc.pack(fill="both", expand=True, padx=px(S["xxl"]), pady=(0, px(S["l"])))
        body = sc.inner
        self.req_rows, self.tool_state, self.tool_cancel = {}, {}, {}
        kit.rule(body).pack(fill="x")
        for t in toolbox.TOOLS:
            key = t["key"]
            row = tk.Frame(body, bg=NAVY)
            row.pack(fill="x")
            inner = tk.Frame(row, bg=NAVY)
            inner.pack(fill="x", pady=px(S["m"]))
            dot = Dot(inner, "idle", 10)
            dot.pack(side="left", padx=(0, px(S["m"])), anchor="n", pady=px(6))
            mid = tk.Frame(inner, bg=NAVY)
            mid.pack(side="left", fill="x", expand=True)
            top = tk.Frame(mid, bg=NAVY)
            top.pack(fill="x")
            kit.label(top, t["label"], "heading", INK).pack(side="left")
            if not t["required"]:
                kit.label(top, "  Optional", "small", INK_FAINT).pack(side="left", pady=(px(2), 0))
            kit.label(mid, t["why"], "small", INK_SOFT).pack(anchor="w")
            detail = kit.label(mid, "Checking…", "small", INK_FAINT, wraplength=px(520))
            detail.pack(anchor="w", pady=(px(2), 0))
            right = tk.Frame(inner, bg=NAVY)
            right.pack(side="right")
            label = "Build" if t["get"][0] == "script" else "Download"
            dl = Button(right, label, lambda k=key: self.get_tool(k), tone="secondary", width=112)
            dl.pack(side="right", padx=(px(S["s"]), 0))
            loc = None
            if t["locate"]:
                loc = Button(right, "Locate", lambda k=key: self.locate_tool(k), tone="ghost")
                loc.pack(side="right")
            prog = Progress(right, 160, 4)
            kit.rule(body).pack(fill="x")
            self.req_rows[key] = {"dot": dot, "detail": detail, "dl": dl, "loc": loc, "prog": prog,
                                  "label": label}

    def set_tool(self, key, state, text=""):
        """state: checking, ok, missing, downloading, installing, failed."""
        self.tool_state[key] = (state, text)
        r, t = self.req_rows[key], toolbox.BY_KEY[key]
        busy = state in ("downloading", "installing")
        dot = {"ok": "ok", "checking": "idle", "downloading": "busy", "installing": "busy", "failed": "err",
               "missing": "err" if t["required"] else "warn"}[state]
        r["dot"].set(dot)
        mine = key in self.s.get("tools", {})
        if state == "ok":
            r["detail"].config(text=("Using your copy: " if mine else "") + text, fg=INK_SOFT)
        elif state == "failed":
            r["detail"].config(text="Failed: " + text, fg=ERR)
        elif state == "missing":
            r["detail"].config(text=text, fg=ERR if t["required"] else WARN)
        else:
            r["detail"].config(text=text, fg=INK if busy else INK_FAINT)
        if busy:
            r["prog"].pack(side="right", padx=(0, px(S["m"])))
            self.spin(r["prog"], state == "installing")
            if state == "downloading" and r["prog"].value is None:
                r["prog"].set(0)
        else:
            self.spin(r["prog"], False)
            r["prog"].pack_forget()
        if state == "downloading":
            r["dl"].set_text("Cancel")
            r["dl"].set_tone("danger")
            r["dl"].command = lambda k=key: self.tool_cancel.get(k) and self.tool_cancel[k].set()
            r["dl"].set_enabled(True)
        else:
            r["dl"].set_tone("secondary")
            r["dl"].set_text(r["label"] if state != "ok" else ("Built" if r["label"] == "Build" else "Installed"))
            r["dl"].command = lambda k=key: self.get_tool(k)
            r["dl"].set_enabled(state not in ("installing", "checking", "ok"))
        if r["loc"]:
            r["loc"].set_enabled(not busy)

    def check_reqs(self, only=None):
        keys = [only] if only else [t["key"] for t in toolbox.TOOLS]
        for k in keys:
            if self.tool_state.get(k, ("",))[0] not in ("downloading", "installing"):
                self.set_tool(k, "checking", "Checking…")

        def work():
            env = self.tool_env()
            res = {}
            for k in keys:
                if k == "python" and self.build_python() != python_exe():
                    rc, out = toolbox.run([self.build_python(), "-c",
                                           "import sys; print('%d %d %d' % sys.version_info[:3])"])
                    try:
                        ver = tuple(int(x) for x in out.split()[:3])
                    except ValueError:
                        ver = (0, 0, 0)
                else:
                    ver = sys.version_info[:3]
                try:
                    res[k] = toolbox.check(k, env, ver)
                except Exception as e:  # noqa: BLE001 - a broken check shouldn't kill the page
                    res[k] = (False, f"Check failed: {e}")
            self.q.put(("reqs", res, env))

        threading.Thread(target=work, daemon=True).start()

    def on_reqs(self, res, env):
        self.env = env
        for key, (ok, msg) in res.items():
            t = toolbox.BY_KEY[key]
            if self.tool_state.get(key, ("",))[0] in ("downloading", "installing"):
                continue
            prev = getattr(self, "after_install", {}).pop(key, None)
            if ok:
                self.set_tool(key, "ok", msg)
            elif prev:
                self.set_tool(key, "failed", prev + f" ({msg})")
            else:
                self.set_tool(key, "missing", msg)
            self.req_results[key] = (ok, msg, t["required"], t["label"])
        ready = sum(1 for v in self.req_results.values() if v[0])
        need = [v[3] for v in self.req_results.values() if v[2] and not v[0]]
        self.req_summary.config(text=f"{ready} of {len(toolbox.TOOLS)} ready", fg=INK_SOFT if need else OK)
        self.refresh_home()

    def locate_tool(self, key):
        t = toolbox.BY_KEY[key]
        kind, what = t["locate"]
        if kind == "dir":
            p = filedialog.askdirectory(title=f"Choose {what}", parent=self)
        else:
            p = filedialog.askopenfilename(title=f"Find {what}", parent=self,
                                           filetypes=[(what, what), ("Programs", "*.exe")])
        if not p:
            return
        p = os.path.normpath(p)
        ok, why = toolbox.validate_located(key, p)
        if not ok:
            messagebox.showwarning("Leonida", f"That isn't usable for {t['label']}:\n\n{why}", parent=self)
            return
        self.s.setdefault("tools", {})[key] = p
        save_settings(self.s)
        self.check_reqs(only=key)

    def get_tool(self, key):
        t = toolbox.BY_KEY[key]
        kind = t["get"][0]
        if kind == "script":
            missing = [toolbox.BY_KEY[k]["label"] for k in ("git", "vs")
                       if not self.req_results.get(k, (False,))[0]]
            if missing:
                self.set_tool(key, "failed", "Needs " + " and ".join(missing) + " first")
                return
            if self.proc:
                self.set_tool(key, "failed", f"Wait for the {self.job_name} to finish")
                return
            if not messagebox.askyesno("Leonida", "Build llvm-mc with the AMDGPU target now?\n\nThis runs the repo's "
                                       "scripts\\build_llvm_mc.ps1: it fetches LLVM's source with git and compiles "
                                       "only llvm-mc (about 2 GB temporary space, 20-40 minutes), then sets LLVM_MC.",
                                       parent=self):
                return
            self.set_tool(key, "installing", "Building llvm-mc. The log is on the Build page.")
            self.start_process("llvm-mc build", ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                                 os.path.join(ROOT, "scripts", "build_llvm_mc.ps1")])
            if not self.proc:
                self.set_tool(key, "failed", "PowerShell couldn't start")
            return
        cancel = threading.Event()
        self.tool_cancel[key] = cancel
        self.set_tool(key, "downloading", "Connecting…")

        def work():
            try:
                path = toolbox.download(t["get"][1], lambda d, tot: self.q.put(("tprog", key, d, tot)), cancel)
                if kind == "zip":
                    self.q.put(("tstate", key, "installing", "Unpacking…"))
                    exe = toolbox.unzip_tool(path, t["get"][2])
                    self.q.put(("tlocated", key, exe))
                else:
                    self.q.put(("tstate", key, "installing",
                                "The installer is open. Follow its steps and allow the Windows prompt."))
                    args = t["get"][2] if len(t["get"]) > 2 else ()
                    code = toolbox.run_installer(path, args)
                    if code not in (0, 3010, 1641):
                        raise RuntimeError(f"The installer closed with code {code} (cancelled or failed)")
                    self.q.put(("tinstalled", key, code))
            except Exception as e:  # noqa: BLE001 - every failure is shown on the row
                self.q.put(("tfail", key, toolbox.explain(e)))

        threading.Thread(target=work, daemon=True).start()

    def on_tool_msg(self, m):
        kind, key = m[0], m[1]
        if kind == "tprog":
            if self.tool_state.get(key, ("",))[0] != "downloading":
                self.set_tool(key, "downloading", "Downloading…")
            done, total = m[2], m[3]
            r = self.req_rows[key]
            if total:
                r["prog"].set(done / total)
                r["detail"].config(text=f"Downloading  ·  {done / total * 100:.0f}%  ·  "
                                        f"{done / 1048576:,.0f} of {total / 1048576:,.0f} MB", fg=INK)
            else:
                r["detail"].config(text=f"Downloading  ·  {done / 1048576:,.0f} MB", fg=INK)
        elif kind == "tstate":
            self.set_tool(key, m[2], m[3])
        elif kind == "tfail":
            self.set_tool(key, "failed", m[2])
        elif kind == "tlocated":
            self.s.setdefault("tools", {})[key] = m[2]
            save_settings(self.s)
            self.tool_state[key] = ("checking", "")
            self.check_reqs(only=key)
        elif kind == "tinstalled":
            self.tool_state[key] = ("checking", "")
            if not hasattr(self, "after_install"):
                self.after_install = {}
            self.after_install[key] = ("The installer finished, but the tool still isn't found. If it asked to "
                                       "restart, do that; otherwise use Locate")
            self.check_reqs(only=key)

    # ------------------------------------------------ BUILD
    def build_build(self):
        f = self.page("build", "Build")
        body = tk.Frame(f, bg=NAVY)
        body.pack(fill="both", expand=True, padx=px(S["xxl"]), pady=(0, px(S["l"])))
        cols = tk.Frame(body, bg=NAVY)
        cols.pack(fill="x")
        left = tk.Frame(cols, bg=NAVY)
        left.pack(side="left", fill="y", anchor="n")
        kit.section(left, "Stages").pack(anchor="w", pady=(0, px(S["s"])))
        self.stage_vars = {}
        for s in STAGES:
            v = tk.BooleanVar(value=self.s["stages"].get(s, True))
            Check(left, STAGE_INFO[s], v, command=self.save_build_opts).pack(anchor="w", pady=px(S["xs"]))
            self.stage_vars[s] = v
        tk.Frame(cols, bg=LINE, width=1).pack(side="left", fill="y", padx=px(S["xl"]))
        right = tk.Frame(cols, bg=NAVY)
        right.pack(side="left", fill="both", expand=True, anchor="n")
        kit.section(right, "Options").pack(anchor="w", pady=(0, px(S["s"])))
        self.wav_var = tk.BooleanVar(value=self.s.get("wav", False))
        Check(right, "Also make WAV audio (much larger, needs ffmpeg)", self.wav_var,
              command=self.save_build_opts).pack(anchor="w", pady=px(S["xs"]))
        jr = tk.Frame(right, bg=NAVY)
        jr.pack(anchor="w", pady=(px(S["m"]), 0))
        kit.label(jr, "Parallel jobs", "body").pack(side="left", padx=(0, px(S["m"])))
        self.jobs_var = tk.IntVar(value=self.s.get("jobs", 4))
        Stepper(jr, self.jobs_var, 1, max(2, os.cpu_count() or 2), command=self.save_build_opts).pack(side="left")

        acts = tk.Frame(body, bg=NAVY)
        acts.pack(fill="x", pady=(px(S["xl"]), px(S["m"])))
        self.build_btn = Button(acts, "Build", self.start_build, tone="primary", size="l", width=140)
        self.build_btn.pack(side="left")
        self.checkonly_btn = Button(acts, "Check files only", lambda: self.start_build(check_only=True),
                                    tone="secondary", size="l")
        self.checkonly_btn.pack(side="left", padx=(px(S["s"] + 4), 0))
        self.stop_btn = Button(acts, "Stop", self.stop, tone="danger", size="l")
        self.stop_btn.pack(side="left", padx=(px(S["s"] + 4), 0))
        self.stop_btn.set_enabled(False)
        Button(acts, "Open output", lambda: self.open_folder(self.s.get("out")), tone="ghost", size="l").pack(
            side="right")

        st = tk.Frame(body, bg=NAVY)
        st.pack(fill="x", pady=(0, px(S["s"])))
        self.stage_dot = Dot(st, "idle", 8)
        self.stage_dot.pack(side="left", padx=(0, px(S["s"])))
        self.stage_lbl = kit.label(st, "Idle", "body", INK_SOFT)
        self.stage_lbl.pack(side="left")
        self.progress = Progress(body, 400, 3)
        self.progress.pack(fill="x", pady=(0, px(S["m"])))
        logf = tk.Frame(body, bg=RAIL, highlightthickness=1, highlightbackground=LINE)
        logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, bg=RAIL, fg=INK_SOFT, insertbackground=INK, relief="flat", font=kit.T["mono"],
                           wrap="word", padx=px(S["m"]), pady=px(S["m"]), highlightthickness=0,
                           selectbackground=INDIGO)
        self.log.pack(fill="both", expand=True)
        self.log.tag_config("err", foreground=ERR)
        self.log.tag_config("ok", foreground=OK)
        self.log.tag_config("me", foreground=CREAM)

    def save_build_opts(self):
        self.s["stages"] = {s: v.get() for s, v in self.stage_vars.items()}
        self.s["wav"] = self.wav_var.get()
        try:
            self.s["jobs"] = max(1, int(self.jobs_var.get()))
        except (ValueError, tk.TclError):
            pass
        save_settings(self.s)

    def write_log(self, text, tag=None):
        if tag is None:
            low = text.lower()
            tag = "err" if any(w in low for w in ("error", "failed", "traceback", "missing")) else None
        self.log.insert("end", text, tag)
        if int(self.log.index("end-1c").split(".")[0]) > 5000:
            self.log.delete("1.0", "1000.0")
        self.log.see("end")

    def start_build(self, check_only=False):
        if self.proc:
            return
        self.save_build_opts()
        g = self.s.get("game", "")
        if not looks_like_game(g):
            messagebox.showwarning("Leonida", "Set your game package folder first (Folders page).", parent=self)
            self.show("folders")
            return
        args = [self.build_python(), "-u", "-I", os.path.join(ROOT, "make_game.py"), "--game", g]
        if self.s.get("out"):
            args += ["--out", self.s["out"]]
        if self.s.get("work"):
            args += ["--work", self.s["work"]]
        if check_only:
            args.append("--check")
        else:
            chosen = [s for s in STAGES if self.s["stages"].get(s)]
            if not chosen:
                messagebox.showwarning("Leonida", "Tick at least one stage.", parent=self)
                return
            if chosen != STAGES:
                args += ["--stages", ",".join(chosen)]
            if self.s.get("wav"):
                args.append("--wav")
            args += ["--jobs", str(self.s.get("jobs", 4))]
        self.start_process("check" if check_only else "build", args)

    def start_process(self, name, args):
        if self.proc:
            messagebox.showinfo("Leonida", f"Wait for the {self.job_name} to finish, or stop it.", parent=self)
            return
        env = self.tool_env()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        self.write_log(f"\n>>> {name}: {' '.join(args)}\n", "me")
        try:
            self.proc = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        except OSError as e:
            self.write_log(f"could not start: {e}\n", "err")
            return
        self.job_name = name
        self.job_start = time.time()
        self.build_btn.set_enabled(False)
        self.checkonly_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)
        self.stage_dot.set("busy")
        self.stage_lbl.config(text="Starting…", fg=INK)
        self.spin(self.progress, True)
        proc = self.proc

        def reader():
            for raw in iter(proc.stdout.readline, b""):
                self.q.put(("log", raw.decode("utf-8", "replace").replace("\r\n", "\n")))
            proc.wait()
            self.q.put(("done", proc.returncode))

        threading.Thread(target=reader, daemon=True).start()
        self.refresh_home()

    def stop(self):
        if self.proc and messagebox.askyesno("Leonida", f"Stop the {self.job_name}? A build continues where it "
                                             "left off next time.", parent=self):
            self.user_stopped = True
            run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], timeout=30)

    def on_done(self, rc):
        mins = (time.time() - self.job_start) / 60
        ok = rc == 0
        self.write_log(f"<<< {self.job_name} {'finished' if ok else f'stopped (code {rc})'} after {mins:.1f} min\n",
                       "ok" if ok else "err")
        if ok:
            text, state = f"Finished in {mins:.1f} min", "ok"
        elif self.user_stopped:
            text, state = "Stopped. The next build continues from here.", "warn"
        else:
            text, state = f"Stopped with an error (code {rc}). See the log.", "err"
        self.stage_dot.set(state)
        self.stage_lbl.config(text=text, fg={"ok": OK, "warn": WARN, "err": ERR}[state])
        self.user_stopped = False
        name = self.job_name
        self.proc, self.job_name = None, None
        self.build_btn.set_enabled(True)
        self.checkonly_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        self.spin(self.progress, False)
        self.progress.set(1.0 if ok else 0)
        if name == "llvm-mc build":
            if rc != 0:
                self.after_install = getattr(self, "after_install", {})
                self.after_install["llvm"] = f"The llvm-mc build stopped (code {rc}); its log is on the Build page"
            self.tool_state["llvm"] = ("checking", "")
            self.check_reqs(only="llvm")
        self.refresh_home()
        self.bell()

    def track_stage(self, line):
        low = line.lower()
        for s in STAGES:
            if low.startswith(s + ":") or low.startswith(f"[{s}]") or f"stage {s}" in low:
                n = STAGES.index(s) + 1
                self.stage_lbl.config(text=f"{STAGE_INFO[s]}  ·  stage {n} of {len(STAGES)}", fg=INK)
                break

    # ------------------------------------------------ UPDATES
    def build_updates(self):
        f = self.page("updates", "Updates")
        body = tk.Frame(f, bg=NAVY)
        body.pack(fill="both", expand=True, padx=px(S["xxl"]), pady=(0, px(S["l"])))
        st = tk.Frame(body, bg=NAVY)
        st.pack(fill="x")
        self.upd_head = kit.label(st, "", "stat", INK)
        self.upd_head.pack(anchor="w")
        self.upd_sub = kit.label(st, "", "small", INK_SOFT, wraplength=px(720))
        self.upd_sub.pack(anchor="w", pady=(px(S["xs"]), 0))
        row = tk.Frame(body, bg=NAVY)
        row.pack(anchor="w", pady=(px(S["m"]), px(S["xl"])))
        self.upd_btn = Button(row, "Update now", self.do_update, tone="primary")
        self.upd_btn.pack(side="left")
        self.upd_btn.set_enabled(False)
        Button(row, "Check now", lambda: self.check_updates(auto=False), tone="ghost").pack(
            side="left", padx=(px(S["s"]), 0))

        kit.section(body, "Settings").pack(anchor="w", pady=(0, px(S["s"])))
        self.auto_check_var = tk.BooleanVar(value=self.s.get("auto_check", True))
        self.auto_upd_var = tk.BooleanVar(value=self.s.get("auto_update", True))
        Check(body, "Check for updates automatically", self.auto_check_var,
              command=self.save_upd_opts).pack(anchor="w", pady=px(S["xs"]))
        Check(body, "Install updates automatically when no build is running", self.auto_upd_var,
              command=self.save_upd_opts).pack(anchor="w", pady=px(S["xs"]))
        ir = tk.Frame(body, bg=NAVY)
        ir.pack(anchor="w", pady=(px(S["s"]), px(S["xl"])))
        kit.label(ir, "Check every", "body").pack(side="left", padx=(0, px(S["m"])))
        self.interval_var = tk.IntVar(value=self.s.get("interval_min", 30))
        Stepper(ir, self.interval_var, 5, 1440, 5, command=self.save_upd_opts, unit=" min").pack(side="left")

        kit.section(body, "New on GitHub").pack(anchor="w", pady=(0, px(S["s"])))
        lf = tk.Frame(body, bg=RAIL, highlightthickness=1, highlightbackground=LINE)
        lf.pack(fill="both", expand=True)
        self.commits = tk.Text(lf, bg=RAIL, fg=INK_SOFT, relief="flat", font=kit.T["mono"], wrap="none",
                               padx=px(S["m"]), pady=px(S["m"]), height=8, highlightthickness=0,
                               selectbackground=INDIGO)
        self.commits.pack(fill="both", expand=True)
        self.next_check = time.time() + 60 * self.s.get("interval_min", 30)

    def save_upd_opts(self):
        self.s["auto_check"] = self.auto_check_var.get()
        self.s["auto_update"] = self.auto_upd_var.get()
        try:
            self.s["interval_min"] = max(5, int(self.interval_var.get()))
        except (ValueError, tk.TclError):
            pass
        self.next_check = time.time() + 60 * self.s["interval_min"]
        save_settings(self.s)

    def git(self, *args, timeout=120):
        return run(["git", "-C", ROOT, *args], env=self.env, timeout=timeout)

    def check_updates(self, auto=False):
        self.upd_head.config(text="Checking…", fg=INK_SOFT)

        def work():
            info = {"when": time.strftime("%H:%M")}
            if not os.path.isdir(os.path.join(ROOT, ".git")):
                info["nogit"] = True
                self.q.put(("upd", info, auto))
                return
            rc, out = self.git("fetch", "--quiet", "origin")
            if rc != 0:
                info["error"] = "Couldn't reach GitHub: " + out.strip()[-200:]
            else:
                _, head = self.git("rev-parse", "HEAD")
                _, branch = self.git("rev-parse", "--abbrev-ref", "HEAD")
                _, up = self.git("rev-parse", "@{u}")
                _, cnt = self.git("rev-list", "--count", "HEAD..@{u}")
                _, ahead = self.git("rev-list", "--count", "@{u}..HEAD")
                _, log = self.git("log", "--format=%h  %ad  %s", "--date=format:%b %d %H:%M", "-n", "40",
                                  "HEAD..@{u}")
                _, cur = self.git("log", "-1", "--format=%h  %ad  %s", "--date=format:%b %d %H:%M")
                _, dirty = self.git("status", "--porcelain", "--untracked-files=no")
                info.update(head=head.strip(), branch=branch.strip(), upstream=up.strip(), current=cur.strip(),
                            behind=int(cnt.strip() or 0) if cnt.strip().isdigit() else 0,
                            ahead=int(ahead.strip()) if ahead.strip().isdigit() else 0,
                            log=log.strip(), dirty=dirty.strip())
            self.q.put(("upd", info, auto))

        threading.Thread(target=work, daemon=True).start()

    def on_upd(self, info, auto):
        self.update_info = info
        self.next_check = time.time() + 60 * self.s.get("interval_min", 30)
        self.commits.delete("1.0", "end")
        self.rail_version.config(text=f"Version {info.get('head', '')[:7]}" if info.get("head") else "")
        if info.get("nogit"):
            self.upd_head.config(text="Not connected to GitHub", fg=WARN)
            self.upd_sub.config(text="This copy came as a ZIP. Update now connects it to "
                                     f"{REPO_URL} so it can update itself (needs Git).")
            self.upd_btn.set_enabled(self.proc is None)
            if auto and self.s.get("auto_update") and self.proc is None and which("git", self.env):
                self.do_update(silent=True)
            self.refresh_home()
            return
        if info.get("error"):
            self.upd_head.config(text="Couldn't check", fg=WARN)
            self.upd_sub.config(text=info["error"])
            self.upd_btn.set_enabled(False)
        else:
            behind = info["behind"]
            self.upd_head.config(text=(f"{behind} update{'s' if behind != 1 else ''} available" if behind
                                       else "Up to date"), fg=INK)
            sub = f"Installed {info['current']}\nBranch {info['branch']}  ·  checked at {info['when']}"
            if info.get("dirty"):
                sub += "\nThis folder has local edits; updating keeps them if they don't clash."
            if info.get("ahead"):
                sub += f"\nThis folder has {info['ahead']} commit(s) GitHub doesn't."
            self.upd_sub.config(text=sub)
            self.commits.insert("end", info["log"] or "Nothing new.")
            self.upd_btn.set_enabled(bool(behind) and self.proc is None)
            if behind and auto and self.s.get("auto_update") and self.proc is None:
                self.do_update(silent=True)
        self.refresh_home()

    def do_update(self, silent=False):
        if self.proc:
            if not silent:
                messagebox.showinfo("Leonida", "Finish or stop the build before updating.", parent=self)
            return
        self.upd_head.config(text="Updating…", fg=INK_SOFT)
        self.upd_btn.set_enabled(False)

        def work():
            if not os.path.isdir(os.path.join(ROOT, ".git")):
                # A ZIP download: make it a clone of the repo in place, keeping the files that are here.
                out = ""
                for step in (("init", "-q", "-b", "main"), ("remote", "add", "origin", REPO_URL),
                             ("fetch", "-q", "origin", "main"), ("reset", "-q", "origin/main"),
                             ("branch", "-q", "--set-upstream-to=origin/main", "main")):
                    rc, o = self.git(*step, timeout=600)
                    out += o
                    if rc != 0:
                        self.q.put(("updated", rc, out, silent))
                        return
            rc, out = self.git("pull", "--ff-only", "--autostash", timeout=600)
            if rc == 0:
                rc2, out2 = self.git("submodule", "update", "--init", "--recursive", timeout=600)
                out += out2
                rc = rc2
            self.q.put(("updated", rc, out, silent))

        threading.Thread(target=work, daemon=True).start()

    def on_updated(self, rc, out, silent):
        self.write_log(f"\n>>> update from GitHub\n{out}\n", "me" if rc == 0 else "err")
        if rc != 0:
            self.upd_head.config(text="Update failed", fg=ERR)
            self.upd_sub.config(text=out.strip()[-500:])
            if not silent:
                messagebox.showerror("Leonida", "The update didn't apply:\n\n" + out.strip()[-600:], parent=self)
        else:
            self.set_status("Updated " + time.strftime("%H:%M"), "ok")
        self.check_updates(auto=False)

    def update_timer(self):
        if self.s.get("auto_check", True) and time.time() >= self.next_check:
            self.check_updates(auto=True)
        self.after(60_000, self.update_timer)

    # ------------------------------------------------ misc
    def play(self):
        exe = self.exe_path()
        if not os.path.isfile(exe):
            messagebox.showinfo("Leonida", "Build the game first (the runtime stage makes gta6rt.exe).",
                                parent=self)
            return
        try:
            subprocess.Popen([exe], cwd=os.path.dirname(exe))
        except OSError as e:
            messagebox.showerror("Leonida", f"Couldn't start the game: {e}", parent=self)

    def open_folder(self, d):
        if d and os.path.isdir(d):
            os.startfile(d)
        else:
            messagebox.showinfo("Leonida", "That folder doesn't exist yet.", parent=self)

    def ensure_git_exclude(self):
        """Keep the personal shortcut out of git (the launcher itself is tracked in the fork)."""
        p = os.path.join(ROOT, ".git", "info", "exclude")
        if not os.path.isdir(os.path.dirname(os.path.dirname(p))):
            return
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cur = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
            lines = [l for l in cur.splitlines() if l not in ("/_launcher/", "/Leonida Launcher.bat")]
            if "/Leonida Launcher.lnk" not in lines:
                lines.append("/Leonida Launcher.lnk")
            new = "\n".join(lines) + "\n"
            if new != cur:
                with open(p, "w", encoding="utf-8", newline="\n") as f:
                    f.write(new)
        except OSError:
            pass

    def pump(self):
        try:
            while True:
                m = self.q.get_nowait()
                kind = m[0]
                if kind == "log":
                    self.write_log(m[1])
                    self.track_stage(m[1])
                elif kind == "done":
                    self.on_done(m[1])
                elif kind == "reqs":
                    self.on_reqs(m[1], m[2])
                elif kind == "found":
                    self.on_found(m[1], m[2])
                elif kind == "upd":
                    self.on_upd(m[1], m[2])
                elif kind in ("tprog", "tstate", "tfail", "tlocated", "tinstalled"):
                    self.on_tool_msg(m)
                elif kind == "updated":
                    self.on_updated(m[1], m[2], m[3])
        except queue.Empty:
            pass
        self.after(100, self.pump)

    def on_close(self):
        if self.proc:
            if not messagebox.askyesno("Leonida", f"The {self.job_name} is still running. Stop it and quit?",
                                       parent=self):
                return
            run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], timeout=30)
        self.destroy()


if __name__ == "__main__":
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Leonida.Launcher")
        except (AttributeError, OSError):
            pass
    App().mainloop()
