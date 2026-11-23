# Leonida Launcher - design notes

## Principles
- One dusk palette: navy base, a pastel-pink primary with dark ink, cream section labels, indigo secondary
  buttons. Colour lives in the one Home picture; the chrome stays flat (no glows, gradients, skew or italics).
- Fixed spacing scale, few type sizes, one accent with a job, hairlines instead of shadows, everything on one left
  margin, one primary action per screen, plain status text instead of pop-ups.
- No game or publisher names, logos, key art or fonts in the UI.

## Decisions
| Question | Choice |
|---|---|
| Direction | Navy, pastel-pink primary, cream condensed caps headings, flat surfaces |
| Hero | Generated painted dusk sky with faint coastline - no sun stripes, palms or neon |
| Fonts | Big Shoulders Display (OFL, bundled in fonts/, static SemiBold + ExtraBold cut from the variable font) for display; Segoe UI for text |
| Extras | Removed: mission banners, wanted stars, phone toasts |

## Kit (kit.py)
Tokens: NAVY, RAIL, SURFACE, SURFACE_HI, LINE, INK, INK_SOFT, INK_FAINT, CREAM, PINK (+HI/LO, PINK_INK),
INDIGO, OK/WARN/ERR (muted). Spacing 4/8/16/24/32/48. Type: hero 88, title 44, stat 30 (display); heading 15,
body 14, button 14/16, section 12 caps, small 12 (Segoe UI). Controls: Button (primary/secondary/ghost/danger,
m/l), Dot, Progress, Check, Stepper, entry, Scroll. `python kit.py` shows the style test.

## Chunks
| Chunk | Work | Status |
|---|---|---|
| 1 Kit | tokens, fonts, controls, style test | done |
| 2 Shell + Home | side rail with wordmark, hero art, home stats and actions | done |
| 3 Folders + Tools | rows, inputs, download/locate states | done |
| 4 Build + Updates | stage list, log, progress; update panel | done |
| 5 Polish | 1280x720 / 1920x1080 / 150% scale screenshots, leftovers | 1280x720 + 1920x1080 checked; 150% DPI not seen |
