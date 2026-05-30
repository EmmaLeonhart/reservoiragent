"""Generate docs/diagram-residual-reservoir.svg.

A reservoir-augmented version of the classic residual-stream picture (cf.
`data_lake/diagrams/transformer-diagram-misconceptions-claude-preview-03.webp`):
token residual streams with FFN-query (green), attention (blue) and FFN-value (pink)
neurons predicting "Paris" from "The capital of France is" — PLUS a reservoir column
whose fixed random nodes enter the attention layer as extra keys/values (read via
W_in, written via W_out) and persist across forward passes.

Themed to the report's light "paper" palette; renders identically in browser,
cairosvg, and the PDF builder (no CSS vars / classes — all literal). Re-runnable:

    python data_lake/build_residual_reservoir_svg.py
"""
import os

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "docs",
                                    "diagram-residual-reservoir.svg"))

# palette
NODE_F, NODE_S = "#eceae4", "#cfccc4"
GREEN_F, GREEN_S = "#7ba35a", "#5b7a3a"
BLUE_F, BLUE_S = "#7f9fcb", "#3f6487"
PINK_F, PINK_S = "#d98f8f", "#b85a5a"
RES_F, RES_S = "#e8b85c", "#a07a2b"
ATTN_LINE = "#2a9bb5"
RES_LINE = "#c98f2e"
GREEN_LINE = "#6f9a44"
INK, DIM = "#2d2a26", "#6e6a61"

TOKENS = ["The", "capital", "of", "France", "is"]
XCOL = [110, 250, 390, 530, 670]      # token column centres
XRES = 858                            # reservoir column centre
# node-row y-centres, bottom (embedding) -> top (output)
ROWS = [560, 505, 450, 392, 334, 276, 218]
ATTN_ROW = 3          # index into ROWS: the injection / attention layer
NODE_W, NODE_H = 42, 26

svg = []
def add(s): svg.append(s)

def node(cx, cy, fill=NODE_F, stroke=NODE_S, w=NODE_W, h=NODE_H, rx=5):
    add(f'<rect x="{cx-w/2:.0f}" y="{cy-h/2:.0f}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')

def arrow(x1, y1, x2, y2, stroke=DIM, w=1.4, marker="ah", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
        f'stroke="{stroke}" stroke-width="{w}"{d} marker-end="url(#{marker})"/>')

def text(x, y, s, size=12, fill=INK, anchor="middle", weight="normal", style=""):
    add(f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}"{style}>{s}</text>')

W, H = 960, 660
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
    f'font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif">')
add('<title>Reservoir-augmented residual stream</title>')
add('<desc>Token residual streams predicting "Paris" from "The capital of France is", '
    'with FFN-query, attention and FFN-value neurons, plus a reservoir column whose fixed '
    'random nodes enter the attention layer as extra keys/values and persist across passes.</desc>')
add('<defs>')
for mid, col in (("ah", DIM), ("ah-b", ATTN_LINE), ("ah-r", RES_LINE), ("ah-g", GREEN_LINE), ("ah-p", PINK_S)):
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" '
        f'markerHeight="6.5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" '
        f'stroke="{col}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>')
add('</defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#f6f2ec"/>')

# title
text(W/2, 34, "One attention layer reads and writes the reservoir as extra keys / values",
     size=15, weight="600")

# --- residual streams: vertical gray nodes + gray residual arrows ---
for ci, cx in enumerate(XCOL):
    for ri in range(len(ROWS)):
        node(cx, ROWS[ri])
    # residual arrows upward between rows
    for ri in range(len(ROWS) - 1):
        arrow(cx, ROWS[ri] - NODE_H/2, cx, ROWS[ri+1] + NODE_H/2, stroke="#b9b5ab", w=1.3)
    # token labels
    text(cx, 600, TOKENS[ci], size=13, weight="600")
    arrow(cx, 588, cx, ROWS[0] + NODE_H/2 + 2, stroke="#b9b5ab", w=1.3)

# --- FFN query neurons (green) beside the two lower layers ---
for cx in XCOL:
    for ri in (1, 2):
        gx, gy = cx + 30, ROWS[ri]
        node(gx, gy, GREEN_F, GREEN_S, w=18, h=18, rx=4)
        arrow(gx - 9, gy, cx + NODE_W/2, gy, stroke=GREEN_LINE, w=1.4, marker="ah-g")

# --- attention neurons (blue trapezoids) at the injection layer ---
ay = ROWS[ATTN_ROW]
attn_top = []   # apex points to draw attention lines from
for cx in XCOL:
    tw, bw, th = 26, 40, 26
    add(f'<polygon points="{cx-tw/2:.0f},{ay-th/2:.0f} {cx+tw/2:.0f},{ay-th/2:.0f} '
        f'{cx+bw/2:.0f},{ay+th/2:.0f} {cx-bw/2:.0f},{ay+th/2:.0f}" '
        f'fill="{BLUE_F}" stroke="{BLUE_S}" stroke-width="1"/>')
    attn_top.append((cx, ay - th/2))

# attention lines: every column's attention attends to the LAST column (like the reference)
dst = (XCOL[-1], ay - 14)
for (sx, sy) in attn_top[:-1]:
    add(f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{dst[0]-6:.0f}" y2="{dst[1]:.0f}" '
        f'stroke="{ATTN_LINE}" stroke-width="1.6" marker-end="url(#ah-b)"/>')

# --- FFN value neurons (pink) up the last column, producing the answer ---
lx = XCOL[-1]
for ri in (4, 5):
    px, py = lx + 30, ROWS[ri]
    node(px, py, PINK_F, PINK_S, w=18, h=18, rx=4)
    arrow(px - 9, py, lx + NODE_W/2, py, stroke=PINK_S, w=1.4, marker="ah-p")
# output arrow + answer
arrow(lx, ROWS[-1] - NODE_H/2, lx, 150, stroke="#b9b5ab", w=1.5)
text(lx, 140, "Paris", size=15, weight="700")

# --- RESERVOIR column (amber) at the injection-layer height ---
res_y = [ay - 78, ay - 50, ay - 22, ay + 6, ay + 34, ay + 62]
add(f'<rect x="{XRES-46:.0f}" y="{res_y[0]-22:.0f}" width="92" height="{res_y[-1]-res_y[0]+44:.0f}" '
    f'rx="12" fill="#faf1dd" stroke="{RES_S}" stroke-width="1.3" stroke-dasharray="5 3"/>')
text(XRES, res_y[0] - 30, "Reservoir", size=12.5, weight="700", fill="#7d5e1c")
for i, ry in enumerate(res_y):
    add(f'<circle cx="{XRES}" cy="{ry:.0f}" r="9" fill="{RES_F}" stroke="{RES_S}" stroke-width="1"/>')
# recurrence self-loop (W_r) on the right of the reservoir
add(f'<path d="M{XRES+46:.0f} {ay-30:.0f} q34 -16 34 18 q0 34 -30 24" fill="none" '
    f'stroke="{RES_LINE}" stroke-width="1.5" marker-end="url(#ah-r)"/>')
text(XRES + 86, ay - 4, "Wr", size=11, fill="#7d5e1c")
text(XRES, res_y[-1] + 34, "fixed · random", size=10.5, fill="#8a6a24")
text(XRES, res_y[-1] + 50, "persists across passes", size=10.5, fill="#8a6a24", style=' font-style="italic"')

# read (W_in) from last column's attention -> reservoir ; write (W_out) back
arrow(lx + bw/2 if False else lx + 22, ay - 4, XRES - 48, ay - 18, stroke=RES_LINE, w=1.7, marker="ah-r")
text((lx + XRES)/2, ay - 30, "W_in (read, fixed)", size=11, fill="#7d5e1c")
add(f'<line x1="{XRES-48:.0f}" y1="{ay+16:.0f}" x2="{lx+22:.0f}" y2="{ay+22:.0f}" '
    f'stroke="{PINK_S}" stroke-width="1.7" marker-end="url(#ah-p)"/>')
text((lx + XRES)/2, ay + 42, "W_out (write, learned)", size=11, fill=PINK_S)

# --- legend ---
ly = 636
items = [("FFN query", GREEN_F, GREEN_S), ("attention", BLUE_F, BLUE_S),
         ("FFN value", PINK_F, PINK_S), ("reservoir node (fixed, random)", RES_F, RES_S)]
lx0 = 70
for label, f, s in items:
    add(f'<rect x="{lx0}" y="{ly-11}" width="15" height="15" rx="3" fill="{f}" stroke="{s}"/>')
    text(lx0 + 21, ly + 1, label, size=11.5, anchor="start", fill=INK)
    lx0 += 40 + len(label) * 7.0

add('</svg>')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(svg) + "\n")
print("wrote", OUT)
