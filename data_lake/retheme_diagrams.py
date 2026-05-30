"""Re-theme the recovered architecture SVGs so they render correctly *standalone*.

The diagrams the user wants on the report (forward-pass = `-01`, runtime = `-02`)
are real vector SVGs, but they were authored against Claude's web stylesheet: their
boxes use CSS classes (`node c-teal`, `c-purple`, ...) and CSS custom properties
(`--color-...`) that are undefined standalone, so every box falls back to solid black.

Browsers resolve CSS variables and class selectors, but the other renderers this
project relies on — cairosvg (for the social-preview PNG) and the PDF builder — do
NOT. So instead of injecting a `<style>`, we BAKE literal fill/stroke onto every box
and resolve every `var(--...)` to a literal colour, in the report's light "paper"
palette (matching the user's screenshots). Renders identically in browser, cairosvg,
and PDF. Re-runnable.

    python data_lake/retheme_diagrams.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "diagrams")
DOCS = os.path.normpath(os.path.join(HERE, "..", "docs"))

JOBS = {
    "reservoir-state-injection-in-transformer-architecture-claude-01.svg": "diagram-architecture.svg",
    "reservoir-state-injection-in-transformer-architecture-claude-02.svg": "diagram-runtime.svg",
}

# Resolve the CSS custom properties to literal colours (light palette).
VARS = {
    "--color-background-primary": "#ffffff",
    "--color-background-secondary": "#f7f6f3",
    "--color-border-secondary": "#d8d4cb",
    "--color-border-tertiary": "#c6c1b6",
    "--color-text-primary": "#2d2a26",
    "--color-text-secondary": "#54515a",
    "--color-text-tertiary": "#8a877f",
}

# Per node colour class: (rect fill, rect stroke, header-text fill).
COLORS = {
    "c-gray":   ("#f2f1ed", "#cdc9c0", "#3f3d44"),
    "c-teal":   ("#e4f0ec", "#3f9a86", "#0f6e56"),
    "c-purple": ("#ece9f7", "#8978d6", "#5a49bd"),
    "c-amber":  ("#faf1dd", "#c98f2e", "#8a6313"),
}


# Normalise the few glyphs that common headless/PDF fonts lack (subscripts, the
# heavy multiplication X) to portable equivalents, so the SVG renders identically in
# a browser, in cairosvg (social card), and in the PDF builder.
GLYPHS = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "ₖ": "k", "ₙ": "n",
    "✕": "×",  # heavy multiplication X -> multiplication sign
}


def retheme(t: str) -> str:
    # The recovered <svg> tag has no namespace; browsers loading an SVG via <img>
    # require xmlns or they render nothing (cairosvg is lenient, which hid this).
    t = re.sub(
        r"<svg\b(?![^>]*\bxmlns=)",
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"',
        t,
        count=1,
    )
    for bad, good in GLYPHS.items():
        t = t.replace(bad, good)
    # strip interactive handlers + pointer cursor
    t = re.sub(r'\s+onclick="[^"]*"', "", t)
    t = t.replace("cursor: pointer;", "").replace("cursor:pointer;", "")
    # cairosvg/PDF don't support context-stroke arrowheads
    t = t.replace('stroke="context-stroke"', 'stroke="#8a877f"')
    # resolve every CSS variable to a literal colour
    for var, val in VARS.items():
        t = t.replace(f"var({var})", val)

    # bake fill/stroke onto each node box and colour its header text
    def repl_group(m):
        color, inner = m.group(1), m.group(2)
        rf, rs, th = COLORS.get(color, ("#f7f6f3", "#d8d4cb", "#2d2a26"))
        # add fill+stroke to the box rect (the first <rect> with no fill of its own)
        inner = re.sub(
            r"<rect ((?:(?!fill=)[^>])*?)>",
            f'<rect fill="{rf}" stroke="{rs}" \\1>',
            inner,
            count=1,
        )
        inner = inner.replace(
            '<text class="th"', f'<text class="th" fill="{th}" font-weight="600" font-size="13.5"'
        )
        inner = inner.replace(
            '<text class="ts"', '<text class="ts" fill="#54515a" font-size="11"'
        )
        return f'<g class="node {color}">{inner}</g>'

    t = re.sub(r'<g class="node (c-\w+)"[^>]*>(.*?)</g>', repl_group, t, flags=re.S)

    # any th/ts text outside a node box: give it a literal fill + size too
    t = re.sub(r'<text class="th"(?![^>]*fill=)',
               '<text class="th" fill="#2d2a26" font-weight="600" font-size="13.5"', t)
    t = re.sub(r'<text class="ts"(?![^>]*fill=)',
               '<text class="ts" fill="#54515a" font-size="11"', t)
    return t


def main():
    os.makedirs(DOCS, exist_ok=True)
    for src, out in JOBS.items():
        with open(os.path.join(SRC, src), encoding="utf-8") as fh:
            t = fh.read()
        with open(os.path.join(DOCS, out), "w", encoding="utf-8") as fh:
            fh.write(retheme(t))
        print(f"{src}  ->  docs/{out}")


if __name__ == "__main__":
    main()
