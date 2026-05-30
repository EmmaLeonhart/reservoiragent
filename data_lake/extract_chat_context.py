"""Extract preservable context from the Claude chat HTML exports in this folder.

The raw `*.html` exports + their `*_files/` asset folders are large (web chrome:
analytics, a 7.6 MB JS bundle, facebook-pixel) and, critically, embed a sidebar
"Recents" list of UNRELATED private conversation titles. We do NOT want those raw
exports committed to a public repo. Instead this script distils each export into
small, clean, committable artifacts:

  data_lake/transcripts/<slug>.md      - ordered User/Claude conversation turns
  data_lake/diagrams/<slug>-NN.svg     - embedded SVG architecture diagrams
  data_lake/diagrams/<slug>-preview-NN.webp - rendered diagram raster previews

The raw `*.html` + `*_files/` stay on disk (gitignored) as the source of record.

Re-runnable: overwrites its outputs. Run:  python data_lake/extract_chat_context.py
"""
import os
import re
import glob
import shutil
import sys
import io

from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS = os.path.join(HERE, "transcripts")
DIAGRAMS = os.path.join(HERE, "diagrams")


def slugify(name: str) -> str:
    name = name.replace(".html", "")
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip().lower()
    name = re.sub(r"[\s_-]+", "-", name)
    return name.strip("-") or "chat"


def html_to_text(el) -> str:
    """Light HTML -> markdown-ish text that preserves code blocks and lists."""
    for pre in el.find_all("pre"):
        code = pre.get_text("\n")
        pre.replace_with("\n```\n" + code.strip("\n") + "\n```\n")
    for li in el.find_all("li"):
        li.insert_before("- ")
    for br in el.find_all("br"):
        br.replace_with("\n")
    txt = el.get_text("\n")
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def extract(export_path: str):
    base = os.path.basename(export_path)
    slug = slugify(base)
    with open(export_path, encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

    # --- conversation turns, in document order ---
    order = {id(n): i for i, n in enumerate(soup.find_all(True))}
    items = []
    for node in soup.find_all(attrs={"data-testid": "user-message"}):
        items.append((order.get(id(node), 0), "User", node))
    for node in soup.select(".font-claude-response"):
        items.append((order.get(id(node), 0), "Claude", node))
    items.sort(key=lambda t: t[0])

    md = [f"# Transcript — {base.replace('.html','')}\n",
          "_Extracted from a Claude chat export. The private 'Recents' sidebar in the "
          "raw export is intentionally excluded._\n"]
    last_role = None
    for _, role, node in items:
        text = html_to_text(node)
        if not text:
            continue
        # collapse consecutive same-role fragments (Claude responses are split into blocks)
        if role == last_role and md:
            md.append("\n" + text)
        else:
            md.append(f"\n## {role}\n\n{text}")
            last_role = role
    os.makedirs(TRANSCRIPTS, exist_ok=True)
    out_md = os.path.join(TRANSCRIPTS, slug + ".md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    # --- embedded SVG diagrams ---
    # The architecture diagrams are rendered React widgets: their real SVG lives in
    # the `saved_resource*.html` / `isolated-segment.html` artifact files inside the
    # `_files/` folder, not (mostly) in the main export. Scan both, keep big SVGs,
    # and dedupe by content (the diagram was redrawn a few times across the chat).
    os.makedirs(DIAGRAMS, exist_ok=True)
    files_dir = export_path.replace(".html", "_files")
    html_sources = [export_path]
    if os.path.isdir(files_dir):
        html_sources += sorted(glob.glob(os.path.join(files_dir, "*.html")))

    seen_svg = set()
    n_svg = 0
    for src in html_sources:
        try:
            with open(src, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        for s in re.findall(r"<svg[\s\S]*?</svg>", raw, re.I):
            if len(s) < 3000:          # skip icons / tiny chrome SVGs
                continue
            # real diagrams carry labelled <text> nodes; chrome SVGs (gradients,
            # background illustrations) have none — require a few labels.
            if len(re.findall(r"<text", s, re.I)) < 5:
                continue
            key = re.sub(r"\s+", "", s)
            if key in seen_svg:
                continue
            seen_svg.add(key)
            n_svg += 1
            out = os.path.join(DIAGRAMS, f"{slug}-{n_svg:02d}.svg")
            with open(out, "w", encoding="utf-8") as ofh:
                ofh.write('<?xml version="1.0" encoding="UTF-8"?>\n' + s)

    # --- WEBP raster previews from the _files folder ---
    n_webp = 0
    if os.path.isdir(files_dir):
        for fn in sorted(os.listdir(files_dir)):
            fp = os.path.join(files_dir, fn)
            try:
                with open(fp, "rb") as fh:
                    head = fh.read(12)
            except OSError:
                continue
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                n_webp += 1
                out = os.path.join(DIAGRAMS, f"{slug}-preview-{n_webp:02d}.webp")
                shutil.copyfile(fp, out)

    print(f"{base}\n  -> {out_md}  ({len([i for i in items if html_to_text(i[2])])} turns)")
    print(f"  -> {n_svg} SVG, {n_webp} WEBP preserved")
    return slug, len(items), n_svg, n_webp


def main():
    exports = sorted(glob.glob(os.path.join(HERE, "*.html")))
    if not exports:
        print("No .html exports found in", HERE)
        return
    for p in exports:
        extract(p)
    print("\nDone. Committed artifacts: transcripts/ and diagrams/.")


if __name__ == "__main__":
    main()
