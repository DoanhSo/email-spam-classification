"""Draw the full J48 tree from Weka's printed model (reports/weka/results/j48_test.txt) as figures 23-25."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from spam_detector.weka_tree import collapse_branches, layout_tree, parse_j48_tree, render_tree, tree_stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "bao-cao" / "figures"
tree = parse_j48_tree((ROOT / "reports/weka/results/j48_test.txt").read_text(encoding="utf-8"))
branches = {("W_%", "> 0"): "nhánh A\n(Hình 25)", ("W_mobile", "> 0"): "nhánh B\n(Hình 25)",
            ("W_html", "> 0"): "nhánh C\n(Hình 25)", ("W_money", "> 0"): "nhánh D\n(Hình 25)",
            ("W_http", "<= 0"): "phần dưới\n(Hình 24)"}
main, cut = collapse_branches(tree, branches)
print("main width", max(x for x, _ in layout_tree(main, "first").values()) + 1)
render_tree(main, OUT / "j48-tree-a.png", align="first", unit_inches=0.95, level_inches=0.5, dpi=220)
lower = [sub for attribute, edge, sub in cut if attribute == "W_http"][0]
cut = [c for c in cut if c[0] != "W_http"]
print("lower", tree_stats(lower), "width", max(x for x, _ in layout_tree(lower, "first").values()) + 1)
render_tree(lower, OUT / "j48-tree-b.png", align="first", unit_inches=0.95, level_inches=0.5, dpi=220)

names = {"W_%": "A", "W_mobile": "B", "W_html": "C", "W_money": "D"}
parts = []
for attribute, edge, sub in cut:
    path = OUT / f"_sub_{names[attribute]}.png"
    print(names[attribute], tree_stats(sub), "width", max(x for x, _ in layout_tree(sub, "first").values()) + 1)
    render_tree(sub, path, align="first", unit_inches=0.95, level_inches=0.58, dpi=220)
    parts.append((f"Nhánh {names[attribute]}: {attribute[2:]} {edge}", Image.open(path).convert("RGB")))
    path.unlink()

parts.sort(key=lambda item: item[0])  # Nhánh A, B, C, D
font = ImageFont.truetype("arial.ttf", 34)
gap, title_h = 60, 60

def titled(title, im):
    tile = Image.new("RGB", (im.width, im.height + title_h), "white")
    ImageDraw.Draw(tile).text((10, 8), title, fill="black", font=font)
    tile.paste(im, (0, title_h))
    return tile

tiles = [titled(t, im) for t, im in parts]
left, right = tiles[0], tiles[1:]
right_w = max(t.width for t in right)
right_h = sum(t.height for t in right) + gap * (len(right) - 1)
canvas = Image.new("RGB", (left.width + gap + right_w, max(left.height, right_h)), "white")
canvas.paste(left, (0, 0))
y = 0
for t in right:
    canvas.paste(t, (left.width + gap, y))
    y += t.height + gap
canvas.save(OUT / "j48-tree-c.png")
print(Image.open(OUT / "j48-tree-a.png").size, Image.open(OUT / "j48-tree-b.png").size, canvas.size)
