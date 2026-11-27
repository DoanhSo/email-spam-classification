"""Parse the "J48 pruned tree" text printed by Weka and draw the full tree as an image."""

import re
from dataclasses import dataclass, field
from pathlib import Path

LINE = re.compile(r"^((?:\|   )*)(\S+) (<=|>) (\S+?)(?:: (\w+) \(([^)]*)\))?\s*$")


@dataclass
class TreeNode:
    attribute: str | None = None          # None for a leaf
    label: str | None = None              # leaf class
    counts: str | None = None             # leaf "(n/errors)" as printed by Weka
    children: list[tuple[str, "TreeNode"]] = field(default_factory=list)  # (edge text, child)

    @property
    def is_leaf(self) -> bool:
        return self.attribute is None


def parse_j48_tree(text: str) -> TreeNode:
    """Parse the block after "J48 pruned tree" up to "Number of Leaves"."""
    start = text.index("J48 pruned tree")
    end = text.index("Number of Leaves", start)
    lines = [l for l in text[start:end].splitlines()[1:] if l.strip() and not set(l.strip()) <= {"-"}]
    root = TreeNode()
    stack: list[TreeNode] = [root]  # stack[d] is the node whose tests are printed at depth d
    for line in lines:
        match = LINE.match(line.rstrip())
        if not match:
            raise ValueError(f"Unrecognised J48 tree line: {line!r}")
        bars, attribute, op, value, label, counts = match.groups()
        depth = len(bars) // 4
        del stack[depth + 1:]
        parent = stack[depth]
        if parent.attribute is None:
            parent.attribute = attribute
        elif parent.attribute != attribute:
            raise ValueError(f"Inconsistent split attribute at depth {depth}: {line!r}")
        child = TreeNode(label=label, counts=counts) if label else TreeNode()
        parent.children.append((f"{op} {value}", child))
        if not label:
            stack.append(child)
    if root.attribute is None:
        raise ValueError("Empty J48 tree")
    return root


def tree_stats(root: TreeNode) -> dict[str, int]:
    leaves = nodes = depth = 0

    def walk(node: TreeNode, d: int) -> None:
        nonlocal leaves, nodes, depth
        nodes += 1
        depth = max(depth, d)
        if node.is_leaf:
            leaves += 1
        for _, child in node.children:
            walk(child, d + 1)

    walk(root, 0)
    return {"leaves": leaves, "nodes": nodes, "depth": depth}


def layout_tree(root: TreeNode, align: str = "centre") -> dict[int, tuple[float, int]]:
    """Compact tidy layout: children packed by contours. The parent is centred over its children, or with
    align="first" placed straight above its first ("<=") child so the main branch runs vertically.
    Returns id(node) -> (x, depth)."""
    if align not in ("centre", "first"):
        raise ValueError("align must be 'centre' or 'first'")

    def place(node: TreeNode) -> tuple[dict[int, float], list[tuple[float, float]], dict[int, int]]:
        if node.is_leaf or not node.children:
            return {id(node): 0.0}, [(0.0, 0.0)], {id(node): 0}
        xs: dict[int, float] = {}
        depths: dict[int, int] = {}
        contour: list[tuple[float, float]] = []
        roots: list[float] = []
        for _, child in node.children:
            cx, cc, cd = place(child)
            shift = 0.0
            if contour:
                shift = max(contour[i][1] - cc[i][0] + 1.0 for i in range(min(len(contour), len(cc))))
            for key, value in cx.items():
                xs[key] = value + shift
            depths.update(cd)
            roots.append(shift)
            for i, (lo, hi) in enumerate(cc):
                if i < len(contour):
                    contour[i] = (min(contour[i][0], lo + shift), max(contour[i][1], hi + shift))
                else:
                    contour.append((lo + shift, hi + shift))
        centre = roots[0] if align == "first" else (roots[0] + roots[-1]) / 2
        xs = {key: value - centre for key, value in xs.items()}
        depths = {key: value + 1 for key, value in depths.items()}
        xs[id(node)], depths[id(node)] = 0.0, 0
        return xs, [(0.0, 0.0)] + [(lo - centre, hi - centre) for lo, hi in contour], depths

    xs, _, depths = place(root)
    left = min(xs.values())
    return {key: (x - left, depths[key]) for key, x in xs.items()}


def render_tree(root: TreeNode, path: str | Path, strip_prefix: str = "W_", unit_inches: float = 1.15,
                level_inches: float = 0.62, dpi: int = 200, align: str = "centre") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = layout_tree(root, align)
    width = max(x for x, _ in positions.values()) + 1
    depth = max(d for _, d in positions.values()) + 1
    fig, ax = plt.subplots(figsize=(width * unit_inches, depth * level_inches))
    ax.set_xlim(-0.6, width - 0.4)
    ax.set_ylim(-depth + 0.4, 0.6)
    ax.axis("off")

    def name(attribute: str) -> str:
        return attribute[len(strip_prefix):] if strip_prefix and attribute.startswith(strip_prefix) else attribute

    def draw(node: TreeNode) -> None:
        x, d = positions[id(node)]
        for edge, child in node.children:
            cx, cd = positions[id(child)]
            # Elbow edge: down half a level, across, then down. Rows hold only boxes, the half levels hold
            # only connectors, so long edges never pass behind another node's box.
            mid = -d - 0.5
            ax.plot([x, x, cx, cx], [-d, mid, mid, -cd], color="#7a8793", linewidth=0.6, zorder=1)
            ax.text(cx + 0.04, mid - 0.04, edge, fontsize=5.5, ha="left", va="top", color="#33414d", zorder=2)
            draw(child)
        if node.is_leaf:
            colour = {"spam": "#f6d2d2", "ham": "#d3e3f5"}.get(node.label, "#fff3c4")
            text = f"{node.label}\n({node.counts})" if node.counts else node.label
            ax.text(x, -d, text, fontsize=5.5, ha="center", va="center", zorder=3,
                    bbox={"boxstyle": "square,pad=0.25", "fc": colour, "ec": "#5c6b78", "lw": 0.5})
        else:
            ax.text(x, -d, name(node.attribute), fontsize=6, ha="center", va="center", zorder=3,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "#e9edf0", "ec": "#5c6b78", "lw": 0.5})

    draw(root)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def collapse_branches(root: TreeNode, branches: dict[tuple[str, str], str]
                      ) -> tuple[TreeNode, list[tuple[str, str, TreeNode]]]:
    """Copy the tree, replacing each (attribute, edge) child listed in `branches` by a placeholder leaf.

    Returns the collapsed copy and the removed subtrees as (attribute, edge, subtree), in tree order.
    """
    cut: list[tuple[str, str, TreeNode]] = []

    def copy(node: TreeNode) -> TreeNode:
        new = TreeNode(node.attribute, node.label, node.counts)
        for edge, child in node.children:
            key = (node.attribute, edge)
            if key in branches:
                cut.append((node.attribute, edge, child))
                new.children.append((edge, TreeNode(label=branches[key], counts=None)))
            else:
                new.children.append((edge, copy(child)))
        return new

    return copy(root), cut
