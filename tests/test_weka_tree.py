import pytest

from spam_detector.weka_tree import layout_tree, parse_j48_tree, render_tree, tree_stats

SAMPLE = """J48 pruned tree
------------------

W_free <= 0
|   W_enron <= 0
|   |   W_cash <= 1: ham (10.0/2.0)
|   |   W_cash > 1: spam (4.0)
|   W_enron > 0: ham (30.0)
W_free > 0: spam (12.0/1.0)

Number of Leaves  : \t4

Size of the tree : \t7
"""


def test_parse_structure_and_stats():
    root = parse_j48_tree(SAMPLE)
    assert root.attribute == "W_free"
    assert [edge for edge, _ in root.children] == ["<= 0", "> 0"]
    right = root.children[1][1]
    assert right.is_leaf and right.label == "spam" and right.counts == "12.0/1.0"
    assert root.children[0][1].children[0][1].children[0][0] == "<= 1"
    assert tree_stats(root) == {"leaves": 4, "nodes": 7, "depth": 3}


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_j48_tree("J48 pruned tree\n---\nthis is not a tree line\nNumber of Leaves : 1\n")


def test_layout_has_no_overlap_on_any_level():
    positions = layout_tree(parse_j48_tree(SAMPLE))
    assert len(positions) == 7
    by_level = {}
    for x, depth in positions.values():
        by_level.setdefault(depth, []).append(x)
    for xs in by_level.values():
        xs.sort()
        assert all(b - a >= 1 for a, b in zip(xs, xs[1:]))


def test_render_writes_png(tmp_path):
    out = tmp_path / "tree.png"
    render_tree(parse_j48_tree(SAMPLE), out)
    assert out.stat().st_size > 1000


def test_layout_first_aligned_keeps_main_branch_vertical():
    root = parse_j48_tree(SAMPLE)
    positions = layout_tree(root, align="first")
    first_child = root.children[0][1]
    assert positions[id(root)][0] == positions[id(first_child)][0]


def test_collapse_replaces_named_branch_with_placeholder():
    from spam_detector.weka_tree import collapse_branches

    root = parse_j48_tree(SAMPLE)
    view, cut = collapse_branches(root, {("W_enron", "<= 0"): "xem Hình B"})
    enron = view.children[0][1]
    placeholder = enron.children[0][1]
    assert placeholder.is_leaf and placeholder.label == "xem Hình B"
    assert [(attr, edge) for attr, edge, _ in cut] == [("W_enron", "<= 0")]
    assert cut[0][2].attribute == "W_cash"
    assert root.children[0][1].children[0][1].attribute == "W_cash"  # original untouched
