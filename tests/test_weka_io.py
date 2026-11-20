import csv

import numpy as np
import pytest

from spam_detector.weka_io import read_text_arff, read_vector_arff, write_enron_arff

ROWS = [
    ("10", "ham", 'Subject: it\'s "fine"\nline two \\ back', "0"),
    ("11", "spam", "Subject: café offer\r\nwin money", "1"),
    ("12", "ham", "plain text", "0"),
]


def _enron_csv(path):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "label", "text", "label_num"])
        writer.writerows(ROWS)
    return path


def test_enron_roundtrip(tmp_path):
    out = tmp_path / "enron.arff"
    assert write_enron_arff(_enron_csv(tmp_path / "e.csv"), out) == 3
    records = read_text_arff(out)
    assert [r["text"] for r in records] == [row[2] for row in ROWS]
    assert [r["label"] for r in records] == [0, 1, 0]
    header = out.read_text(encoding="utf-8").lower()
    for name in ("row_index", "label_num", "text", "class"):
        assert f"@attribute {name}" in header


def test_read_weka_escaped_strings(tmp_path):
    path = tmp_path / "weka.arff"
    path.write_text(
        "@relation t\n\n@attribute text string\n@attribute class {ham,spam}\n\n@data\n"
        "'it\\'s a \\\\n test\\nline2',spam\n'ok',ham\n",
        encoding="utf-8",
    )
    records = read_text_arff(path)
    assert records[0] == {"text": "it's a \\n test\nline2", "label": 1}
    assert records[1] == {"text": "ok", "label": 0}


def test_vector_arff_sparse(tmp_path):
    path = tmp_path / "vec.arff"
    path.write_text(
        "@relation v\n@attribute a numeric\n@attribute b numeric\n@attribute c numeric\n"
        "@attribute class {ham,spam}\n@data\n{0 2,2 1,3 spam}\n{1 3,3 ham}\n",
        encoding="utf-8",
    )
    X, y, names = read_vector_arff(path)
    assert X.tolist() == [[2, 0, 1], [0, 3, 0]]
    assert X.dtype == np.float64
    assert y.tolist() == [1, 0]
    assert names == ["a", "b", "c"]


def test_vector_arff_class_by_name(tmp_path):
    path = tmp_path / "vec.arff"
    path.write_text(
        "@relation v\n@attribute class {spam,ham}\n@attribute a numeric\n@data\nham,1\nspam,4\n",
        encoding="utf-8",
    )
    X, y, names = read_vector_arff(path)
    assert X.tolist() == [[1], [4]] and y.tolist() == [0, 1] and names == ["a"]


def test_vector_arff_rejects_missing_class(tmp_path):
    path = tmp_path / "vec.arff"
    path.write_text("@relation v\n@attribute a numeric\n@data\n1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_vector_arff(path)
