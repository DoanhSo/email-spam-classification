"""ARFF files exchanged with Weka: raw Enron export, text splits and filtered vectors."""

import csv
import sys
from pathlib import Path

import arff
import numpy as np

CLASSES = {"ham": 0, "spam": 1}


def write_enron_arff(csv_path: str | Path, out_path: str | Path) -> int:
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header: <index>, label, text, label_num
        data = [[int(row[0]), int(row[3]), row[2], row[1]] for row in reader]
    if any(row[3] not in CLASSES for row in data):
        raise ValueError("Enron labels must be ham or spam")
    payload = {
        "relation": "enron_spam",
        "attributes": [
            ("row_index", "NUMERIC"), ("label_num", "NUMERIC"),
            ("text", "STRING"), ("class", ["ham", "spam"]),
        ],
        "data": data,
    }
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(arff.dumps(payload), encoding="utf-8", newline="\n")
    return len(data)


def _load(path: str | Path, sparse: bool = False) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    return arff.loads(text, return_type=arff.COO if sparse else arff.DENSE)


def _class_index(names: list[str], attributes: list) -> int:
    if "class" not in names:
        raise ValueError("ARFF has no 'class' attribute")
    index = names.index("class")
    if sorted(attributes[index][1]) != ["ham", "spam"]:
        raise ValueError("Class values must be {ham, spam}")
    return index


def read_text_arff(path: str | Path) -> list[dict]:
    raw = _load(path)
    names = [name for name, _ in raw["attributes"]]
    if "text" not in names:
        raise ValueError("ARFF has no 'text' attribute")
    t, c = names.index("text"), _class_index(names, raw["attributes"])
    return [{"text": row[t], "label": CLASSES[row[c]]} for row in raw["data"]]


def read_vector_arff(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    content = Path(path).read_text(encoding="utf-8")
    marker = content.lower().find("@data")
    body = content[marker + 5:] if marker >= 0 else ""
    first = next((line.strip() for line in body.splitlines() if line.strip() and not line.lstrip().startswith("%")), "")
    is_sparse = first.startswith("{")
    # LOD keeps one dict per sparse row, so rows written as "{}" are not lost.
    raw = arff.loads(content, return_type=arff.LOD if is_sparse else arff.DENSE)
    names = [name for name, _ in raw["attributes"]]
    c = _class_index(names, raw["attributes"])
    features = [i for i in range(len(names)) if i != c]
    for i in features:
        if str(raw["attributes"][i][1]).upper() not in ("NUMERIC", "REAL", "INTEGER"):
            raise ValueError(f"Feature attribute '{names[i]}' is not numeric")
    if is_sparse:
        rows = raw["data"]
        X = np.zeros((len(rows), len(names)), dtype=np.float64)
        y = np.zeros(len(rows), dtype=np.int64)
        # Sparse rows omit zero values; for a nominal class, omitted means the first declared value.
        y[:] = CLASSES[raw["attributes"][c][1][0]]
        for r, row in enumerate(rows):
            for col, value in row.items():
                if col == c:
                    y[r] = CLASSES[value]
                else:
                    X[r, col] = float(value)
        return X[:, features], y, [names[i] for i in features]
    data = raw["data"]
    X = np.array([[float(row[i]) for i in features] for row in data], dtype=np.float64).reshape(len(data), len(features))
    y = np.array([CLASSES[row[c]] for row in data], dtype=np.int64)
    return X, y, [names[i] for i in features]
