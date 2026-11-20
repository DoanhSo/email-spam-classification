import json

import numpy as np

from spam_detector.models.naive_bayes import MultinomialNaiveBayes
from spam_detector.weka_compare import compare_with_weka, read_weka_predictions

HEADER = "@relation v\n@attribute class {ham,spam}\n@attribute a numeric\n@attribute b numeric\n@attribute c numeric\n@data\n"
TRAIN = [("ham", 2, 1, 0), ("spam", 0, 1, 3), ("ham", 1, 0, 0), ("spam", 0, 0, 2)]
TEST = [("ham", 1, 0, 1), ("spam", 0, 0, 2), ("spam", 3, 0, 0)]


def _vec(path, rows):
    path.write_text(HEADER + "".join(",".join(map(str, r)) + "\n" for r in rows), encoding="utf-8")
    return path


def test_read_weka_predictions_variants(tmp_path):
    path = tmp_path / "pred.csv"
    path.write_text("inst#,actual,predicted,error,prediction\n1,1:ham,1:ham,,0.9\n2,2:spam,1:ham,+,0.7\n\n",
                    encoding="utf-8")
    actual, predicted, confidence = read_weka_predictions(path)
    assert actual.tolist() == [0, 1] and predicted.tolist() == [0, 0]
    assert np.allclose(confidence, [0.9, 0.7])


def test_compare_identical_on_fixture(tmp_path):
    train, test = _vec(tmp_path / "train.arff", TRAIN), _vec(tmp_path / "test.arff", TEST)
    X_train = np.array([r[1:] for r in TRAIN]); y_train = np.array([r[0] == "spam" for r in TRAIN], dtype=int)
    X_test = np.array([r[1:] for r in TEST]); y_test = [r[0] for r in TEST]
    model = MultinomialNaiveBayes().fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    lines = ["inst#,actual,predicted,error,prediction"]
    for i, (label, row) in enumerate(zip(y_test, proba), start=1):
        p = int(row[1] > 0.5)
        lines.append(f"{i},{1 + (label == 'spam')}:{label},{p + 1}:{'spam' if p else 'ham'},"
                     f"{'+' if (label == 'spam') != p else ''},{row[p]:.3f}")
    weka = tmp_path / "pred.csv"
    weka.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = compare_with_weka(train, test, weka, tmp_path / "out")
    assert result["n_test"] == 3 and result["agreement"] == 3 and result["disagreements"] == []
    assert result["max_abs_proba_diff"] < 1e-3
    assert result["python_metrics"] == result["weka_metrics"]
    assert json.loads((tmp_path / "out" / "nb_python_vs_weka.json").read_text(encoding="utf-8"))["agreement"] == 3
