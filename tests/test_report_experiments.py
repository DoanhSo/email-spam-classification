import csv
import json

import arff

from spam_detector.inference import TextPredictor
from spam_detector.nb_release import NBTextPredictor
from spam_detector.report_experiments import run_report_experiments

SPAM = ["win free money now {i}", "free prize claim cash {i}", "cheap pills offer free {i}", "click here winner money {i}"]
HAM = ["meeting schedule for monday {i}", "please review the gas contract {i}", "notes from the team call {i}",
       "attached the report for review {i}"]


def _arff(path, n):
    rows = [[t.format(i=i), "spam"] for i in range(n) for t in SPAM] + [[t.format(i=i), "ham"] for i in range(n) for t in HAM]
    path.write_text(arff.dumps({"relation": "r", "attributes": [("text", "STRING"), ("class", ["ham", "spam"])],
                                "data": rows}), encoding="utf-8")
    return path


def test_report_run_small(tmp_path):
    external = tmp_path / "sa.csv"
    with open(external, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "Body", "Label"])
        writer.writerows([[0, "free money winner", 1], [1, "monday meeting review", 0], [2, "cheap offer click", 1]])
    out, artifacts = tmp_path / "results", tmp_path / "artifacts"
    summary = run_report_experiments(_arff(tmp_path / "train.arff", 3), _arff(tmp_path / "test.arff", 1), external,
                                     out, artifacts, k=20)
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert saved == summary
    for model in ("nb", "tree"):
        assert set(saved[model]) >= {"test", "external"}
        assert saved[model]["test"]["n_samples"] == 8 and saved[model]["external"]["n_samples"] == 3
    for name in ("confusion_nb_test", "confusion_nb_external", "confusion_tree_test",
                 "confusion_tree_external", "top_words_nb"):
        assert (out / "figures" / f"{name}.png").is_file(), name
    assert NBTextPredictor(artifacts / "nb-weka7030-v1").predict_texts(["free money"])[0]["label"] == 1
    assert TextPredictor(artifacts / "tree-weka7030-v1").predict_texts(["free money"])[0]["label"] in (0, 1)
