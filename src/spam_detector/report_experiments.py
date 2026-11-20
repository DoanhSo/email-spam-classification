"""Python experiments for the report: NB and tree trained on Weka's 70/30 split, tested on test and SpamAssassin."""

import json
from pathlib import Path

import numpy as np

from .artifacts import save_tree_bundle
from .data import clean_csv, sha256, write_json
from .features import BinaryTextFeatures
from .models.decision_tree import DecisionTreeClassifier
from .models.naive_bayes import MultinomialNaiveBayes
from .nb_release import save_nb_release
from .preprocessing import PREPROCESSING_VERSION
from .tree_workflow import classification_metrics
from .weka_io import read_text_arff

TREE_PARAMETERS = {"max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 1}


def _save_tree_release(root: Path, model: DecisionTreeClassifier, features: BinaryTextFeatures, metadata: dict) -> None:
    save_tree_bundle(root / "tree.json", model, features.get_feature_names(), metadata)
    features.save(root / "features.json")
    write_json(root / "release.json", {
        "format_version": 1, "preprocessing_version": PREPROCESSING_VERSION, "selected": metadata,
        "checksums": {name: sha256(root / name) for name in ("tree.json", "features.json")},
    })


def _confusion_figure(path: Path, metrics: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, axis = plt.subplots(figsize=(4.2, 3.6), layout="constrained")
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["ham", "spam"], yticklabels=["ham", "spam"],
             xlabel="Lớp dự đoán", ylabel="Lớp thực tế", title=title)
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _top_words_figure(path: Path, model: MultinomialNaiveBayes, names: list[str], n: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_odds = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    spam, ham = np.argsort(-log_odds)[:n // 2], np.argsort(log_odds)[:n // 2]
    order = np.concatenate([ham[::-1], spam[::-1]])
    fig, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    axis.barh([names[i] for i in order], log_odds[order], color=["#4c72b0" if log_odds[i] < 0 else "#c44e52" for i in order])
    axis.set(xlabel="log P(w|spam) − log P(w|ham)", title="Các từ đặc trưng nhất của mỗi lớp (Naive Bayes)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_report_experiments(train_arff, test_arff, external_csv, output_dir, artifacts_dir, k: int = 1000) -> dict:
    train, test = read_text_arff(train_arff), read_text_arff(test_arff)
    external, _ = clean_csv(external_csv, "spamassassin")
    splits = {"test": test, "external": external}
    texts, labels = [r["text"] for r in train], np.array([r["label"] for r in train])
    features = BinaryTextFeatures(k=k, min_df=2).fit(texts, labels)
    X_train = features.transform(texts)
    models = {
        "nb": MultinomialNaiveBayes(alpha=1.0).fit(X_train, labels),
        "tree": DecisionTreeClassifier(**TREE_PARAMETERS).fit(X_train, labels),
    }
    output, artifacts = Path(output_dir), Path(artifacts_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for name, model in models.items():
        results[name] = {}
        for split, rows in splits.items():
            prediction = model.predict(features.transform([r["text"] for r in rows]))
            metrics = classification_metrics([r["label"] for r in rows], prediction)
            results[name][split] = metrics
            _confusion_figure(output / "figures" / f"confusion_{name}_{split}.png", metrics,
                              f"{'Naive Bayes' if name == 'nb' else 'Cây quyết định'} – {split}")
    _top_words_figure(output / "figures" / "top_words_nb.png", models["nb"], features.get_feature_names())
    metadata = {"train_rows": len(train), "test_rows": len(test), "external_rows": len(external), "k": k,
                "train_sha256": sha256(train_arff), "test_sha256": sha256(test_arff)}
    save_nb_release(artifacts / "nb-weka7030-v1", models["nb"], features, {**metadata, "alpha": 1.0})
    _save_tree_release(artifacts / "tree-weka7030-v1", models["tree"], features,
                       {**metadata, "model": "self_written_binary_tree", "parameters": TREE_PARAMETERS})
    write_json(output / "results.json", results)
    return json.loads(json.dumps(results))
