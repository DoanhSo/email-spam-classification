"""Reproducible real-data training, validation selection, CV and final evaluation."""

import json
import shutil
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .artifacts import load_tree_bundle
from .config import ProjectConfig
from .data import download_dataset, prepare_data, read_records, sha256, write_csv, write_json
from .features import BinaryTextFeatures, build_features
from .inference import TextPredictor
from .models.decision_tree import DecisionTreeClassifier
from .preprocessing import PREPROCESSING_VERSION
from .tree_workflow import classification_metrics, evaluate_tree, train_tree


def cross_validate(records: list[dict], candidates: list[dict], seed: int = 42) -> dict:
    """Every fold starts with raw training text and a fresh feature selector."""
    rows = [r for r in records if r["split"] == "train"]
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("CV expects exactly one deduplicated email per group")
    y = np.array([r["label"] for r in rows])
    if min(np.bincount(y, minlength=2)) < 5:
        raise ValueError("Five-fold CV requires at least 5 train samples per class")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.arange(len(rows)), y))
    results, fold_ids = [], []
    for fold, (train, validation) in enumerate(folds, start=1):
        fold_ids.extend({"sample_id": rows[int(i)]["sample_id"], "fold": fold, "role": "validation"} for i in validation)
    for candidate in candidates:
        for fold, (train, validation) in enumerate(folds, start=1):
            started = perf_counter()
            features = BinaryTextFeatures(candidate["k"]).fit([rows[int(i)]["text"] for i in train], y[train])
            X_train = features.transform([rows[int(i)]["text"] for i in train])
            X_val = features.transform([rows[int(i)]["text"] for i in validation])
            transform_seconds = perf_counter() - started
            model = DecisionTreeClassifier(**candidate["parameters"])
            started = perf_counter()
            model.fit(X_train, y[train])
            fit_seconds = perf_counter() - started
            prediction = model.predict(X_val)
            results.append({
                "candidate": candidate["name"], "fold": fold, "k": candidate["k"],
                "effective_k": len(features.get_feature_names()),
                **classification_metrics(y[validation], prediction),
                "feature_seconds": transform_seconds, "fit_seconds": fit_seconds,
            })
    summary = {}
    for candidate in candidates:
        selected = [r for r in results if r["candidate"] == candidate["name"]]
        summary[candidate["name"]] = {
            metric: {"mean": float(np.mean([r[metric] for r in selected])),
                     "std": float(np.std([r[metric] for r in selected], ddof=1))}
            for metric in ("precision_spam", "recall_spam", "f1_spam")
        }
    return {"folds": results, "fold_assignments": fold_ids, "summary": summary,
            "policy": "5-fold on train only; fresh vocabulary/Chi-square per fold; post-selection diagnostic, not nested CV"}


def evaluate_release(release_dir: Path, rows: list[dict], output: Path, split: str) -> dict:
    if not rows or any(r["split"] != split for r in rows):
        raise ValueError("Evaluation must receive one nonempty split")
    predictor = TextPredictor(release_dir)
    # Enforce the same ID/group safeguards as the matrix evaluation entry point.
    for origin in ("train", "validation"):
        if split in ("test", "external_test"):
            if {r["sample_id"] for r in rows} & set(predictor.bundle.metadata.get(f"{origin}_sample_ids", [])):
                raise ValueError(f"Evaluation IDs overlap {origin}")
            if {r["group_id"] for r in rows} & set(predictor.bundle.metadata.get(f"{origin}_group_ids", [])):
                raise ValueError(f"Evaluation groups overlap {origin}")
    started = perf_counter()
    X = predictor.features.transform([r["text"] for r in rows])
    transform_seconds = perf_counter() - started
    started = perf_counter()
    prediction = predictor.bundle.model.predict(X)
    score = predictor.bundle.model.predict_proba(X)[:, 1]
    predict_seconds = perf_counter() - started
    y = np.array([r["label"] for r in rows])
    metrics = classification_metrics(y, prediction)
    if len(np.unique(y)) == 2:
        metrics.update(roc_auc=float(roc_auc_score(y, score)), average_precision=float(average_precision_score(y, score)))
    result = {"split": split, "metrics": metrics, "transform_seconds": transform_seconds,
              "predict_and_probability_seconds": predict_seconds,
              "zero_selected_features": int(np.sum(~X.any(axis=1)))}
    output.mkdir(parents=True, exist_ok=True)
    predictions = [
        {"sample_id": r["sample_id"], "group_id": r["group_id"], "y_true": r["label"],
         "y_pred": int(p), "spam_fraction": float(s), "score_kind": "uncalibrated_leaf_fraction"}
        for r, p, s in zip(rows, prediction, score)
    ]
    write_csv(output / f"predictions.{split}.csv", predictions)
    errors = [
        {**p, "error_type": "FP" if p["y_pred"] else "FN", "text": r["text"]}
        for r, p in zip(rows, predictions) if p["y_true"] != p["y_pred"]
    ]
    write_csv(output / f"errors.{split}.csv", errors,
              list(predictions[0]) + ["error_type", "text"])
    write_json(output / f"metrics.{split}.json", result)
    return result


def _figures(output: Path, records: list[dict], results: dict, cv: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "figures"
    folder.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, split in zip(axes, ("test", "external_test")):
        m = results[split]["metrics"]
        matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        axis.imshow(matrix, cmap="Blues")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["ham", "spam"],
                 yticklabels=["ham", "spam"], xlabel="Predicted class", ylabel="Actual class",
                 title=f"Confusion matrix: {split} (emails)")
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
    fig.suptitle("Frozen Decision Tree | Enron test and cleaned SpamAssassin")
    fig.savefig(folder / "confusion-matrices.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    names = ["train", "validation", "test", "external_test"]
    ham = [sum(r["split"] == s and r["label"] == 0 for r in records) for s in names]
    spam = [sum(r["split"] == s and r["label"] == 1 for r in records) for s in names]
    axis.bar(names, ham, label="ham")
    axis.bar(names, spam, bottom=ham, label="spam")
    axis.set(xlabel="Dataset split", ylabel="Email count (after cleaning)", title="Class counts after deduplication and cross-source overlap removal")
    axis.legend()
    fig.savefig(folder / "class-distribution.png", dpi=160)
    plt.close(fig)


def generate_report(output: Path, summary: dict, records: list[dict]) -> None:
    lines = ["# Kết quả huấn luyện Decision Tree trên email thật", "",
             "Mô hình tự viết bằng NumPy; vocabulary/Chi-square dùng thư viện. Không dùng cây sklearn để dự đoán trong release.", "",
             "## Dữ liệu", "", "Nguồn: [Enron](https://www.kaggle.com/datasets/venky73/spam-mails-dataset), [SpamAssassin](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset).",
             "", "Chi tiết làm sạch và checksum: `processed/data_quality.json` và `provenance.json`.", ""]
    for source in ("enron", "spamassassin"):
        stats = summary["data_quality"][source]
        lines.append(f"- {source}: raw={stats['raw_rows']}; rỗng/không token={stats['empty_or_no_tokens_removed']}; trùng loại={stats['duplicate_rows_removed']}; nhãn mâu thuẫn loại={stats['conflicting_rows_removed']}; giữ={stats['kept_rows']}.")
    lines.append(f"- External loại trùng Enron: {summary['data_quality']['spamassassin']['overlap_with_enron_removed']}.")
    for split, counts in summary["data_quality"]["splits"].items():
        lines.append(f"- {split}: {counts['total']} email, ham={counts['ham']}, spam={counts['spam']}.")
    selected = summary["selected"]
    lines += ["", "## Chọn cấu hình", "", f"Đã thử 18 ứng viên (K=500/1000, depth=5/10/15, leaf=1/5/10), min_samples_split=10. Chọn bằng F1 validation, rồi Precision, cây nông/ít nút; không chọn bằng test.",
              "", f"Chọn K={selected['k']}, parameters={json.dumps(selected['parameters'])}; depth thực={selected['depth']}, nodes={selected['nodes']}.",
              "", "## Kết quả", "", "| Tập | Precision spam | Recall spam | F1 spam | FP | FN |", "|---|---:|---:|---:|---:|---:|"]
    for split, result in summary["evaluation"].items():
        m = result["metrics"]
        lines.append(f"| {split} | {m['precision_spam']:.4f} | {m['recall_spam']:.4f} | {m['f1_spam']:.4f} | {m['fp']} | {m['fn']} |")
    lines += ["", "## Cross-validation trên phần train", "", "Mỗi fold fit mới vocabulary, Chi-square và cây. CV thực hiện sau chọn cấu hình như phép kiểm tra độ ổn định; không phải nested CV hoặc ước lượng độc lập cho bước lựa chọn.", ""]
    for name, metrics in summary["cv"]["summary"].items():
        score = metrics["f1_spam"]
        lines.append(f"- {name}: F1 mean={score['mean']:.4f}, sample std={score['std']:.4f} (5 fold).")
    lines += ["", "## Phân tích lỗi và giới hạn", "", "Các file errors.test.csv và errors.external_test.csv giữ nội dung cục bộ để đọc và lần theo luật; không đưa toàn bộ email vào Git."]
    external_metrics = summary["evaluation"]["external_test"]["metrics"]
    external_fp_rate = external_metrics["fp"] / (external_metrics["fp"] + external_metrics["tn"])
    lines += ["", f"External chặn nhầm {external_metrics['fp']} thư thật, chiếm {external_fp_rate:.2%} số ham. Kết quả ngoài nguồn này cho thấy cấu hình hiện tại chưa tổng quát tốt; không chỉnh lại mô hình theo external/test trong đợt này.", ""]
    predictor = TextPredictor(output / "release")
    for split in ("test", "external_test"):
        errors_path = output / "evaluation" / f"errors.{split}.csv"
        import csv
        csv.field_size_limit(20_000_000)
        with errors_path.open(encoding="utf-8", newline="") as stream:
            errors = list(csv.DictReader(stream))
        for kind in ("FP", "FN"):
            example = next((r for r in errors if r["error_type"] == kind), None)
            if example:
                explanation = predictor.predict_texts([example["text"]])[0]
                path = "; ".join(f"{step['word']}={'có' if step['present'] else 'không'}" for step in explanation["path"])
                lines.append(f"- {split} {kind}: ID `{example['sample_id']}`; đường đi: {path}; lá có {explanation['leaf_training_samples']} mẫu, tỷ lệ spam={explanation['spam_fraction']:.3f}. Đây là cơ chế tạo lỗi của mẫu, không khẳng định nguyên nhân duy nhất.")
    lines += ["", "- Chỉ khử exact canonical duplicates, chưa xử lý mọi thư gần trùng/thread/sender; random split có thể lạc quan hơn triển khai thực tế.",
              "- Enron và SpamAssassin là corpus lịch sử; độ chính xác không bảo đảm cho email hiện đại hoặc tiếng Việt.",
              "- Tỷ lệ lớp ở lá chưa hiệu chuẩn. Khác biệt ngoài nguồn có thể do phân phối từ, cách làm sạch và thông tin header; chưa có thí nghiệm nhân quả để tách từng yếu tố.",
              "- Cây nhị phân greedy không nhận zero-gain split; giới hạn đã ghi trong kiểm thử thuật toán.",
              "", "## Hình", "", "![Confusion matrices](figures/confusion-matrices.png)", "", "![Class counts](figures/class-distribution.png)",
              "", "## Tái lập", "", "Chạy `python -m spam_detector.cli train-enron --output runs/<new-name> --raw-dir data/raw`. Nếu Kaggle thay bản latest, dùng raw đã có checksum trong provenance. File summary.json là nguồn số liệu máy đọc."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_enron_training(output: str | Path, raw_dir: str | Path = "data/raw", seed: int = 42,
                       progress: Callable[[str], None] = print) -> dict:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    package_root = Path(__file__).parent
    write_json(root / "run_manifest.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "seed": seed,
        "source_sha256": {str(p.relative_to(package_root)).replace('\\', '/'): sha256(p)
                          for p in sorted(package_root.rglob("*.py"))},
        "selection": {"k": [500, 1000], "max_depth": [5, 10, 15], "min_samples_leaf": [1, 5, 10],
                      "min_samples_split": 10, "objective": "validation_f1_then_precision_then_depth_then_nodes"},
    })
    progress("1/8 Download/verify Kaggle raw CSVs")
    enron = download_dataset("enron", raw_dir)
    external = download_dataset("spamassassin", raw_dir)
    provenance = {s: json.loads((Path(raw_dir) / s / "manifest.json").read_text(encoding="utf-8")) for s in ("enron", "spamassassin")}
    write_json(root / "provenance.json", provenance)
    progress("2/8 Clean, deduplicate, remove cross-source overlap, freeze splits")
    quality = prepare_data(enron, external, root / "processed", seed)
    records = read_records(root / "processed")
    train_rows = [r for r in records if r["split"] == "train"]
    candidates = []
    global_best = None
    for k in (500, 1000):
        progress(f"3/8 Train-only binary/Chi-square K={k}; baseline and 9 configurations")
        feature_dir = root / f"features-k{k}"
        features = build_features(records, feature_dir, k)
        config = ProjectConfig(seed=seed, select_k=k)
        baseline = train_tree(feature_dir / "development.npz", config, root / f"baseline-k{k}", benchmark=True)
        candidate_dir = root / f"tuning-k{k}"
        selected = train_tree(feature_dir / "development.npz", config, candidate_dir, benchmark=True,
                              depths=[5, 10, 15], leaf_sizes=[1, 5, 10])
        selected["k"] = k
        model = load_tree_bundle(candidate_dir / "tree.json").model
        selected["train_metrics"] = classification_metrics(
            [r["label"] for r in train_rows], model.predict(features.transform([r["text"] for r in train_rows])))
        write_json(candidate_dir / "summary.json", selected)
        import csv
        with (candidate_dir / "candidates.validation.csv").open(encoding="utf-8", newline="") as stream:
            candidates.extend({"k": k, **row} for row in csv.DictReader(stream))
        rank = (selected["metrics"]["f1_spam"], selected["metrics"]["precision_spam"], -selected["depth"], -selected["nodes"])
        if global_best is None or rank > global_best[0]:
            global_best = (rank, selected, candidate_dir, feature_dir)
    _, selected, model_dir, feature_dir = global_best
    write_csv(root / "candidates.validation.csv", candidates)
    write_json(root / "selection.lock.json", {"selected": selected, "reason": "validation_f1_then_precision_then_depth_then_nodes", "test_used": False})
    progress("4/8 Fresh-pipeline 5-fold CV on train (baseline + selected if different)")
    cv_candidates = [{"name": "baseline_k500", "k": 500, "parameters": ProjectConfig().tree}]
    if selected["k"] != 500 or selected["parameters"] != cv_candidates[0]["parameters"]:
        cv_candidates.append({"name": "selected", "k": selected["k"], "parameters": selected["parameters"]})
    cv = cross_validate(records, cv_candidates, seed)
    write_json(root / "cv.json", cv)
    write_csv(root / "cv.folds.csv", cv["folds"])
    write_csv(root / "cv.assignments.csv", cv["fold_assignments"])
    progress("5/8 Freeze checksummed text release; verify raw-text round trip")
    release = root / "release"
    release.mkdir()
    shutil.copy2(model_dir / "tree.json", release / "tree.json")
    shutil.copy2(feature_dir / "features.json", release / "features.json")
    manifest = {
        "format_version": 1, "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_fingerprint": quality["dataset_fingerprint"], "selected": selected,
        "checksums": {name: sha256(release / name) for name in ("tree.json", "features.json")},
        "versions": {p: version(p) for p in ("numpy", "scikit-learn", "beautifulsoup4")},
    }
    write_json(release / "release.json", manifest)
    predictor = TextPredictor(release)
    validation_rows = [r for r in records if r["split"] == "validation"]
    loaded_prediction = [r["label"] for r in predictor.predict_texts([r["text"] for r in validation_rows])]
    original = load_tree_bundle(model_dir / "tree.json").model
    original_features = BinaryTextFeatures.load(feature_dir / "features.json")
    np.testing.assert_array_equal(loaded_prediction, original.predict(original_features.transform([r["text"] for r in validation_rows])))
    progress("6/8 Final train/validation/test/external evaluation; no retuning")
    evaluation = {}
    for split in ("train", "validation", "test", "external_test"):
        rows = [r for r in records if r["split"] == split]
        evaluation[split] = evaluate_release(release, rows, root / "evaluation", split)
        if split in ("test", "external_test"):
            matrix_score = evaluate_tree(release / "tree.json", feature_dir / f"{split}.npz", split)
            if matrix_score["metrics"]["f1_spam"] != evaluation[split]["metrics"]["f1_spam"]:
                raise ValueError("Text and matrix evaluation disagree")
            shutil.copy2(root / "evaluation" / f"metrics.{split}.json", release / f"metrics.{split}.json")
    progress("7/8 Export rules, feature names, ensemble handoff and figures")
    write_json(release / "rules.json", original.export_rules(original_features.get_feature_names()))
    write_json(release / "feature_names.json", original_features.get_feature_names())
    handoff = root / "handoff"
    handoff.mkdir()
    shutil.copy2(root / "evaluation" / "predictions.validation.csv", handoff / "tree.validation.csv")
    summary = {
        "data_quality": quality, "selected": selected, "evaluation": evaluation,
        "cv": {k: v for k, v in cv.items() if k != "fold_assignments"},
        "release": "release", "round_trip_verified": True,
        "environment": {p: version(p) for p in ("numpy", "scikit-learn", "beautifulsoup4", "matplotlib")},
    }
    _figures(root, records, evaluation, cv)
    write_json(root / "summary.json", summary)
    progress("8/8 Generate report and finish")
    generate_report(root, summary, records)
    return summary
