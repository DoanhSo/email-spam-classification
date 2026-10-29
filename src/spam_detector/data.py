"""Kaggle acquisition, auditable cleaning and stable stratified email splits."""

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from .preprocessing import canonical_text, preprocess

DATASETS = {
    "enron": ("venky73/spam-mails-dataset", "spam_ham_dataset.csv"),
    "spamassassin": ("nitishabharathi/email-spam-dataset", "completeSpamAssassin.csv"),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download_dataset(source: str, raw_dir: str | Path) -> Path:
    slug, member = DATASETS[source]
    folder = Path(raw_dir) / source
    folder.mkdir(parents=True, exist_ok=True)
    target, manifest_path = folder / member, folder / "manifest.json"
    if target.exists():
        if not manifest_path.exists():
            raise ValueError(f"Existing raw file has no provenance manifest: {target}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["file_sha256"] != sha256(target) or manifest["dataset"] != slug:
            raise ValueError(f"Raw file checksum/source mismatch: {target}")
        return target
    api_url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # Read exactly the selected member, never extract arbitrary archive paths.
        content = archive.read(member)
    manifest = {
        "dataset": slug, "page_url": f"https://www.kaggle.com/datasets/{slug}",
        "download_url": api_url, "file": member,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "license_at_download": "CC0: Public Domain",
        "version_policy": "Kaggle latest at download; immutable local file/ZIP checksums recorded",
    }
    target.write_bytes(content)
    write_json(manifest_path, manifest)
    return target


def clean_csv(path: str | Path, source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source not in DATASETS:
        raise ValueError("Unknown source")
    text_column, label_column = ("text", "label_num") if source == "enron" else ("Body", "Label")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_labels: Counter = Counter()
    empty = 0
    total = 0
    csv.field_size_limit(20_000_000)
    with Path(path).open(newline="", encoding="utf-8-sig", errors="strict") as stream:
        reader = csv.DictReader(stream)
        if not {text_column, label_column} <= set(reader.fieldnames or []):
            raise ValueError(f"Missing required CSV columns for {source}")
        for row in reader:
            total += 1
            value = row[label_column]
            if value not in ("0", "1"):
                raise ValueError(f"Invalid label at {source} row {total}")
            label = int(value)
            raw_labels[label] += 1
            if source == "enron" and row.get("label") != ("spam" if label else "ham"):
                raise ValueError(f"Inconsistent text/numeric label at row {total}")
            text = row[text_column]
            if not text or not preprocess(text):
                empty += 1
                continue
            canonical = canonical_text(text)
            group = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            record = {
                "sample_id": source + "-" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text, "label": label, "source": source, "group_id": group,
            }
            buckets[group].append(record)
    kept = []
    duplicates = conflicts = conflict_groups = 0
    for group in sorted(buckets):
        rows = buckets[group]
        if len({r["label"] for r in rows}) != 1:
            conflicts += len(rows)
            conflict_groups += 1
            continue
        kept.append(min(rows, key=lambda r: r["sample_id"]))
        duplicates += len(rows) - 1
    kept.sort(key=lambda r: r["sample_id"])
    stats = {
        "source": source, "raw_rows": total, "raw_labels": dict(raw_labels),
        "empty_or_no_tokens_removed": empty, "duplicate_rows_removed": duplicates,
        "conflicting_rows_removed": conflicts, "conflicting_groups": conflict_groups,
        "kept_rows": len(kept), "kept_labels": dict(Counter(r["label"] for r in kept)),
        "raw_sha256": sha256(path),
        "accounting_valid": total == empty + duplicates + conflicts + len(kept),
        "dedup_policy": "NFKC + casefold + whitespace normalization; one per group; conflicts removed",
        "limitation": "No fuzzy near-duplicate, thread, sender or chronological split",
    }
    return kept, stats


def assign_splits(records: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda r: r["sample_id"])
    if len({r["group_id"] for r in ordered}) != len(ordered):
        raise ValueError("Deduplicate groups before stratified splitting")
    indices = np.arange(len(ordered))
    labels = np.array([r["label"] for r in ordered])
    train, holdout = train_test_split(indices, test_size=0.4, random_state=seed, stratify=labels)
    validation, test = train_test_split(
        holdout, test_size=0.5, random_state=seed, stratify=labels[holdout],
    )
    names = {int(i): split for split, ids in (("train", train), ("validation", validation), ("test", test)) for i in ids}
    return [{**r, "split": names[i]} for i, r in enumerate(ordered)]


def prepare_data(enron: str | Path, external: str | Path, output: str | Path, seed: int = 42) -> dict[str, Any]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    records, primary_stats = clean_csv(enron, "enron")
    other, external_stats = clean_csv(external, "spamassassin")
    groups = {r["group_id"] for r in records}
    retained = [r for r in other if r["group_id"] not in groups]
    external_stats["overlap_with_enron_removed"] = len(other) - len(retained)
    external_stats["final_rows"] = len(retained)
    external_stats["final_labels"] = dict(Counter(r["label"] for r in retained))
    records = assign_splits(records, seed)
    records.extend({**r, "split": "external_test"} for r in retained)
    write_csv(destination / "records.csv", records)
    fingerprint = sha256(destination / "records.csv")
    write_csv(destination / "splits.csv", [
        {k: r[k] for k in ("sample_id", "group_id", "split")} | {"dataset_fingerprint": fingerprint}
        for r in records
    ])
    counts = {
        split: {"total": sum(r["split"] == split for r in records),
                "ham": sum(r["split"] == split and r["label"] == 0 for r in records),
                "spam": sum(r["split"] == split and r["label"] == 1 for r in records)}
        for split in ("train", "validation", "test", "external_test")
    }
    summary = {"seed": seed, "dataset_fingerprint": fingerprint,
               "enron": primary_stats, "spamassassin": external_stats, "splits": counts}
    write_json(destination / "data_quality.json", summary)
    return summary


def read_records(prepared_dir: str | Path) -> list[dict[str, Any]]:
    csv.field_size_limit(20_000_000)
    root = Path(prepared_dir)
    quality = json.loads((root / "data_quality.json").read_text(encoding="utf-8"))
    if sha256(root / "records.csv") != quality["dataset_fingerprint"]:
        raise ValueError("Prepared records checksum mismatch")
    with (root / "records.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        record["label"] = int(record["label"])
    return records
