"""Commands for Doanh's implemented workstream, not unimplemented team tasks."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .tree_workflow import create_demo_archive, evaluate_tree, train_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-written binary Decision Tree")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-config")
    check.add_argument("--config", default="configs/baseline.yaml")
    for name in ("train-tree", "tune-tree"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/baseline.yaml")
        command.add_argument("--data", required=True, help="NPZ train/validation feature handoff")
        command.add_argument("--output", required=True, help="New run directory")
        command.add_argument("--benchmark", action="store_true")
        if name == "tune-tree":
            command.add_argument("--depths", type=int, nargs="+", default=[5, 10, 15])
            command.add_argument("--leaf-sizes", type=int, nargs="+", default=[1, 5, 10])
    evaluate = sub.add_parser("evaluate-tree")
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--split", choices=["validation", "test", "external_test"], default="test")
    demo = sub.add_parser("demo-tree", help="Synthetic features only; no real email dataset")
    demo.add_argument("--output", required=True, help="New demo directory")
    demo.add_argument("--config", default="configs/baseline.yaml")
    demo.add_argument("--benchmark", action="store_true")
    real = sub.add_parser("train-enron", help="Download, prepare, tune, CV, evaluate and package real email data")
    real.add_argument("--output", required=True)
    real.add_argument("--raw-dir", default="data/raw")
    real.add_argument("--seed", type=int, default=42)
    prepare = sub.add_parser("prepare-enron")
    prepare.add_argument("--enron", required=True)
    prepare.add_argument("--external", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    feature = sub.add_parser("build-tree-features")
    feature.add_argument("--prepared", required=True)
    feature.add_argument("--output", required=True)
    feature.add_argument("--k", type=int, default=500)
    text = sub.add_parser("predict-text")
    text.add_argument("--release", required=True)
    text.add_argument("--text", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train-enron":
            from .config import ProjectConfig
            from .enron_training import run_enron_training
            ProjectConfig(seed=args.seed)
            summary = run_enron_training(args.output, args.raw_dir, args.seed,
                                         progress=lambda message: print(message, file=sys.stderr, flush=True))
            result = {"output": args.output, "selected": summary["selected"],
                      "evaluation": summary["evaluation"], "round_trip_verified": summary["round_trip_verified"]}
        elif args.command == "prepare-enron":
            from .data import prepare_data
            result = prepare_data(args.enron, args.external, args.output, args.seed)
        elif args.command == "build-tree-features":
            from .data import read_records
            from .features import build_features
            fitted = build_features(read_records(args.prepared), args.output, args.k)
            result = {"output": args.output, "effective_k": len(fitted.get_feature_names())}
        elif args.command == "predict-text":
            from .inference import TextPredictor
            result = TextPredictor(args.release).predict_texts([args.text])[0]
        elif args.command == "check-config":
            config = load_config(args.config)
            result = {"status": "valid", "implemented_models": ["tree"], "tree": config.tree}
        elif args.command == "evaluate-tree":
            result = evaluate_tree(args.bundle, args.data, args.split)
        else:
            config = load_config(args.config)
            if args.command == "demo-tree":
                root = Path(args.output)
                root.mkdir(parents=True, exist_ok=False)
                archive = root / "synthetic.npz"
                create_demo_archive(archive)
                result = train_tree(archive, config, root / "run", benchmark=args.benchmark)
                result["dataset_kind"] = "synthetic_only"
            else:
                result = train_tree(
                    args.data, config, args.output, benchmark=args.benchmark,
                    depths=getattr(args, "depths", None), leaf_sizes=getattr(args, "leaf_sizes", None),
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
