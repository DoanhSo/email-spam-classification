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
    args = parser.parse_args(argv)
    try:
        if args.command == "check-config":
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
