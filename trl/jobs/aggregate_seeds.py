#!/usr/bin/env python3
"""Summarize one checkpoint/variant across seeds using sample standard deviation."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--datasets", nargs="+", help="Defaults to the datasets in metrics.json.")
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least two distinct seeds, without duplicates.")

    runs = [json.loads((args.input_dir / f"seed-{seed}/metrics.json").read_text()) for seed in args.seeds]
    datasets = args.datasets or runs[0]["dataset_sizes"]
    suffixes = {"accuracy": "_accuracy", "mean_length": "_completions/mean_length", "aes": "_aes"}
    evaluation = {key: value for key, value in runs[0]["evaluation"].items() if key not in {"seed", "save_rendered_prompts"}}
    if evaluation.get("eval_sampling_strategy") == "independent":
        evaluation.setdefault("bfs_po_k", 0)
    summary = {
        "model": runs[0]["model"],
        "checkpoint_path": runs[0]["checkpoint_path"],
        "evaluation": evaluation,
        "dataset_sizes": {dataset: runs[0]["dataset_sizes"][dataset] for dataset in datasets},
        "seeds": args.seeds,
        "num_seeds": len(runs),
        "std_ddof": 1,
        "metric_suffixes": suffixes,
        "datasets": {},
    }
    rows = []
    print(f"Aggregated {len(runs)} seeds; sample standard deviation (ddof=1).")
    print(f"{'Dataset':<12} {'Metric':<12} {'Mean':>14} {'Std':>14} {'N':>4}")
    for dataset in datasets:
        summary["datasets"][dataset] = {}
        for metric, suffix in suffixes.items():
            values = [float(run["metrics"][f"eval_{dataset}{suffix}"]) for run in runs]
            stats = {"mean": mean(values), "std": stdev(values), "n": len(values)}
            summary["datasets"][dataset][metric] = stats
            rows.append([
                summary["checkpoint_path"], evaluation["presence_penalty"], dataset, metric,
                stats["mean"], stats["std"], stats["n"],
            ])
            print(f"{dataset:<12} {metric:<12} {stats['mean']:>14.6f} {stats['std']:>14.6f} {stats['n']:>4}")

    (args.input_dir / "aggregate_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with (args.input_dir / "aggregate_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint_path", "presence_penalty", "dataset", "metric", "mean", "std", "n"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
