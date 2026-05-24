#!/usr/bin/env python3
"""Run a simple hyperparameter sweep and store the best result."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from pathlib import Path


def parse_csv_numbers(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def score_run(learning_rate: float, batch_size: int, epochs: int, seed: int) -> float:
    rng = random.Random(seed + int(learning_rate * 10_000) + batch_size + epochs)
    noise = rng.uniform(-0.01, 0.01)
    baseline = 0.9
    lr_penalty = abs(learning_rate - 0.01) * 3
    batch_penalty = abs(batch_size - 64) / 150
    epoch_gain = min(math.log(max(epochs, 1), 10) * 0.08, 0.12)
    return max(0.0, min(1.0, baseline - lr_penalty - batch_penalty + epoch_gain + noise))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", default="0.001,0.01,0.1")
    parser.add_argument("--batch-sizes", default="32,64,128")
    parser.add_argument("--epochs", default="5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="sweep_results.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    learning_rates = parse_csv_numbers(args.learning_rates, float)
    batch_sizes = parse_csv_numbers(args.batch_sizes, int)
    epochs_list = parse_csv_numbers(args.epochs, int)

    if not learning_rates or not batch_sizes or not epochs_list:
        raise SystemExit("Provide at least one value for learning rates, batch sizes, and epochs.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best = None
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["learning_rate", "batch_size", "epochs", "score"])
        for learning_rate, batch_size, epochs in itertools.product(
            learning_rates, batch_sizes, epochs_list
        ):
            score = score_run(learning_rate, batch_size, epochs, args.seed)
            writer.writerow([learning_rate, batch_size, epochs, f"{score:.4f}"])
            if best is None or score > best[3]:
                best = (learning_rate, batch_size, epochs, score)

    print(f"Wrote sweep results to: {output_path}")
    print(
        "Best config: "
        f"lr={best[0]}, batch_size={best[1]}, epochs={best[2]}, score={best[3]:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
