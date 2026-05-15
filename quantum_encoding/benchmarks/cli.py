from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import BenchmarkConfig, BenchmarkRunner, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmarks for the quantum encoding implementation.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reservoir-dim", type=int, default=64)
    args = parser.parse_args()

    default_config = BenchmarkConfig()
    config = BenchmarkConfig(
        datasets=tuple(args.datasets) if args.datasets else default_config.datasets,
        seeds=tuple(args.seeds) if args.seeds else default_config.seeds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        reservoir_dim=args.reservoir_dim,
        output_dir=args.output_dir,
    )
    results = BenchmarkRunner(config).run()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(summarize_results(results), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} benchmark rows to {output_dir}")
