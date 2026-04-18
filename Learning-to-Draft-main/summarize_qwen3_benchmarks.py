"""Summarize Qwen3 benchmark JSONL outputs into one CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_DATASETS = ("mt_bench", "gsm8k", "alpaca", "qa")
METHOD_LABELS = {
    "baseline": "Baseline",
    "eagle3": "Eagle3",
    "ltd": "Eagle3+LTD",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Summarize Qwen3 benchmark JSONL outputs into one CSV file."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing method subdirectories with JSONL outputs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--model-label",
        default="Qwen3",
        help="Label to write in the model column.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to summarize.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "eagle3", "ltd"],
        default=["baseline", "eagle3", "ltd"],
        help="Methods to include.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: The JSONL file path.

    Returns:
        Parsed JSON records.
    """
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def first_choice(record: dict[str, Any]) -> dict[str, Any]:
    """Return the first choice entry from a benchmark record.

    Args:
        record: One JSONL record.

    Returns:
        The first choice object.
    """
    return record["choices"][0]


def extract_speed(record: dict[str, Any]) -> float:
    """Compute token throughput for one benchmark record.

    Args:
        record: One JSONL record.

    Returns:
        Tokens per second for the first choice.
    """
    choice = first_choice(record)
    tokens = float(sum(choice.get("new_tokens", [])))
    wall_time = float(sum(choice.get("wall_time", [])))
    prefill_time = float(sum(choice.get("pre_len_times", [])))
    total_time = wall_time + prefill_time
    if total_time <= 0:
        return 0.0
    return tokens / total_time


def extract_accept_lengths(record: dict[str, Any]) -> list[float]:
    """Extract acceptance lengths from one benchmark record.

    Args:
        record: One JSONL record.

    Returns:
        Acceptance lengths for the first choice.
    """
    choice = first_choice(record)
    return [float(value) for value in choice.get("pre_num", [])]


def load_metrics(path: Path) -> tuple[dict[int, float], dict[int, list[float]]]:
    """Load per-question speed and acceptance statistics from a JSONL file.

    Args:
        path: The JSONL file path.

    Returns:
        A tuple of ``question_id -> speed`` and ``question_id -> acceptance lengths``.
    """
    data = read_jsonl(path)
    speeds: dict[int, float] = {}
    accept_lengths: dict[int, list[float]] = {}
    for record in data:
        question_id = int(record["question_id"])
        speeds[question_id] = extract_speed(record)
        accept_lengths[question_id] = extract_accept_lengths(record)
    return speeds, accept_lengths


def dataset_metrics(
    baseline_path: Path,
    method_path: Path,
    method_name: str,
) -> tuple[float, str]:
    """Compute dataset speedup and acceptance length.

    Args:
        baseline_path: Baseline JSONL path.
        method_path: Method JSONL path.
        method_name: Method name.

    Returns:
        A tuple of dataset speedup and acceptance length string.
    """
    baseline_speeds, _ = load_metrics(baseline_path)
    if method_name == "baseline":
        return 1.0, ""

    method_speeds, accept_lengths_by_qid = load_metrics(method_path)
    shared_ids = sorted(set(baseline_speeds) & set(method_speeds))
    if not shared_ids:
        raise ValueError(
            f"No overlapping question ids between {baseline_path} and {method_path}."
        )
    baseline_mean = mean(baseline_speeds[qid] for qid in shared_ids)
    method_mean = mean(method_speeds[qid] for qid in shared_ids)
    speedup = method_mean / baseline_mean if baseline_mean > 0 else 0.0
    accept_lengths = [
        value
        for qid in shared_ids
        for value in accept_lengths_by_qid.get(qid, [])
    ]
    acceptance = mean(accept_lengths) if accept_lengths else None
    acceptance_text = f"{acceptance:.4f}" if acceptance is not None else ""
    return speedup, acceptance_text


def build_header(datasets: list[str]) -> list[str]:
    """Build the CSV header.

    Args:
        datasets: Dataset names.

    Returns:
        The CSV header row.
    """
    header = ["model", "method"]
    for dataset in datasets:
        header.append(f"{dataset}_speedup")
        header.append(f"{dataset}_tau")
    header.extend(["mean_speedup", "mean_tau"])
    return header


def build_row(
    input_root: Path,
    model_label: str,
    method_name: str,
    datasets: list[str],
) -> list[str]:
    """Build one CSV row for one method.

    Args:
        input_root: Root directory of benchmark JSONL files.
        model_label: Model label for the row.
        method_name: Method name.
        datasets: Dataset names.

    Returns:
        The CSV row.
    """
    row = [model_label, METHOD_LABELS.get(method_name, method_name)]
    speedups: list[float] = []
    taus: list[float] = []
    for dataset in datasets:
        baseline_path = input_root / "baseline" / f"{dataset}.jsonl"
        method_path = input_root / method_name / f"{dataset}.jsonl"
        speedup, tau_text = dataset_metrics(baseline_path, method_path, method_name)
        row.append(f"{speedup:.4f}")
        row.append(tau_text)
        speedups.append(speedup)
        if tau_text:
            taus.append(float(tau_text))
    row.append(f"{mean(speedups):.4f}" if speedups else "")
    row.append(f"{mean(taus):.4f}" if taus else "")
    return row


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write the summary CSV file.

    Args:
        path: Output CSV path.
        header: CSV header row.
        rows: CSV data rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    """Summarize benchmark JSONL files into a CSV."""
    args = parse_args()
    header = build_header(args.datasets)
    rows = [
        build_row(args.input_root, args.model_label, method_name, args.datasets)
        for method_name in args.methods
    ]
    write_csv(args.output_csv, header, rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
