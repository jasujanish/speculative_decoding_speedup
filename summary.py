"""Build a flat CSV summary from the local ablation results tree."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BENCHMARK_METHODS = ("Baseline", "Eagle3")
CUMULATIVE_TIME_SUMMARY = "cumulative_time_summary.jsonl"
CSV_COLUMNS = [
    "model",
    "ablation_family",
    "ablation_name",
    "ablation_method",
    "ablation_path",
    "baseline_mean_speedup",
    "baseline_mean_tau",
    "eagle3_mean_speedup",
    "eagle3_mean_tau",
    "ablation_mean_speedup",
    "ablation_mean_tau",
    "dataset_collection_time_seconds",
    "dataset_target_calls",
    "dataset_drafter_calls",
    "dataset_flops",
    "training_time_seconds",
    "training_target_calls",
    "training_drafter_calls",
    "training_flops",
    "validation_time_seconds",
    "validation_target_calls",
    "validation_drafter_calls",
    "validation_flops",
    "total_time_seconds",
    "total_target_calls",
    "total_drafter_calls",
    "total_flops",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Summarize ablation results into final_summary.csv."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("final_summary.csv"),
        help="Path to the CSV file to write.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk.

    Args:
        path: Path to the JSONL file.

    Returns:
        Parsed JSON records.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def number_text(value: float | int | None) -> str:
    """Format a numeric value for CSV output.

    Args:
        value: Numeric value or ``None``.

    Returns:
        A stable string representation, or an empty string for missing values.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def add_number(
    totals: dict[str, float],
    key: str,
    value: Any,
) -> None:
    """Add a numeric value to a totals dictionary when present.

    Args:
        totals: Mutable totals dictionary.
        key: Key to update.
        value: Candidate numeric value.
    """
    if value is None or value == "":
        return
    totals[key] = totals.get(key, 0.0) + float(value)


def read_benchmark_rows(summary_path: Path) -> dict[str, dict[str, str]]:
    """Read benchmark rows keyed by method name.

    Args:
        summary_path: Path to a benchmark ``summary.csv`` file.

    Returns:
        Mapping from method label to CSV row.
    """
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def learned_method_name(rows_by_method: dict[str, dict[str, str]]) -> str:
    """Return the ablation method label from a benchmark summary.

    Args:
        rows_by_method: Benchmark rows keyed by method name.

    Returns:
        The method that is not Baseline or Eagle3, or an empty string.
    """
    for method_name in rows_by_method:
        if method_name not in BENCHMARK_METHODS:
            return method_name
    return ""


def metric_text(
    rows_by_method: dict[str, dict[str, str]],
    method_name: str,
    metric_name: str,
) -> str:
    """Return one benchmark metric from a method row.

    Args:
        rows_by_method: Benchmark rows keyed by method name.
        method_name: Method row to read.
        metric_name: Column name to read.

    Returns:
        Metric text, or an empty string when missing.
    """
    return rows_by_method.get(method_name, {}).get(metric_name, "")


def add_collection_manifest(totals: dict[str, float], manifest: dict[str, Any]) -> None:
    """Add supervised dataset collection costs.

    Args:
        totals: Mutable totals dictionary.
        manifest: Parsed ``dataset_manifest.json`` from a dataset directory.
    """
    add_number(
        totals,
        "dataset_collection_time_seconds",
        manifest.get("total_collection_time_seconds"),
    )
    add_number(totals, "dataset_target_calls", manifest.get("target_model_calls"))
    add_number(totals, "dataset_drafter_calls", manifest.get("draft_model_calls"))
    add_number(
        totals,
        "dataset_flops",
        manifest.get("collection_total_flops") or manifest.get("total_flops"),
    )


def add_stage_manifest(totals: dict[str, float], manifest: dict[str, Any]) -> None:
    """Add target and drafter call counts from a training or validation stage.

    Args:
        totals: Mutable totals dictionary.
        manifest: Parsed stage ``dataset_manifest.json``.
    """
    add_number(totals, "training_target_calls", manifest.get("training_target_model_calls"))
    add_number(totals, "training_drafter_calls", manifest.get("training_draft_model_calls"))
    add_number(totals, "validation_target_calls", manifest.get("validation_target_model_calls"))
    add_number(totals, "validation_drafter_calls", manifest.get("validation_draft_model_calls"))


def add_time_summary(totals: dict[str, float], record: dict[str, Any]) -> None:
    """Add time and FLOP data from one time-summary record.

    Args:
        totals: Mutable totals dictionary.
        record: Parsed ``time_summary.jsonl`` record.
    """
    add_number(totals, "training_time_seconds", record.get("training_time_seconds"))
    add_number(totals, "training_flops", record.get("training_flops"))
    add_number(totals, "training_target_calls", record.get("training_target_model_calls"))
    add_number(totals, "training_drafter_calls", record.get("training_draft_model_calls"))
    add_number(totals, "validation_time_seconds", record.get("validation_time_seconds"))
    add_number(totals, "validation_flops", record.get("validation_flops"))
    add_number(totals, "validation_target_calls", record.get("validation_target_model_calls"))
    add_number(totals, "validation_drafter_calls", record.get("validation_draft_model_calls"))


def add_manifest_costs(
    totals: dict[str, float],
    manifest_path: Path,
    use_cumulative_stage_costs: bool,
) -> None:
    """Add cost data from one manifest when it is not superseded.

    Args:
        totals: Mutable totals dictionary.
        manifest_path: Path to a ``dataset_manifest.json`` file.
        use_cumulative_stage_costs: Whether stage-level training and
            validation costs are already covered by a cumulative summary.
    """
    manifest = read_json(manifest_path)
    if manifest_path.parent.name.startswith("dataset"):
        add_collection_manifest(totals, manifest)
    elif not use_cumulative_stage_costs:
        add_stage_manifest(totals, manifest)


def add_time_summary_records(
    totals: dict[str, float],
    time_path: Path,
    accepted_events: tuple[str | None, ...],
) -> None:
    """Add accepted timing records from a JSONL file.

    Args:
        totals: Mutable totals dictionary.
        time_path: Path to a timing JSONL file.
        accepted_events: Event labels that should be included.
    """
    for record in read_jsonl(time_path):
        if record.get("event") in accepted_events:
            add_time_summary(totals, record)


def collect_costs(ablation_dir: Path) -> dict[str, float]:
    """Collect cost metadata for one ablation directory.

    Args:
        ablation_dir: Directory that contains benchmark outputs and run artifacts.

    Returns:
        Aggregated cost fields.
    """
    totals: dict[str, float] = {}
    cumulative_time_path = ablation_dir / CUMULATIVE_TIME_SUMMARY
    use_cumulative_stage_costs = cumulative_time_path.exists()

    for manifest_path in sorted(ablation_dir.rglob("dataset_manifest.json")):
        add_manifest_costs(totals, manifest_path, use_cumulative_stage_costs)

    if use_cumulative_stage_costs:
        add_time_summary_records(totals, cumulative_time_path, ("cumulative_summary",))
    else:
        for time_path in sorted(ablation_dir.rglob("time_summary.jsonl")):
            add_time_summary_records(totals, time_path, (None, "summary"))

    totals["total_time_seconds"] = (
        totals.get("dataset_collection_time_seconds", 0.0)
        + totals.get("training_time_seconds", 0.0)
        + totals.get("validation_time_seconds", 0.0)
    )
    totals["total_target_calls"] = (
        totals.get("dataset_target_calls", 0.0)
        + totals.get("training_target_calls", 0.0)
        + totals.get("validation_target_calls", 0.0)
    )
    totals["total_drafter_calls"] = (
        totals.get("dataset_drafter_calls", 0.0)
        + totals.get("training_drafter_calls", 0.0)
        + totals.get("validation_drafter_calls", 0.0)
    )
    totals["total_flops"] = (
        totals.get("dataset_flops", 0.0)
        + totals.get("training_flops", 0.0)
        + totals.get("validation_flops", 0.0)
    )

    return totals


def ablation_identity(results_root: Path, ablation_dir: Path) -> tuple[str, str, str]:
    """Extract model, family, and ablation name from a result path.

    Args:
        results_root: Root results directory.
        ablation_dir: One ablation directory under the results root.

    Returns:
        A ``(model, family, ablation_name)`` tuple.
    """
    relative_parts = ablation_dir.relative_to(results_root).parts
    model = relative_parts[0] if len(relative_parts) > 0 else ""
    family = relative_parts[1] if len(relative_parts) > 1 else ""
    ablation_name = relative_parts[2] if len(relative_parts) > 2 else ""
    return model, family, ablation_name


def build_row(results_root: Path, summary_path: Path) -> dict[str, str]:
    """Build one final CSV row from a benchmark summary.

    Args:
        results_root: Root results directory.
        summary_path: Path to a benchmark ``summary.csv`` file.

    Returns:
        Final summary CSV row.
    """
    ablation_dir = summary_path.parent.parent
    model, family, ablation_name = ablation_identity(results_root, ablation_dir)
    benchmark_rows = read_benchmark_rows(summary_path)
    ablation_method = learned_method_name(benchmark_rows)
    costs = collect_costs(ablation_dir)

    row = {
        "model": model,
        "ablation_family": family,
        "ablation_name": ablation_name,
        "ablation_method": ablation_method,
        "ablation_path": str(ablation_dir),
        "baseline_mean_speedup": metric_text(
            benchmark_rows, "Baseline", "mean_speedup"
        ),
        "baseline_mean_tau": metric_text(benchmark_rows, "Baseline", "mean_tau"),
        "eagle3_mean_speedup": metric_text(benchmark_rows, "Eagle3", "mean_speedup"),
        "eagle3_mean_tau": metric_text(benchmark_rows, "Eagle3", "mean_tau"),
        "ablation_mean_speedup": metric_text(
            benchmark_rows, ablation_method, "mean_speedup"
        ),
        "ablation_mean_tau": metric_text(benchmark_rows, ablation_method, "mean_tau"),
    }

    for column in CSV_COLUMNS:
        if column not in row:
            row[column] = number_text(costs.get(column))

    return row


def find_summary_paths(results_root: Path) -> list[Path]:
    """Find benchmark summary CSV files under a results root.

    Args:
        results_root: Root results directory.

    Returns:
        Sorted list of benchmark summary paths.
    """
    return sorted(
        path
        for path in results_root.rglob("summary.csv")
        if path.parent.name.startswith("benchmark_outputs")
    )


def write_summary_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Write the final summary CSV.

    Args:
        output_path: Destination CSV path.
        rows: Rows to write.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Summarize all local ablations into one CSV."""
    args = parse_args()
    summary_paths = find_summary_paths(args.results_root)
    rows = [build_row(args.results_root, path) for path in summary_paths]
    rows.sort(
        key=lambda row: (
            row["model"],
            row["ablation_family"],
            row["ablation_name"],
            row["ablation_method"],
        )
    )
    write_summary_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
