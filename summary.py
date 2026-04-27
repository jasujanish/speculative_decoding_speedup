"""Build a flat CSV summary from the local ablation results tree."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


BENCHMARK_BASE_METHODS = ("Baseline", "Eagle3")
CUMULATIVE_TIME_SUMMARY = "cumulative_time_summary.jsonl"
MEAN_COMPARISON_METRICS = [
    ("time", "total_time_seconds"),
    ("speedup", "ablation_mean_speedup"),
    ("drafter calls", "total_drafter_calls"),
    ("target calls", "total_target_calls"),
    ("approximate flops", "total_flops"),
]
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
    parser.add_argument(
        "--output-mean-results",
        type=Path,
        default=Path("mean_results.txt"),
        help="Path to the mean comparison report to write.",
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
    """Read non-empty JSONL records from disk.

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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries.

    Args:
        path: Path to the CSV file.

    Returns:
        CSV rows as dictionaries.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number_text(value: float | int | None) -> str:
    """Format a numeric value for stable CSV output.

    Args:
        value: Numeric value or ``None``.

    Returns:
        Formatted number, or an empty string when missing.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def numeric_value(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return the first present numeric value from a record.

    Args:
        record: Source dictionary.
        keys: Candidate source keys in priority order.

    Returns:
        The first numeric value, or ``None`` when all keys are absent.
    """
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return float(value)
    return None


def add_value(totals: dict[str, float], key: str, value: float | int | None) -> None:
    """Add one numeric value to a totals dictionary.

    Args:
        totals: Mutable totals dictionary.
        key: Output field to update.
        value: Numeric value to add.
    """
    if value is None:
        return
    totals[key] = totals.get(key, 0.0) + float(value)


def add_record_value(
    totals: dict[str, float],
    output_key: str,
    record: dict[str, Any],
    source_keys: tuple[str, ...],
) -> None:
    """Add the first matching source value from a record to a total.

    Args:
        totals: Mutable totals dictionary.
        output_key: Output field to update.
        record: Source dictionary.
        source_keys: Candidate source keys in priority order.
    """
    add_value(totals, output_key, numeric_value(record, source_keys))


def benchmark_rows_by_method(summary_path: Path) -> dict[str, dict[str, str]]:
    """Read benchmark summary rows keyed by method.

    Args:
        summary_path: Path to a benchmark ``summary.csv`` file.

    Returns:
        Mapping from method label to benchmark row.
    """
    return {row["method"]: row for row in read_csv_rows(summary_path)}


def learned_method_name(rows_by_method: dict[str, dict[str, str]]) -> str:
    """Return the learned ablation method from benchmark rows.

    Args:
        rows_by_method: Benchmark rows keyed by method name.

    Returns:
        The method that is not a built-in baseline method.
    """
    for method_name in rows_by_method:
        if method_name not in BENCHMARK_BASE_METHODS:
            return method_name
    return ""


def metric_text(
    rows_by_method: dict[str, dict[str, str]],
    method_name: str,
    metric_name: str,
) -> str:
    """Return one metric from one benchmark method row.

    Args:
        rows_by_method: Benchmark rows keyed by method name.
        method_name: Method row to read.
        metric_name: CSV column name to read.

    Returns:
        Metric text, or an empty string when missing.
    """
    return rows_by_method.get(method_name, {}).get(metric_name, "")


def is_dataset_collection_manifest(path: Path) -> bool:
    """Return whether a manifest describes supervised dataset collection.

    Args:
        path: Path to a ``dataset_manifest.json`` file.

    Returns:
        ``True`` when the parent directory is a supervised dataset directory.
    """
    return path.parent.name.startswith("dataset")


def add_collection_costs(totals: dict[str, float], manifest: dict[str, Any]) -> None:
    """Add supervised dataset collection costs.

    Args:
        totals: Mutable totals dictionary.
        manifest: Dataset collection manifest.
    """
    add_record_value(
        totals,
        "dataset_collection_time_seconds",
        manifest,
        ("total_collection_time_seconds",),
    )
    add_record_value(
        totals,
        "dataset_target_calls",
        manifest,
        ("target_model_calls", "total_target_model_calls"),
    )
    add_record_value(
        totals,
        "dataset_drafter_calls",
        manifest,
        ("draft_model_calls", "total_draft_model_calls", "drafter_model_calls"),
    )
    add_record_value(
        totals,
        "dataset_flops",
        manifest,
        ("collection_total_flops", "total_flops"),
    )


def add_stage_call_costs(totals: dict[str, float], manifest: dict[str, Any]) -> None:
    """Add target and drafter call counts from a stage manifest.

    Args:
        totals: Mutable totals dictionary.
        manifest: LTD stage manifest.
    """
    add_record_value(
        totals,
        "training_target_calls",
        manifest,
        ("training_target_model_calls",),
    )
    add_record_value(
        totals,
        "training_drafter_calls",
        manifest,
        ("training_draft_model_calls", "training_drafter_model_calls"),
    )
    add_record_value(
        totals,
        "validation_target_calls",
        manifest,
        ("validation_target_model_calls",),
    )
    add_record_value(
        totals,
        "validation_drafter_calls",
        manifest,
        ("validation_draft_model_calls", "validation_drafter_model_calls"),
    )


def add_stage_flop_fallbacks(totals: dict[str, float], manifest: dict[str, Any]) -> None:
    """Add stage FLOPs from a manifest when no timing summary is available.

    Args:
        totals: Mutable totals dictionary.
        manifest: LTD stage manifest.
    """
    add_record_value(totals, "training_flops", manifest, ("training_flops",))
    add_record_value(totals, "validation_flops", manifest, ("validation_flops",))


def add_timing_costs(totals: dict[str, float], record: dict[str, Any]) -> None:
    """Add timing, FLOP, and call-count data from one timing record.

    Args:
        totals: Mutable totals dictionary.
        record: ``time_summary.jsonl`` or ``cumulative_time_summary.jsonl`` record.
    """
    add_record_value(totals, "training_time_seconds", record, ("training_time_seconds",))
    add_record_value(totals, "training_flops", record, ("training_flops",))
    add_record_value(
        totals,
        "training_target_calls",
        record,
        ("training_target_model_calls",),
    )
    add_record_value(
        totals,
        "training_drafter_calls",
        record,
        ("training_draft_model_calls", "training_drafter_model_calls"),
    )
    add_record_value(
        totals,
        "validation_time_seconds",
        record,
        ("validation_time_seconds",),
    )
    add_record_value(totals, "validation_flops", record, ("validation_flops",))
    add_record_value(
        totals,
        "validation_target_calls",
        record,
        ("validation_target_model_calls",),
    )
    add_record_value(
        totals,
        "validation_drafter_calls",
        record,
        ("validation_draft_model_calls", "validation_drafter_model_calls"),
    )


def accepted_timing_records(
    path: Path,
    accepted_events: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Read timing records whose event label should be summarized.

    Args:
        path: Path to a timing JSONL file.
        accepted_events: Event labels accepted for this file.

    Returns:
        Accepted timing records.
    """
    return [
        record
        for record in read_jsonl(path)
        if str(record.get("event", "")) in accepted_events
    ]


def add_cumulative_ltd_costs(totals: dict[str, float], cumulative_path: Path) -> bool:
    """Add a cumulative LTD timing summary when it exists.

    Args:
        totals: Mutable totals dictionary.
        cumulative_path: Path to ``cumulative_time_summary.jsonl``.

    Returns:
        ``True`` when at least one cumulative record was applied.
    """
    if not cumulative_path.exists():
        return False
    records = accepted_timing_records(cumulative_path, ("cumulative_summary",))
    for record in records:
        add_timing_costs(totals, record)
    return bool(records)


def add_per_stage_costs(
    totals: dict[str, float],
    stage_manifest_paths: list[Path],
) -> set[Path]:
    """Add per-stage LTD costs from stage manifests and timing records.

    Args:
        totals: Mutable totals dictionary.
        stage_manifest_paths: LTD stage manifest paths.

    Returns:
        Directories that were processed as stage directories.
    """
    stage_dirs: set[Path] = set()
    for manifest_path in stage_manifest_paths:
        stage_dirs.add(manifest_path.parent)
        manifest = read_json(manifest_path)
        add_stage_call_costs(totals, manifest)

        time_path = manifest_path.parent / "time_summary.jsonl"
        records = accepted_timing_records(time_path, ("summary",)) if time_path.exists() else []
        if records:
            for record in records:
                add_timing_costs(totals, record)
        else:
            add_stage_flop_fallbacks(totals, manifest)
    return stage_dirs


def add_unclaimed_timing_costs(
    totals: dict[str, float],
    ablation_dir: Path,
    claimed_dirs: set[Path],
) -> None:
    """Add timing records that are not paired with an LTD stage manifest.

    Args:
        totals: Mutable totals dictionary.
        ablation_dir: Ablation directory to search.
        claimed_dirs: Directories already processed as LTD stages.
    """
    for time_path in sorted(ablation_dir.rglob("time_summary.jsonl")):
        if time_path.parent in claimed_dirs:
            continue
        for record in accepted_timing_records(time_path, ("summary",)):
            add_timing_costs(totals, record)


def finalize_totals(totals: dict[str, float]) -> None:
    """Calculate aggregate total fields from phase fields.

    Args:
        totals: Mutable totals dictionary.
    """
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


def collect_costs(ablation_dir: Path) -> dict[str, float]:
    """Collect all cost metadata for one ablation directory.

    Args:
        ablation_dir: Directory containing benchmark outputs and run artifacts.

    Returns:
        Aggregated cost fields for the final CSV row.
    """
    totals: dict[str, float] = {}
    manifest_paths = sorted(ablation_dir.rglob("dataset_manifest.json"))
    collection_manifest_paths = [
        path for path in manifest_paths if is_dataset_collection_manifest(path)
    ]
    stage_manifest_paths = [
        path for path in manifest_paths if not is_dataset_collection_manifest(path)
    ]

    for manifest_path in collection_manifest_paths:
        add_collection_costs(totals, read_json(manifest_path))

    used_cumulative = add_cumulative_ltd_costs(
        totals,
        ablation_dir / CUMULATIVE_TIME_SUMMARY,
    )
    if used_cumulative:
        claimed_dirs = {path.parent for path in stage_manifest_paths}
    else:
        claimed_dirs = add_per_stage_costs(totals, stage_manifest_paths)

    add_unclaimed_timing_costs(totals, ablation_dir, claimed_dirs)
    finalize_totals(totals)
    return totals


def ablation_identity(results_root: Path, ablation_dir: Path) -> tuple[str, str, str]:
    """Extract model, family, and ablation name from a result path.

    Args:
        results_root: Root results directory.
        ablation_dir: One ablation directory below the results root.

    Returns:
        A ``(model, family, ablation_name)`` tuple.
    """
    relative_parts = ablation_dir.relative_to(results_root).parts
    model = relative_parts[0] if len(relative_parts) > 0 else ""
    family = relative_parts[1] if len(relative_parts) > 1 else ""
    ablation_name = relative_parts[2] if len(relative_parts) > 2 else ""
    return model, family, ablation_name


def build_row(results_root: Path, summary_path: Path) -> dict[str, str]:
    """Build one final CSV row from one benchmark summary.

    Args:
        results_root: Root results directory.
        summary_path: Path to a benchmark ``summary.csv`` file.

    Returns:
        Final summary CSV row.
    """
    ablation_dir = summary_path.parent.parent
    model, family, ablation_name = ablation_identity(results_root, ablation_dir)
    benchmark_rows = benchmark_rows_by_method(summary_path)
    ablation_method = learned_method_name(benchmark_rows)
    costs = collect_costs(ablation_dir)

    row = {
        "model": model,
        "ablation_family": family,
        "ablation_name": ablation_name,
        "ablation_method": ablation_method,
        "ablation_path": str(ablation_dir),
        "baseline_mean_speedup": metric_text(
            benchmark_rows,
            "Baseline",
            "mean_speedup",
        ),
        "baseline_mean_tau": metric_text(benchmark_rows, "Baseline", "mean_tau"),
        "eagle3_mean_speedup": metric_text(benchmark_rows, "Eagle3", "mean_speedup"),
        "eagle3_mean_tau": metric_text(benchmark_rows, "Eagle3", "mean_tau"),
        "ablation_mean_speedup": metric_text(
            benchmark_rows,
            ablation_method,
            "mean_speedup",
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
        Sorted benchmark summary paths.
    """
    return sorted(
        path
        for path in results_root.rglob("summary.csv")
        if path.parent.name.startswith("benchmark_outputs")
    )


def warn_about_incomplete_ltd_rows(rows: list[dict[str, str]]) -> None:
    """Warn when an LTD co-op row appears to lack a cumulative summary.

    Args:
        rows: Final CSV rows.
    """
    for row in rows:
        if row["ablation_family"] != "LTD" or not row["ablation_name"].endswith("_coop"):
            continue
        cumulative_path = Path(row["ablation_path"]) / CUMULATIVE_TIME_SUMMARY
        if not cumulative_path.exists():
            print(
                f"Warning: {row['ablation_path']} has no {CUMULATIVE_TIME_SUMMARY}; "
                "cost fields only cover local stage artifacts.",
                file=sys.stderr,
            )


def row_float(row: dict[str, str], column: str) -> float | None:
    """Read one numeric CSV row value.

    Args:
        row: Final summary row.
        column: Column name to read.

    Returns:
        Parsed float, or ``None`` when missing.
    """
    value = row.get(column, "")
    if value == "":
        return None
    return float(value)


def step_budget_name(ablation_name: str) -> str:
    """Return the step-budget prefix from an ablation name.

    Args:
        ablation_name: Ablation name such as ``10k_coop`` or ``10k_20``.

    Returns:
        The first underscore-separated component.
    """
    return ablation_name.split("_", maxsplit=1)[0]


def index_ltd_rows(
    rows: list[dict[str, str]],
    ltd_suffix: str,
) -> dict[tuple[str, str], dict[str, str]]:
    """Index LTD rows by model and step budget.

    Args:
        rows: Final summary rows.
        ltd_suffix: Required LTD ablation-name suffix, such as ``_coop``.

    Returns:
        Mapping from ``(model, step_budget)`` to LTD row.
    """
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["ablation_family"] != "LTD":
            continue
        if not row["ablation_name"].endswith(ltd_suffix):
            continue
        index[(row["model"], step_budget_name(row["ablation_name"]))] = row
    return index


def mean(values: list[float]) -> float | None:
    """Calculate the arithmetic mean.

    Args:
        values: Values to average.

    Returns:
        Mean value, or ``None`` for an empty list.
    """
    if not values:
        return None
    return sum(values) / len(values)


def percent_change(old_value: float, new_value: float) -> float | None:
    """Calculate percent change from an old value to a new value.

    Args:
        old_value: Baseline value.
        new_value: Comparison value.

    Returns:
        Percent change, or ``None`` when the baseline is zero.
    """
    if old_value == 0:
        return None
    return ((new_value - old_value) / old_value) * 100.0


def change_text(value: float | None) -> str:
    """Format one mean percent change with direction text.

    Args:
        value: Mean percent change.

    Returns:
        Human-readable percent change.
    """
    if value is None:
        return "n/a"
    direction = "increase"
    if value < 0:
        direction = "reduction"
    elif value == 0:
        direction = "no change"
    return f"{abs(value):.2f}% {direction} ({value:+.2f}%)"


def comparison_changes(
    rows: list[dict[str, str]],
    comparison_family: str,
    comparison_suffix: str,
    ltd_suffix: str,
) -> tuple[dict[str, list[float]], list[str], int]:
    """Calculate percent changes from matched LTD rows.

    Args:
        rows: Final summary rows.
        comparison_family: ``SupervisedLearning`` or ``SupervisedLearningTitan``.
        comparison_suffix: Required supervised ablation-name suffix.
        ltd_suffix: Required LTD ablation-name suffix.

    Returns:
        Metric changes, skipped comparison notes, and matched row count.
    """
    ltd_rows = index_ltd_rows(rows, ltd_suffix)
    changes = {metric_name: [] for metric_name, _ in MEAN_COMPARISON_METRICS}
    skipped: list[str] = []
    matched_count = 0

    for row in rows:
        if row["ablation_family"] != comparison_family:
            continue
        if not row["ablation_name"].endswith(comparison_suffix):
            continue
        step_budget = step_budget_name(row["ablation_name"])
        ltd_row = ltd_rows.get((row["model"], step_budget))
        if ltd_row is None:
            skipped.append(
                f"{row['ablation_path']}: no matching LTD {step_budget}{ltd_suffix} row"
            )
            continue
        matched_any_metric = False
        for metric_name, column in MEAN_COMPARISON_METRICS:
            old_value = row_float(ltd_row, column)
            new_value = row_float(row, column)
            if old_value is None or new_value is None:
                skipped.append(
                    f"{row['ablation_path']}: missing {column} for matched comparison"
                )
                continue
            change = percent_change(old_value, new_value)
            if change is None:
                skipped.append(
                    f"{row['ablation_path']}: LTD {column} is zero for matched comparison"
                )
                continue
            changes[metric_name].append(change)
            matched_any_metric = True
        if matched_any_metric:
            matched_count += 1

    return changes, skipped, matched_count


def incomplete_ltd_notes(rows: list[dict[str, str]]) -> list[str]:
    """Return notes about LTD co-op rows without cumulative timing.

    Args:
        rows: Final summary rows.

    Returns:
        Warning notes for mean-results reporting.
    """
    notes: list[str] = []
    for row in rows:
        if row["ablation_family"] != "LTD" or not row["ablation_name"].endswith("_coop"):
            continue
        cumulative_path = Path(row["ablation_path"]) / CUMULATIVE_TIME_SUMMARY
        if not cumulative_path.exists():
            notes.append(
                f"{row['ablation_path']} has no {CUMULATIVE_TIME_SUMMARY}; "
                "cost comparisons using this row only reflect local stage artifacts."
            )
    return notes


def build_mean_results_text(rows: list[dict[str, str]]) -> str:
    """Build the mean comparison report.

    Args:
        rows: Final summary rows.

    Returns:
        Text report for ``mean_results.txt``.
    """
    sections = [
        "Mean Results",
        "",
        "Comparison rule: each 20-epoch SupervisedLearning or "
        "SupervisedLearningTitan row is matched to the same model and "
        "step-budget LTD row.",
        "Percent change formula: (comparison - LTD) / LTD * 100. "
        "Positive means the comparison value is higher than LTD.",
        "",
    ]

    comparisons = [
        ("LTD (co-op) to Base (20 epochs)", "SupervisedLearning", "_20", "_coop"),
        (
            "LTD (no co-op) to Base (20 epochs)",
            "SupervisedLearning",
            "_20",
            "_notcoop",
        ),
        (
            "LTD (co-op) to Titan (20 epochs)",
            "SupervisedLearningTitan",
            "_20",
            "_coop",
        ),
        (
            "LTD (no co-op) to Titan (20 epochs)",
            "SupervisedLearningTitan",
            "_20",
            "_notcoop",
        ),
    ]
    all_skipped: list[str] = []
    for title, family, comparison_suffix, ltd_suffix in comparisons:
        changes, skipped, matched_count = comparison_changes(
            rows,
            family,
            comparison_suffix,
            ltd_suffix,
        )
        all_skipped.extend(skipped)
        sections.append(title)
        sections.append(f"Matched rows: {matched_count}")
        for metric_name, _ in MEAN_COMPARISON_METRICS:
            sections.append(f"{metric_name}: {change_text(mean(changes[metric_name]))}")
        sections.append("")

    notes = incomplete_ltd_notes(rows)
    if notes:
        sections.append("Notes")
        sections.extend(notes)
        sections.append("")

    if all_skipped:
        sections.append("Skipped Comparisons")
        sections.extend(all_skipped)
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def write_mean_results(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Write the mean comparison report.

    Args:
        output_path: Destination text path.
        rows: Final summary rows.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_mean_results_text(rows), encoding="utf-8")


def write_summary_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Write final CSV rows.

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
    rows = [build_row(args.results_root, path) for path in find_summary_paths(args.results_root)]
    rows.sort(
        key=lambda row: (
            row["model"],
            row["ablation_family"],
            row["ablation_name"],
            row["ablation_method"],
        )
    )
    warn_about_incomplete_ltd_rows(rows)
    write_summary_csv(args.output_csv, rows)
    write_mean_results(args.output_mean_results, rows)
    print(f"Wrote {args.output_csv} with {len(rows)} rows.")
    print(f"Wrote {args.output_mean_results}.")


if __name__ == "__main__":
    main()
