"""Run the locked publication sweep in independent seed/regime shards.

Each shard invokes the unchanged ``run_sweep.py`` entry point.  Parallelism
only changes scheduling; trial seeds, candidate configurations, random-input
pairing, and timed-kernel boundaries remain those of the locked experiment.
The final merge is deterministic (distribution order, then configured seed
order) and rejects incomplete or mixed-provenance shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    seeds = config.get("seeds")
    distributions = config.get("distributions")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) for s in seeds):
        raise ValueError("configuration must contain integer seeds")
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("configuration must contain distributions")
    return config


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _part_paths(part_dir: Path, distribution: str, seed: int) -> tuple[Path, Path, Path]:
    stem = f"part-{distribution}-{seed}"
    return part_dir / f"{stem}.csv", part_dir / f"{stem}.jsonl", part_dir / f"{stem}.log"


def _part_is_complete(csv_path: Path, trace_path: Path) -> bool:
    """Check the expected one-seed shard shape before allowing a resume."""
    if not csv_path.is_file() or not trace_path.is_file() or csv_path.stat().st_size == 0 or trace_path.stat().st_size == 0:
        return False
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 5:
        return False
    with trace_path.open(encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    # Four kernels (9 configurations each) plus Poisson (13 configurations),
    # each evaluated on 256+256+1024 observations.
    return len(lines) == 1 + (4 * 9 + 13) * (256 + 256 + 1024) and lines[0].startswith('{"metadata":')


def _run_part(root: Path, config: Path, part_dir: Path, distribution: str, seed: int, resume: bool) -> str:
    csv_path, trace_path, log_path = _part_paths(part_dir, distribution, seed)
    if resume and _part_is_complete(csv_path, trace_path):
        return str(csv_path)
    command = [
        sys.executable,
        str(root / "run_sweep.py"),
        "--config",
        str(config),
        "--seeds",
        str(seed),
        "--distribution",
        distribution,
        "--output",
        str(csv_path),
        "--raw-output",
        str(trace_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    return str(csv_path)


def _read_part_csv(path: Path, expected_distribution: str, expected_seed: int) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV shard: {path}")
    for row in rows:
        if row.get("distribution") != expected_distribution or int(row.get("seed", -1)) != expected_seed:
            raise ValueError(f"shard identity mismatch in {path}")
    return list(reader.fieldnames), rows


def _read_part_trace(path: Path, expected_distribution: str, expected_seed: int) -> tuple[dict, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty trace shard: {path}")
    metadata_record = json.loads(lines[0])
    if metadata_record.get("record_type") != "metadata":
        raise ValueError(f"trace shard lacks metadata record: {path}")
    metadata = metadata_record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid metadata record: {path}")
    trace_lines = []
    for line in lines[1:]:
        item = json.loads(line)
        if item.get("distribution") != expected_distribution or int(item.get("seed_group", -1)) != expected_seed:
            raise ValueError(f"trace identity mismatch in {path}")
        trace_lines.append(line)
    if not trace_lines:
        raise ValueError(f"trace shard has no observations: {path}")
    return metadata, trace_lines


def _merge_distribution(
    part_dir: Path,
    distribution: str,
    seeds: Iterable[int],
    output_csv: Path,
    output_trace: Path,
) -> dict:
    seeds = list(seeds)
    all_rows: list[dict] = []
    fieldnames: list[str] | None = None
    metadata: dict | None = None
    trace_lines: list[str] = []
    for seed in seeds:
        part_csv, part_trace, _ = _part_paths(part_dir, distribution, seed)
        names, rows = _read_part_csv(part_csv, distribution, seed)
        if fieldnames is None:
            fieldnames = names
        elif names != fieldnames:
            raise ValueError(f"CSV schema mismatch in {part_csv}")
        all_rows.extend(rows)
        part_metadata, lines = _read_part_trace(part_trace, distribution, seed)
        if metadata is None:
            metadata = part_metadata
        elif part_metadata != metadata:
            raise ValueError(f"metadata mismatch in {part_trace}")
        trace_lines.extend(lines)
    if fieldnames is None or metadata is None:
        raise ValueError(f"no shards for {distribution}")
    expected_rows = len(seeds) * 5
    if len(all_rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows for {distribution}, got {len(all_rows)}")
    keys = {(row["case"], row["distribution"], int(row["seed"])) for row in all_rows}
    if len(keys) != len(all_rows):
        raise ValueError(f"duplicate CSV keys for {distribution}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with output_trace.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "metadata", "metadata": metadata}, sort_keys=True) + "\n")
        handle.write("\n".join(trace_lines) + "\n")
    return {"distribution": distribution, "csv_rows": len(all_rows), "trace_records": len(trace_lines)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="publication-config.json")
    parser.add_argument("--output-dir", default="publication-rerun-20260901")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--keep-parts", action="store_true", help="retain per-seed shard files")
    parser.add_argument("--resume", action="store_true", help="reuse complete shards already present in output-dir")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    root = Path(__file__).resolve().parent
    config = Path(args.config).resolve()
    if not config.is_file():
        parser.error(f"missing config: {config}")
    config_data = _load_config(config)
    seeds = [int(seed) for seed in config_data["seeds"]]
    distributions = [str(item) for item in config_data["distributions"]]
    output_dir = (root / args.output_dir).resolve()
    part_dir = output_dir / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(distribution, seed) for distribution in distributions for seed in seeds]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_part, root, config, part_dir, distribution, seed, args.resume): (distribution, seed)
            for distribution, seed in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - exercised by failed jobs
                failures.append({"distribution": job[0], "seed": job[1], "error": repr(exc)})
    if failures:
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} shard(s) failed; see {output_dir / 'failures.json'}")
    summaries = []
    for distribution in distributions:
        summaries.append(
            _merge_distribution(
                part_dir,
                distribution,
                seeds,
                output_dir / f"publication-{distribution}.csv",
                output_dir / f"publication-{distribution}-traces.jsonl",
            )
        )
    run_record = {
        "config_file": config.name,
        "config_sha256": _digest(config),
        "seeds": seeds,
        "distributions": distributions,
        "workers": args.workers,
        "jobs": len(jobs),
        "summaries": summaries,
        "merge_order": "configured distribution order, then configured seed order",
        "timing_semantics": "each shard invokes unchanged run_sweep.py; scheduling only is parallelized",
    }
    (output_dir / "run-record.json").write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
    if not args.keep_parts:
        for path in part_dir.iterdir():
            path.unlink()
        part_dir.rmdir()
    print(json.dumps(run_record, indent=2))


if __name__ == "__main__":
    main()
