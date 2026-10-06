"""Regenerate every cited figure from the canonical archived observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-only", action="store_true")
    args = parser.parse_args()
    regimes = ("light", "heavy", "correlated")
    commands = [
        ["audit_manuscript_evidence.py"],
        ["plot_high_level_figures.py"],
        ["plot_results.py", "--summary", *[f"publication-{d}-summary.json" for d in regimes],
         "--poisson-traces", *[f"publication-{d}-traces.jsonl" for d in regimes],
         "--output-dir", "figures-publication"],
        ["plot_heat_audit.py"], ["plot_production_heat_audit.py"],
        ["plot_scaling_audit.py"], ["plot_stability_audit.py"],
        ["plot_sensitivity_audit.py"], ["plot_call_count_audit.py"],
    ]
    if not args.provenance_only:
        for command in commands:
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
    inputs = [*[f"publication-{d}-{suffix}" for d in regimes
                for suffix in ("summary.json", "traces.jsonl")],
              "confirmatory-mc.csv", "confirmatory-mc-config.json", "confirmatory-mc-summary.json",
              "heat-experiment-config-full.json", "heat-audit-full-fair.csv", "heat-audit-full.jsonl",
              "heat-production-config.json", "heat-production-audit.jsonl", "heat-production-audit-summary.json",
              "scaling-audit.json", "stability-audit.csv", "sensitivity-audit-delivery.json",
              "call-count-audit-fair.json"]
    figures = json.loads((ROOT / "FIGURES.json").read_text(encoding="utf-8"))
    files = sorted(set(inputs + figures + [f.replace(".pdf", ".png") for f in figures]
                       + [command[0] for command in commands]))
    provenance = {"commands": commands, "figure_count": len(figures),
                  "files": {name: digest(ROOT / name) for name in files},
                  "notes": ["Existing observations only; no new measurements.",
                            "Sensitivity fallback denotes an empty full accepted set.",
                            "Broad cost figure: top mean paired ratios; bottom ratios of mean costs.",
                            "Legacy small heat costs are descriptive because group test seeds overlap."]}
    (ROOT / "figures-publication" / "delivery-figure-provenance.json").write_text(
        json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(f"Recorded provenance for {len(figures)} manuscript figures")


if __name__ == "__main__":
    main()
