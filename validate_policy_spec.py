"""Validate the locked fair O* replay-policy specification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "publication-policy-replay.json"


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    source = ROOT / str(spec["source_config"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != spec["source_config_sha256"]:
        raise SystemExit("policy sidecar source configuration digest mismatch")
    policy = str(spec["policy"])
    required_terms = ("all candidates", "including", "exact/reference", "calibration mean error")
    if any(term not in policy for term in required_terms):
        raise SystemExit("policy sidecar does not declare fair reference-inclusive O* selection")
    if set(spec["selection_exclusions"]) != {
        "validation_error", "test_error", "validation_timing", "test_timing"
    }:
        raise SystemExit("policy sidecar selection exclusions are incomplete")
    if spec["raw_trace_status"] != "immutable; no trace values or timings are changed":
        raise SystemExit("policy sidecar must preserve immutable raw traces")
    print(json.dumps({
        "spec": SPEC_PATH.name,
        "source_config": source.name,
        "source_config_sha256": digest,
        "reference_included": True,
        "raw_traces_immutable": True,
        "pass": True,
    }, indent=2))


if __name__ == "__main__":
    main()
