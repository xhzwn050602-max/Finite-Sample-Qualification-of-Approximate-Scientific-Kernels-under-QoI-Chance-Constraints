"""Restore losslessly compressed traces and verify all recorded data (stdlib only)."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def checked_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f"Path outside repository: {name}")
    return path

def verify(path, size, sha):
    if path.stat().st_size != size or digest(path) != sha:
        raise ValueError(f"Checksum mismatch; existing file left untouched: {path.name}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Require traces to have already been restored")
    args = parser.parse_args()
    rows = json.loads((ROOT / "DATA_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    for item in rows:
        target = checked_path(item["path"])
        if "archive" in item:
            archive = checked_path(item["archive"])
            verify(archive, item["archive_bytes"], item["archive_sha256"])
            if not target.exists() and not args.verify_only:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=ROOT, prefix=".restore-", suffix=".tmp", delete=False) as out:
                        temp_path = Path(out.name)
                        with gzip.open(archive, "rb") as source:
                            shutil.copyfileobj(source, out, length=1024 * 1024)
                    verify(temp_path, item["bytes"], item["sha256"])
                    # Never overwrite observations created by another process.
                    with target.open("xb") as out, temp_path.open("rb") as source:
                        shutil.copyfileobj(source, out, length=1024 * 1024)
                    print(f"Restored {target.name}", flush=True)
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
        verify(target, item["bytes"], item["sha256"])
    print(f"Verified all {len(rows)} recorded data/configuration files against their original SHA-256 hashes.")

if __name__ == "__main__":
    main()
