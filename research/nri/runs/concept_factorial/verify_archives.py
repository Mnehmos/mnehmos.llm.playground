#!/usr/bin/env python3
"""Verify downloaded concept-factorial checkpoint archives against SHA256SUMS (standard library only).

Usage (any OS with Python 3.8+):
    python verify_archives.py <folder-with-the-.tar.xz-files> [path/to/SHA256SUMS]

Prints OK / MISMATCH / MISSING per archive and a summary; exit code 0 only if all 96 are present and match.
Optional deep check of one archive's contents against its per-file manifest:
    python verify_archives.py --contents <archive.tar.xz> <path/to/<run>/manifest.json>
"""
import hashlib
import json
import sys
import tarfile
from pathlib import Path


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def verify_folder(folder, sums):
    folder = Path(folder)
    expected = [l.split() for l in Path(sums).read_text().splitlines() if l.strip()]
    ok = bad = missing = 0
    for digest, name in expected:
        p = folder / name
        if not p.exists():
            print(f"MISSING   {name}")
            missing += 1
        elif sha256(p) == digest:
            print(f"OK        {name}")
            ok += 1
        else:
            print(f"MISMATCH  {name}")
            bad += 1
    print(f"\n{ok} OK, {bad} mismatched, {missing} missing (of {len(expected)})")
    return ok == len(expected)


def verify_contents(archive, manifest):
    files = json.loads(Path(manifest).read_text())["files"]
    seen = bad = 0
    with tarfile.open(archive, "r:xz") as t:
        for m in t:
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1]
            h = hashlib.sha256(t.extractfile(m).read()).hexdigest()
            seen += 1
            if files.get(rel) != h:
                bad += 1
                print("MISMATCH", rel)
    print(f"{seen} files checked, {bad} mismatched, manifest lists {len(files)}")
    return bad == 0 and seen == len(files)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--contents":
        sys.exit(0 if verify_contents(sys.argv[2], sys.argv[3]) else 1)
    here = Path(__file__).resolve().parent
    sums = sys.argv[2] if len(sys.argv) > 2 else here / "SHA256SUMS"
    sys.exit(0 if verify_folder(sys.argv[1], sums) else 1)
