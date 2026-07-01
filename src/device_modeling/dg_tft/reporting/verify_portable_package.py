from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from build_release_manifest import EXCLUDE_PARTS, build_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a clean portable release against PACKAGE_MANIFEST.json."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--strict-clean",
        action="store_true",
        help="Also fail if generated environments/caches are present.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "validation" / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if manifest.get("release") != root.name:
        errors.append(
            f"manifest release={manifest.get('release')!r} != folder={root.name!r}"
        )
    expected = {item["path"]: item for item in manifest.get("files", [])}
    actual_manifest = build_manifest(root)
    actual = {item["path"]: item for item in actual_manifest["files"]}
    if set(expected) != set(actual):
        errors.append(
            f"file set differs: missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )
    for relative, item in expected.items():
        path = root / relative
        if not path.is_file():
            continue
        if path.stat().st_size != int(item["bytes"]):
            errors.append(f"size mismatch: {relative}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"sha256 mismatch: {relative}")

    if args.strict_clean:
        forbidden = sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file()
            and any(
                part in EXCLUDE_PARTS for part in path.relative_to(root).parts
            )
        )
        if forbidden:
            errors.append(f"forbidden runtime/cache files present: {forbidden[:20]}")

    report = {
        "status": "ok" if not errors else "failed",
        "root": str(root),
        "manifest_files": len(expected),
        "manifest_bytes": sum(int(item["bytes"]) for item in expected.values()),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
