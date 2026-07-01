from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from build_release_manifest import should_include, write_manifest


SOURCE_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_portable_text(path: Path) -> None:
    """Remove workstation-specific paths from text and JSON evidence."""

    text = path.read_text(encoding="utf-8", errors="replace")
    for sibling in SOURCE_ROOT.parent.iterdir():
        if sibling.is_dir() and sibling.name.lower().startswith("dg_tft_v"):
            for spelling in (str(sibling), str(sibling).replace("\\", "/")):
                text = text.replace(spelling, "<release_root>")
    text = text.replace(
        r"E:\synopsys\Hspice_S-2021.09\WIN64\hspice.com",
        "<configured_hspice_executable>",
    )
    path.write_text(text, encoding="utf-8")


def safe_destination(name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Archive name must be one plain directory name.")
    destination = (SOURCE_ROOT.parent / name).resolve()
    if destination.parent != SOURCE_ROOT.parent.resolve():
        raise RuntimeError(f"Destination escapes the workspace parent: {destination}")
    if destination == SOURCE_ROOT.resolve():
        raise RuntimeError("Destination must differ from the working release.")
    return destination


def build(destination: Path, *, replace: bool) -> tuple[Path, dict]:
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        if not replace:
            raise FileExistsError(
                f"Destination already exists: {destination} or {archive}. "
                "Choose another --name or pass --replace."
            )
        if destination.exists():
            shutil.rmtree(destination)
        if archive.exists():
            archive.unlink()
    destination.mkdir(parents=True)

    copied = 0
    for source in sorted(SOURCE_ROOT.rglob("*")):
        if not should_include(source, SOURCE_ROOT):
            continue
        relative = source.relative_to(SOURCE_ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix.lower() in {".json", ".log"}:
            sanitize_portable_text(target)
        copied += 1

    manifest_path = write_manifest(
        destination, destination / "validation" / "PACKAGE_MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["file_count"] != copied:
        raise RuntimeError(
            f"Copied {copied} files but manifest contains {manifest['file_count']}."
        )

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as handle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                handle.write(path, Path(destination.name) / path.relative_to(destination))

    with zipfile.ZipFile(archive, "r") as handle:
        bad_member = handle.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC validation failed at {bad_member}")

    summary = {
        "release": destination.name,
        "directory": str(destination),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": file_sha256(archive),
        "manifest_files": manifest["file_count"],
        "manifest_bytes": manifest["total_bytes"],
        "zip_crc_pass": True,
    }
    (destination.parent / f"{destination.name}_SHA256.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return archive, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean portable DG-TFT release directory and ZIP."
    )
    parser.add_argument(
        "--name",
        default=f"{SOURCE_ROOT.name}_PORTABLE_COPY",
        help="Sibling directory/ZIP basename.",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    destination = safe_destination(args.name)
    _, summary = build(destination, replace=args.replace)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
