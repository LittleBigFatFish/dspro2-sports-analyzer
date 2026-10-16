"""Inspect local image files without modifying the source data."""

from __future__ import annotations

import csv
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

REPORT_FIELDS = [
    "filename",
    "size_bytes",
    "sha256",
    "duplicate_of",
    "format",
    "width",
    "height",
    "mode",
    "status",
    "error",
]


def calculate_sha256(path: Path) -> str:
    """Calculate a file hash for identifying byte-identical files."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_images(source_directory: Path) -> list[dict[str, object]]:
    """Inspect supported files directly inside a directory."""
    source_directory = source_directory.resolve()

    if not source_directory.is_dir():
        raise NotADirectoryError(
            f"Image directory does not exist: {source_directory}"
        )

    paths = sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file()
    )

    image_paths = [
        path for path in paths
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    for path in paths:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            LOGGER.warning("Unsupported file skipped: %s", path.name)

    if not image_paths:
        raise ValueError(
            f"No supported image files found in: {source_directory}"
        )

    rows: list[dict[str, object]] = []
    first_file_by_hash: dict[str, str] = {}

    for path in image_paths:
        row: dict[str, object] = {
            field: "" for field in REPORT_FIELDS
        }
        row["filename"] = path.name
        row["status"] = "ok"

        try:
            row["size_bytes"] = path.stat().st_size
            file_hash = calculate_sha256(path)
            row["sha256"] = file_hash

            if file_hash in first_file_by_hash:
                row["duplicate_of"] = first_file_by_hash[file_hash]
            else:
                first_file_by_hash[file_hash] = path.name

            # Verify file integrity, then reopen for actual pixel decoding.
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                row["format"] = image.format
                image.load()

                # Report dimensions after applying display orientation.
                oriented_image = ImageOps.exif_transpose(image)
                row["width"], row["height"] = oriented_image.size
                row["mode"] = oriented_image.mode

        except (
            OSError,
            ValueError,
            SyntaxError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as error:
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
            LOGGER.error("Could not inspect %s: %s", path.name, error)

        rows.append(row)

    return rows


def write_report(
    rows: list[dict[str, object]],
    report_path: Path,
) -> None:
    """Write a new CSV report; never overwrite an existing report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    source_directory = PROJECT_ROOT / "data" / "samples"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = (
        PROJECT_ROOT / "outputs" / "data_audit" / timestamp / "inventory.csv"
    )

    try:
        rows = inspect_images(source_directory)
        write_report(rows, report_path)
    except (OSError, ValueError):
        LOGGER.exception("Data audit failed.")
        return 1

    errors = sum(row["status"] == "error" for row in rows)
    duplicates = sum(bool(row["duplicate_of"]) for row in rows)

    LOGGER.info("Inspected images: %d", len(rows))
    LOGGER.info("Unreadable images: %d", errors)
    LOGGER.info("Byte-identical duplicates: %d", duplicates)
    LOGGER.info("Report saved to: %s", report_path)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())