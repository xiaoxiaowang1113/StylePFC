#!/usr/bin/env python
"""Validate a reconstructed StyleFamilyBench without modifying any files."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_RELEASED_MAPPING = SCRIPT_ROOT / "manifests" / "name_mapping.json"
DEFAULT_RELEASED_PIXELS = SCRIPT_ROOT / "manifests" / "stylefamilybench_pixels.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXPECTED_CATEGORIES = [
    "C1_Impressionist",
    "C2_Expressionist",
    "C3_Geometric",
    "C4_Decorative",
    "C5_Realistic",
    "C6_Modern",
]


def image_files(path):
    return sorted(
        item
        for item in Path(path).iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_pixels(image):
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def validate_image(path, expected_size, expected_mode, errors):
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (expected_size, expected_size):
                errors.append(
                    f"{path}: size {image.size} != {(expected_size, expected_size)}"
                )
            if image.mode != expected_mode:
                errors.append(f"{path}: mode {image.mode} != {expected_mode}")
    except Exception as error:
        errors.append(f"{path}: unreadable image ({error})")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate StyleFamilyBench counts, names, images, mapping, and checksums."
    )
    parser.add_argument("--content_dir", type=Path, required=True)
    parser.add_argument("--style_dir", type=Path, required=True)
    parser.add_argument(
        "--released_mapping", type=Path, default=DEFAULT_RELEASED_MAPPING,
        help="Released name_mapping.json used to verify the same 240 references."
    )
    parser.add_argument(
        "--released_pixels", type=Path, default=DEFAULT_RELEASED_PIXELS,
        help="Released decoded-pixel hashes for encoder-independent verification."
    )
    parser.add_argument("--content_count", type=int, default=20)
    parser.add_argument("--styles_per_category", type=int, default=40)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    errors = []

    if not args.content_dir.is_dir():
        raise SystemExit(f"Content directory not found: {args.content_dir}")
    if not args.style_dir.is_dir():
        raise SystemExit(f"Style benchmark directory not found: {args.style_dir}")

    content_files = image_files(args.content_dir)
    expected_content_names = [f"{index:02d}.png" for index in range(args.content_count)]
    if [path.name for path in content_files] != expected_content_names:
        errors.append(
            "Content files must be exactly: " + ", ".join(expected_content_names)
        )
    for path in content_files:
        validate_image(path, args.size, "RGB", errors)

    category_dirs = sorted(
        path.name for path in args.style_dir.iterdir() if path.is_dir()
    )
    if category_dirs != EXPECTED_CATEGORIES:
        errors.append(
            f"Style categories {category_dirs} != expected {EXPECTED_CATEGORIES}"
        )

    released_mapping = load_json(args.released_mapping)
    released_pixels = load_json(args.released_pixels)
    generated_mapping_path = args.style_dir / "name_mapping.json"
    if not generated_mapping_path.is_file():
        errors.append(f"Missing generated mapping: {generated_mapping_path}")
        generated_mapping = {}
    else:
        generated_mapping = load_json(generated_mapping_path)
        if generated_mapping != released_mapping:
            errors.append(
                "Generated name_mapping.json differs from the released 240-image selection."
            )

    checksum_manifest_path = args.style_dir / "benchmark_manifest.json"
    checksum_entries = {}
    if checksum_manifest_path.is_file():
        checksum_manifest = load_json(checksum_manifest_path)
        for category, rows in checksum_manifest.get("categories", {}).items():
            for row in rows:
                checksum_entries[(category, row["filename"])] = row
    else:
        errors.append(f"Missing checksum manifest: {checksum_manifest_path}")

    expected_style_names = [
        f"{index:02d}.png" for index in range(args.styles_per_category)
    ]
    style_total = 0
    for category in EXPECTED_CATEGORIES:
        category_dir = args.style_dir / category
        if not category_dir.is_dir():
            continue
        files = image_files(category_dir)
        style_total += len(files)
        if [path.name for path in files] != expected_style_names:
            errors.append(
                f"{category}: files must be exactly 00.png through "
                f"{args.styles_per_category - 1:02d}.png"
            )
        for path in files:
            validate_image(path, args.size, "RGB", errors)
            with Image.open(path) as source_image:
                rgb_image = source_image.convert("RGB")
                expected_pixel_hash = (
                    released_pixels.get("categories", {})
                    .get(category, {})
                    .get(path.name)
                )
                if expected_pixel_hash is None:
                    errors.append(f"Released pixel manifest missing {category}/{path.name}")
                elif sha256_pixels(rgb_image) != expected_pixel_hash:
                    errors.append(f"Released pixel mismatch: {category}/{path.name}")
            entry = checksum_entries.get((category, path.name))
            if entry is None:
                errors.append(f"Checksum manifest missing {category}/{path.name}")
            elif sha256_file(path) != entry.get("sha256"):
                errors.append(f"Checksum mismatch: {category}/{path.name}")

    if errors:
        print("StyleFamilyBench validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    pair_count = len(content_files) * style_total
    print("StyleFamilyBench validation passed.")
    print(f"  content images: {len(content_files)}")
    print(f"  style images:   {style_total}")
    print(f"  categories:     {len(EXPECTED_CATEGORIES)}")
    print(f"  total pairs:    {pair_count}")


if __name__ == "__main__":
    main()
