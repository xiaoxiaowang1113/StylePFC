"""
Rename category-wise style images to digit format (00.png, 01.png, ...).
================================================================
将 select_wikiart_styles.py 筛选出的 6 类 style images 重命名为
与 StylePFC data/sty/ 一致的数字命名格式，方便与其他风格迁移方法对齐。

输入: wikiart_classify/ (筛选后的 6 类文件夹, 含原始文件名的 JPG)
输出: wikiart_benchmark/ (数字命名的 PNG, 保留类别子文件夹)

命名规则:
  - 每类内按原文件名排序, 依次命名为 00.png, 01.png, ..., 39.png
  - 转换为 PNG 格式 (与 StylePFC data/ 一致)
  - 生成 name_mapping.json 记录 {新名: 原名} 的完整映射

用法:
    python rename_to_digits.py \
        --input_dir ./wikiart_classify \
        --output_dir ./wikiart_benchmark
"""

import json
import argparse
import hashlib
from pathlib import Path
from PIL import Image


EXPECTED_CATEGORIES = [
    "C1_Impressionist",
    "C2_Expressionist",
    "C3_Geometric",
    "C4_Decorative",
    "C5_Realistic",
    "C6_Modern",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_pixels(image):
    """Hash decoded RGB pixels so PNG encoder metadata cannot change identity."""
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def require_new_output_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {path}. Use a new directory so old "
            "benchmark files cannot survive a rebuild."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Rename style images to digit format (00.png, 01.png, ...)"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory from select_wikiart_styles.py (e.g. wikiart_classify/)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory with digit-named PNGs (e.g. wikiart_benchmark/)"
    )
    parser.add_argument(
        "--expected_per_category", type=int, default=40,
        help="Required image count in every category (default: 40)."
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print(f"Error: input_dir does not exist: {input_root}")
        return

    # Find category subdirectories (C1_*, C2_*, ...)
    cat_dirs = sorted([
        d for d in input_root.iterdir()
        if d.is_dir() and d.name.startswith("C")
    ])

    if not cat_dirs:
        print(f"Error: no category folders (C1_*, C2_*, ...) found in {input_root}")
        return

    category_names = [path.name for path in cat_dirs]
    if category_names != EXPECTED_CATEGORIES:
        raise RuntimeError(
            "Expected exactly the six StyleFamilyBench directories, found: "
            f"{category_names}"
        )
    output_root = require_new_output_directory(output_root)
    mapping = {}
    benchmark_manifest = {
        "format_version": 1,
        "expected_per_category": args.expected_per_category,
        "categories": {},
    }
    total = 0

    for cat_dir in cat_dirs:
        cat_name = cat_dir.name
        out_cat = output_root / cat_name
        out_cat.mkdir(exist_ok=True)

        # Collect image files, sorted by name for deterministic ordering
        imgs = sorted([
            f for f in cat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        if len(imgs) != args.expected_per_category:
            raise RuntimeError(
                f"{cat_name} contains {len(imgs)} images; "
                f"expected {args.expected_per_category}."
            )

        cat_mapping = {}
        cat_manifest = []
        for idx, img_path in enumerate(imgs):
            new_name = f"{idx:02d}.png"
            dst = out_cat / new_name

            # Load and save as PNG (consistent with StylePFC data/ format)
            with Image.open(img_path) as source_image:
                image = source_image.convert("RGB")
                image.save(str(dst), "PNG")

            cat_mapping[new_name] = img_path.name
            cat_manifest.append(
                {
                    "filename": new_name,
                    "source_filename": img_path.name,
                    "sha256": sha256_file(dst),
                    "pixel_sha256": sha256_pixels(image),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
            )
            total += 1

        mapping[cat_name] = cat_mapping
        benchmark_manifest["categories"][cat_name] = cat_manifest
        print(f"  {cat_name}: {len(imgs)} images -> {out_cat}")

    # Save mapping
    mapping_path = output_root / "name_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    manifest_path = output_root / "benchmark_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {total} images renamed to digit format.")
    print(f"Output: {output_root}")
    print(f"Mapping: {mapping_path}")
    print(f"Checksums: {manifest_path}")


if __name__ == "__main__":
    main()
