"""
Download WikiArt from HuggingFace and export to folder structure.
================================================================
从 HuggingFace huggan/wikiart 数据集下载 81,444 张图像,
按 style label 导出为 select_wikiart_styles.py 期望的目录结构:

    output_dir/
      Impressionism/
        artist-name_title.jpg
        ...
      Cubism/
        ...
      ... (27 style folders)

用法:
    # 完整下载 (~36 GB Parquet, 导出后 ~25 GB JPG)
    python download_wikiart.py --output_dir ./wikiart/images

    # 只导出前 1000 张 (测试用)
    python download_wikiart.py --output_dir ./wikiart/images --max_samples 1000

    # Cache and exported images default to repository-local data/ paths.
    python download_wikiart.py

支持断点续传: 中断后重启会自动跳过已导出的 JPG, 只处理剩余图片。

依赖:
    pip install datasets Pillow
"""

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "wikiart" / "images"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".cache" / "huggingface"
DEFAULT_DATASET_ID = "huggan/wikiart"
DEFAULT_REVISION = "d559852d2b232e0fcf195e775866964f0564f2b5"
EXPECTED_FULL_SAMPLES = 81_444
EXPECTED_STYLE_LABELS = 27


def preserve_existing(path):
    """Move an existing metadata file aside instead of deleting it."""
    path = Path(path)
    if not path.exists():
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.name}.previous.{timestamp}")
    suffix = 1
    while target.exists():
        target = path.with_name(f"{path.name}.previous.{timestamp}.{suffix}")
        suffix += 1
    path.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(
        description="Download WikiArt and export it into style-label directories."
    )
    parser.add_argument(
        "--dataset_id", default=DEFAULT_DATASET_ID,
        help=f"Hugging Face dataset id (default: {DEFAULT_DATASET_ID})."
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help=f"Pinned dataset repository revision (default: {DEFAULT_REVISION})."
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split containing the WikiArt images (default: train)."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output root (default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Limit number of images to export (0 = all, default: 0)"
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=DEFAULT_CACHE_DIR,
        help=f"Repository-local Hugging Face cache (default: {DEFAULT_CACHE_DIR})."
    )
    parser.add_argument(
        "--endpoint", default="",
        help="Optional Hugging Face endpoint override; official Hugging Face is used by default."
    )
    parser.add_argument(
        "--expected_samples", type=int, default=EXPECTED_FULL_SAMPLES,
        help=f"Expected full-dataset size (default: {EXPECTED_FULL_SAMPLES}; 0 disables)."
    )
    parser.add_argument(
        "--expected_styles", type=int, default=EXPECTED_STYLE_LABELS,
        help=f"Expected number of style labels (default: {EXPECTED_STYLE_LABELS}; 0 disables)."
    )
    parser.add_argument(
        "--allow_incomplete", action="store_true",
        help="Allow an incomplete/debug source. Never use this flag for benchmark reconstruction."
    )
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    from datasets import load_dataset, load_dataset_builder

    output_root = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_config_path = output_root.parent / "export_source.json"
    source_config = {
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "split": args.split,
    }
    if source_config_path.is_file():
        previous_config = json.loads(source_config_path.read_text(encoding="utf-8"))
        if previous_config != source_config:
            raise RuntimeError(
                f"Existing export source {previous_config} does not match {source_config}. "
                "Use a new --output_dir; datasets from different revisions must not be mixed."
            )
    elif any(output_root.iterdir()):
        raise RuntimeError(
            f"Output directory already contains files but has no export source record: "
            f"{output_root}. Use a new --output_dir."
        )
    else:
        source_config_path.write_text(
            json.dumps(source_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Loading {args.dataset_id}@{args.revision} from Hugging Face...")
    print(f"  Endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"  Cache: {cache_dir}")

    builder = load_dataset_builder(
        args.dataset_id,
        revision=args.revision,
        cache_dir=str(cache_dir),
    )
    features = builder.info.features
    if features is None or "style" not in features or "artist" not in features:
        raise RuntimeError(
            "The selected dataset must expose ClassLabel features named 'style' and 'artist'."
        )
    style_labels = features["style"].names
    artist_labels = features["artist"].names

    if (
        args.expected_styles
        and len(style_labels) != args.expected_styles
        and not args.allow_incomplete
    ):
        raise RuntimeError(
            f"Expected {args.expected_styles} style labels, found {len(style_labels)}. "
            "This source cannot reproduce StyleFamilyBench."
        )

    use_streaming = args.max_samples > 0
    if use_streaming:
        print(f"  Streaming mode: only downloading data needed for {args.max_samples} samples.")
        ds = load_dataset(
            args.dataset_id,
            split=args.split,
            streaming=True,
            revision=args.revision,
            cache_dir=str(cache_dir),
        )
        limit = args.max_samples
        print(f"  {len(style_labels)} styles, {len(artist_labels)} artists.")
        print(f"  Styles: {style_labels}")
        print(f"  Exporting up to {limit} images...\n")
    else:
        print("  Full download: ~31 GB Parquet + ~35 GB Arrow cache.")
        ds = load_dataset(
            args.dataset_id,
            split=args.split,
            revision=args.revision,
            cache_dir=str(cache_dir),
        )
        total = len(ds)
        if (
            args.expected_samples
            and total != args.expected_samples
            and not args.allow_incomplete
        ):
            raise RuntimeError(
                f"Expected {args.expected_samples} images, found {total}. "
                "The dataset revision is not the source used to build StyleFamilyBench."
            )
        limit = total
        print(f"  Dataset loaded: {total} images, {len(style_labels)} styles, {len(artist_labels)} artists.")
        print(f"  Styles: {style_labels}")
        print(f"  Exporting {limit} images...\n")

    # Create style subdirectories
    for style in style_labels:
        (output_root / style).mkdir(exist_ok=True)

    style_counts = {s: 0 for s in style_labels}
    exported = 0
    skipped = 0

    if use_streaming:
        for i, row in enumerate(ds):
            if i >= limit:
                break
            image = row["image"]
            style_idx = row["style"]
            artist_idx = row["artist"]

            if style_idx < 0 or style_idx >= len(style_labels):
                skipped += 1
                continue
            style_name = style_labels[style_idx]

            if 0 <= artist_idx < len(artist_labels):
                artist_name = artist_labels[artist_idx]
            else:
                artist_name = f"unknown-{artist_idx}"

            safe_artist = artist_name.replace(" ", "-").replace("/", "_")
            fname = f"{safe_artist}_{i:06d}.jpg"
            dst = output_root / style_name / fname

            if dst.exists():
                style_counts[style_name] += 1
                exported += 1
                if exported % 500 == 0:
                    print(f"  [{exported}/{limit}] skipped (exists)...")
                continue

            image.convert("RGB").save(str(dst), "JPEG", quality=95)
            style_counts[style_name] += 1
            exported += 1

            if (exported) % 500 == 0:
                print(f"  [{exported}/{limit}] exported...")
    else:
        for i in range(limit):
            row = ds[i]
            image = row["image"]
            style_idx = row["style"]
            artist_idx = row["artist"]

            if style_idx < 0 or style_idx >= len(style_labels):
                skipped += 1
                continue
            style_name = style_labels[style_idx]

            if 0 <= artist_idx < len(artist_labels):
                artist_name = artist_labels[artist_idx]
            else:
                artist_name = f"unknown-{artist_idx}"

            safe_artist = artist_name.replace(" ", "-").replace("/", "_")
            fname = f"{safe_artist}_{i:06d}.jpg"
            dst = output_root / style_name / fname

            if dst.exists():
                style_counts[style_name] += 1
                exported += 1
                if exported % 5000 == 0:
                    print(f"  [{exported}/{limit}] skipped (exists)...")
                continue

            image.convert("RGB").save(str(dst), "JPEG", quality=95)
            style_counts[style_name] += 1
            exported += 1

            if (exported) % 5000 == 0:
                print(f"  [{exported}/{limit}] exported...")

    # Summary
    print(f"\nDone. Exported {exported} images, skipped {skipped}.")
    print(f"Output: {output_root}")
    print(f"\nPer-style counts:")
    for style in style_labels:
        c = style_counts[style]
        if c > 0:
            print(f"  {style:<35} {c:>6}")

    summary_path = output_root.parent / "export_summary.json"
    preserved = preserve_existing(summary_path)
    if preserved is not None:
        print(f"Preserved previous export summary as: {preserved}")
    summary = {
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "split": args.split,
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "max_samples": args.max_samples,
        "exported_or_existing": exported,
        "invalid_rows_skipped": skipped,
        "style_label_count": len(style_labels),
        "style_labels": style_labels,
        "style_counts": style_counts,
        "output_dir": str(output_root),
        "complete_benchmark_source": (
            not use_streaming
            and (not args.expected_samples or exported == args.expected_samples)
            and (not args.expected_styles or len(style_labels) == args.expected_styles)
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Export summary: {summary_path}")


if __name__ == "__main__":
    main()
