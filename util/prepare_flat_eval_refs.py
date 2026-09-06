#!/usr/bin/env python
"""Prepare fixed-size flat evaluation references for StylePFC."""

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MANIFEST_NAME = ".prepare_flat_eval_refs.json"


def image_files(path):
    return sorted(
        p
        for p in Path(path).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )


def ensure_unique_stems(paths, label):
    stems = [path.stem for path in paths]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise SystemExit("{} has duplicate image stems: {}".format(label, ", ".join(duplicates[:10])))


def pair_names(content_files, style_files):
    return [
        "{}_stylized_{}.png".format(content_path.stem, style_path.stem)
        for content_path in content_files
        for style_path in style_files
    ]


def open_rgb(path):
    with Image.open(path) as image:
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")


def resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def save_eval_image(src_path, dst_path, size):
    image = open_rgb(src_path)
    image = image.resize((size, size), resample_filter())
    image.save(dst_path, format="PNG")


def valid_eval_dir(path, names, size):
    path = Path(path)
    if not path.is_dir():
        return False

    files = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )
    if [p.name for p in files] != names:
        return False

    manifest_path = path / MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            manifest.get("size") == size
            and manifest.get("count") == len(names)
            and manifest.get("filenames") == names
        )

    for image_path in files:
        try:
            with Image.open(image_path) as image:
                if image.mode != "RGB" or image.size != (size, size):
                    return False
                image.verify()
        except Exception:
            return False
    return True


def replace_dir(src, dst):
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(src, dst)


def write_manifest(path, names, size):
    manifest = {
        "size": size,
        "count": len(names),
        "filenames": names,
    }
    (Path(path) / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare fixed-size flat eval references.")
    parser.add_argument("--cnt", required=True, type=Path, help="Content image directory.")
    parser.add_argument("--sty", required=True, type=Path, help="Style image directory.")
    parser.add_argument("--cnt-eval", required=True, type=Path, help="Output content eval directory.")
    parser.add_argument("--sty-eval", required=True, type=Path, help="Output style eval directory.")
    parser.add_argument("--size", default=512, type=int, help="Saved eval image size.")
    parser.add_argument("--expected-count", default=0, type=int, help="Expected pair count; 0 disables.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if eval dirs are already valid.")
    return parser.parse_args()


def main():
    args = parse_args()
    content_files = image_files(args.cnt)
    style_files = image_files(args.sty)
    ensure_unique_stems(content_files, "content directory")
    ensure_unique_stems(style_files, "style directory")

    expected = len(content_files) * len(style_files)
    if expected == 0:
        raise SystemExit("No content/style pairs found.")
    if args.expected_count and expected != args.expected_count:
        raise SystemExit("Expected pair count mismatch: {} vs {}".format(expected, args.expected_count))

    names = pair_names(content_files, style_files)
    if (
        not args.force
        and valid_eval_dir(args.cnt_eval, names, args.size)
        and valid_eval_dir(args.sty_eval, names, args.size)
    ):
        print("Reusing prepared eval refs: {} and {}".format(args.cnt_eval, args.sty_eval))
        return

    args.cnt_eval.parent.mkdir(parents=True, exist_ok=True)
    args.sty_eval.parent.mkdir(parents=True, exist_ok=True)
    tmp_cnt = args.cnt_eval.parent / ".{}.tmp.{}".format(args.cnt_eval.name, os.getpid())
    tmp_sty = args.sty_eval.parent / ".{}.tmp.{}".format(args.sty_eval.name, os.getpid())
    if tmp_cnt.exists():
        shutil.rmtree(tmp_cnt)
    if tmp_sty.exists():
        shutil.rmtree(tmp_sty)
    tmp_cnt.mkdir(parents=True)
    tmp_sty.mkdir(parents=True)

    try:
        for content_path in content_files:
            for style_path in style_files:
                name = "{}_stylized_{}.png".format(content_path.stem, style_path.stem)
                save_eval_image(content_path, tmp_cnt / name, args.size)
                save_eval_image(style_path, tmp_sty / name, args.size)
        write_manifest(tmp_cnt, names, args.size)
        write_manifest(tmp_sty, names, args.size)
        replace_dir(tmp_cnt, args.cnt_eval)
        replace_dir(tmp_sty, args.sty_eval)
    except Exception:
        if tmp_cnt.exists():
            shutil.rmtree(tmp_cnt)
        if tmp_sty.exists():
            shutil.rmtree(tmp_sty)
        raise

    print("Prepared eval refs: pairs={}, size={}x{}".format(expected, args.size, args.size))
    print("content eval: {}".format(args.cnt_eval))
    print("style eval:   {}".format(args.sty_eval))


if __name__ == "__main__":
    main()
