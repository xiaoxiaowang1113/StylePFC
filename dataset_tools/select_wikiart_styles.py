"""
WikiArt Category-wise Style Image Selector
==========================================
从 WikiArt 数据集中按 6 大类公平筛选 style images。

v2 更新 (2026-04-03):
  - per_category 默认 50（多选候选，人工审核后保留 40）
  - min_size 提升到 768（crop 到 512 后更清晰）
  - 新增 --max_per_artist 参数（显式控制 artist 上限，默认 3）
  - 新增长短边比过滤（--max_aspect_ratio，默认 3.0）
  - 新增最小文件大小过滤（--min_file_kb，默认 50）
  - 新增灰度图检测（排除纯灰度图）
  - 新增 sub-style 最低覆盖约束（--min_per_substyle，默认 2）
  - scan 改为按 substyle 层级组织，保证子风格覆盖
  - 输出 filter_stats.json 记录各阶段淘汰统计

用法（抽 50 候选，人工审核后保留 40）:
    python select_wikiart_styles.py \
        --wikiart_root /path/to/wikiart/images \
        --output_dir /path/to/candidates_50 \
        --per_category 50 \
        --min_size 768 \
        --max_per_artist 3 \
        --min_per_substyle 2 \
        --resize_to 512 \
        --seed 42

输入: WikiArt 数据集 (huggan/wikiart 格式, images/{style}/{artist_title}.jpg)
输出: output_dir/{category_name}/ 下各 N 张 512×512 图像
      + selection_log.json（完整溯源）
      + filter_stats.json（各阶段淘汰统计）

--resize_to 512 会对每张图执行与 run_stylepfc.py load_img 一致的预处理:
  1. CenterCrop(min(w,h)) → 裁成正方形
  2. Resize(512, 512, LANCZOS) → 缩放到 512×512
这样输出的图像与 StylePFC 评测输入格式一致。
"""

import os
import json
import random
import argparse
import shutil
from pathlib import Path
from collections import defaultdict


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_SELECTION_MANIFEST = SCRIPT_ROOT / "manifests" / "selection_log.json"
EXPECTED_SOURCE_IMAGES = 81_444
EXPECTED_SOURCE_STYLE_FOLDERS = 27
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# ============================================================
# 核心: WikiArt 27 styles → 6 coarse-grained categories
# 学术依据:
#   - WikiArt taxonomy (art-historical standard)
#   - WeART (2026): 152 styles hierarchical taxonomy
#   - StyleShot (2024): per-style-type grouped evaluation
#   - OmniStyle (CVPR 2025): per-category evaluation
# ============================================================

CATEGORY_MAPPING = {
    # C1: 笔触/色彩过渡为主, baseline 友好类
    # 测试压力: 色彩迁移准确性
    "C1_Impressionist": [
        "Impressionism",
        "Post_Impressionism",
        "Pointillism",
        "Fauvism",
    ],
    # C2: 高饱和/强对比/情绪化
    # 测试压力: content preservation under extreme style
    "C2_Expressionist": [
        "Expressionism",
        "Abstract_Expressionism",
        "Color_Field_Painting",
        "Action_painting",
    ],
    # C3: 几何化结构重排
    # 测试压力: 空间保真性
    "C3_Geometric": [
        "Cubism",
        "Analytical_Cubism",
        "Synthetic_Cubism",
        "Art_Nouveau",        # Art_Nouveau_Modern in some versions
        "Art_Nouveau_Modern",
    ],
    # C4: 密集纹理/装饰细节 — 频域模块的核心验证类
    # 测试压力: 高频纹理迁移与结构保持
    "C4_Decorative": [
        "Baroque",
        "Rococo",
        "Mannerism_Late_Renaissance",
        "Symbolism",
    ],
    # C5: 写实/低风格化, 与 C4 形成对照的控制组
    # 测试压力: 不过度风格化
    "C5_Realistic": [
        "Realism",
        "Romanticism",
        "High_Renaissance",
        "Early_Renaissance",
        "Northern_Renaissance",
    ],
    # C6: 扁平/简约/非传统
    # 测试压力: content-style 解耦能力
    "C6_Modern": [
        "Minimalism",
        "Contemporary_Realism",
        "New_Realism",
        "Pop_Art",
        "Naive_Art_Primitivism",
        "Ukiyo_e",
    ],
}

# WikiArt 实际文件夹名可能有变体 (空格/下划线/连字符)
# 这个函数做模糊匹配
def normalize_style_name(name):
    """统一 style 名: 小写, 去空格/连字符/下划线"""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def build_reverse_mapping(category_mapping):
    """构建 {normalized_wikiart_style -> (category_name, original_style)} 的反向映射"""
    reverse = {}
    for cat, styles in category_mapping.items():
        for s in styles:
            reverse[normalize_style_name(s)] = (cat, s)
    return reverse


# ============================================================
# 质量过滤函数
# 所有过滤基于图像本身属性，不涉及任何方法的输出
# ============================================================

def check_image_quality(img_path, min_size, max_aspect_ratio, min_file_kb):
    """
    检查图像是否通过所有自动化质量门槛。

    返回: (passed: bool, reason: str, metadata: dict)
    metadata 包含 width, height, file_kb, is_grayscale 等信息
    """
    meta = {"path": str(img_path)}

    # 1. 文件大小过滤
    file_size_kb = os.path.getsize(img_path) / 1024
    meta["file_kb"] = round(file_size_kb, 1)
    if file_size_kb < min_file_kb:
        return False, f"file_too_small ({file_size_kb:.0f}KB < {min_file_kb}KB)", meta

    try:
        from PIL import Image, ImageStat
        with Image.open(img_path) as img:
            w, h = img.size
            meta["width"] = w
            meta["height"] = h

            # 2. 分辨率过滤: 短边 >= min_size
            if min(w, h) < min_size:
                return False, f"too_small ({w}x{h}, min_edge={min(w,h)} < {min_size})", meta

            # 3. 长短边比过滤
            aspect = max(w, h) / min(w, h)
            meta["aspect_ratio"] = round(aspect, 2)
            if aspect > max_aspect_ratio:
                return False, f"extreme_aspect ({aspect:.1f} > {max_aspect_ratio})", meta

            # 4. 灰度检测
            # RGB 三通道均值极差 < 5 且标准差极差 < 3 → 判定为灰度
            rgb = img.convert("RGB")
            stat = ImageStat.Stat(rgb)
            means = stat.mean  # [R_mean, G_mean, B_mean]
            stddevs = stat.stddev
            mean_range = max(means) - min(means)
            meta["channel_mean_range"] = round(mean_range, 2)
            if mean_range < 5.0:
                stddev_range = max(stddevs) - min(stddevs)
                if stddev_range < 3.0:
                    meta["is_grayscale"] = True
                    return False, "grayscale_image", meta

            meta["is_grayscale"] = False

    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for deterministic image-quality filtering."
        ) from error
    except Exception as e:
        return False, f"read_error ({e})", meta

    return True, "passed", meta


def scan_wikiart(wikiart_root, reverse_map):
    """
    扫描 WikiArt 目录, 返回按 substyle 层级组织的候选池:
    {
        category_name: {
            substyle_name: {
                artist_name: [img_path, ...]
            }
        }
    }
    以及 unmapped_styles
    """
    pool = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    unmapped_styles = set()
    mapped_style_folders = set()

    root = Path(wikiart_root)
    if not root.exists():
        raise FileNotFoundError(f"WikiArt root not found: {wikiart_root}")

    for style_dir in sorted(root.iterdir()):
        if not style_dir.is_dir():
            continue

        style_name = style_dir.name
        norm = normalize_style_name(style_name)

        if norm not in reverse_map:
            unmapped_styles.add(style_name)
            continue

        category, original_style = reverse_map[norm]
        mapped_style_folders.add(style_name)

        for img_file in style_dir.iterdir():
            if img_file.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            # 从文件名解析 artist: "claude-monet_water-lilies.jpg" -> "claude-monet"
            fname = img_file.stem
            parts = fname.split("_", 1)
            artist = parts[0] if len(parts) > 1 else "unknown"

            pool[category][original_style][artist].append(str(img_file))

    return pool, unmapped_styles, mapped_style_folders


def stratified_sample(substyle_pool, n, min_size, max_aspect_ratio,
                      min_file_kb, max_per_artist, min_per_substyle, rng,
                      fallback_min_size=0):
    """
    分层采样: 保证 sub-style 覆盖 + artist 多样性。

    策略:
    1. 对所有候选做质量过滤
    2. 每个 sub-style 先确保 min_per_substyle 张 (round-robin by artist)
    3. 在满足 sub-style 最低覆盖后，按 artist 分层补齐到 n 张
    4. 全程遵守 max_per_artist 限制

    返回: (selected_paths, filter_stats)
    """
    # Phase 0: 质量过滤
    valid_by_substyle = defaultdict(lambda: defaultdict(list))
    filter_stats = {
        "total_scanned": 0,
        "passed_quality": 0,
        "rejected": defaultdict(int),
    }

    for substyle, artist_pool in substyle_pool.items():
        for artist, paths in artist_pool.items():
            for p in paths:
                filter_stats["total_scanned"] += 1
                passed, reason, meta = check_image_quality(
                    p, min_size, max_aspect_ratio, min_file_kb
                )
                if passed:
                    filter_stats["passed_quality"] += 1
                    valid_by_substyle[substyle][artist].append(p)
                else:
                    filter_stats["rejected"][reason] += 1

    # Optional, explicit fallback. The released benchmark does not use a silent
    # threshold change; keeping this opt-in prevents accidental protocol drift.
    total_valid = sum(
        len(ps) for artists in valid_by_substyle.values() for ps in artists.values()
    )
    if total_valid < n and fallback_min_size:
        print(f"    [WARN] Only {total_valid} passed quality filter (need {n}), "
              f"falling back to min_size={fallback_min_size}")
        valid_by_substyle = defaultdict(lambda: defaultdict(list))
        for substyle, artist_pool in substyle_pool.items():
            for artist, paths in artist_pool.items():
                for p in paths:
                    passed, reason, meta = check_image_quality(
                        p, fallback_min_size, max_aspect_ratio, min_file_kb
                    )
                    if passed:
                        valid_by_substyle[substyle][artist].append(p)

    # Phase 1: 确保每个 sub-style 的最低覆盖
    selected = []
    artist_count = defaultdict(int)

    substyles = sorted(valid_by_substyle.keys())
    for substyle in substyles:
        artists_in_substyle = valid_by_substyle[substyle]
        artist_list = sorted(artists_in_substyle.keys())
        rng.shuffle(artist_list)

        count = 0
        for artist in artist_list:
            if count >= min_per_substyle:
                break
            if artist_count[artist] >= max_per_artist:
                continue
            if not artists_in_substyle[artist]:
                continue

            pick = rng.choice(artists_in_substyle[artist])
            selected.append(pick)
            artists_in_substyle[artist].remove(pick)
            artist_count[artist] += 1
            count += 1

    # Phase 2: 用剩余候选补齐到 n（artist-stratified round-robin）
    if len(selected) < n:
        remaining_by_artist = defaultdict(list)
        for substyle, artists in valid_by_substyle.items():
            for artist, paths in artists.items():
                for p in paths:
                    if p not in selected and artist_count[artist] < max_per_artist:
                        remaining_by_artist[artist].append(p)

        artists_with_remaining = sorted(remaining_by_artist.keys())
        rng.shuffle(artists_with_remaining)

        while len(selected) < n and artists_with_remaining:
            next_round = []
            for artist in artists_with_remaining:
                if len(selected) >= n:
                    break
                if remaining_by_artist[artist] and artist_count[artist] < max_per_artist:
                    pick = rng.choice(remaining_by_artist[artist])
                    selected.append(pick)
                    remaining_by_artist[artist].remove(pick)
                    artist_count[artist] += 1
                    if remaining_by_artist[artist] and artist_count[artist] < max_per_artist:
                        next_round.append(artist)
            artists_with_remaining = next_round

    filter_stats["n_selected"] = len(selected)
    filter_stats["n_unique_artists"] = len([a for a, c in artist_count.items() if c > 0])
    filter_stats["rejected"] = dict(filter_stats["rejected"])

    # 统计 sub-style 覆盖
    substyle_counts = defaultdict(int)
    selected_set = set(selected)
    for substyle, artists in substyle_pool.items():
        for artist, paths in artists.items():
            for p in paths:
                if p in selected_set:
                    substyle_counts[substyle] += 1
    filter_stats["substyle_coverage"] = dict(substyle_counts)

    return selected[:n], filter_stats


def center_crop_and_resize(img_path, dst_path, target_size):
    """
    与 run_stylepfc.py load_img 一致的预处理（纯 PIL 实现，无需 torchvision）:
      1. CenterCrop(min(w,h)) → 裁成正方形
      2. Resize(target_size, target_size, LANCZOS) → 缩放到目标尺寸
      3. 保存为 JPG (quality=95)
    """
    from PIL import Image

    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
    image.save(dst_path, "JPEG", quality=95)


def require_new_output_directory(path):
    """Create an output directory, refusing to mix with an earlier run."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {path}. Use a new directory so stale "
            "files cannot change the reconstructed benchmark."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_image_index(wikiart_root, reverse_map):
    """Index mapped source images by filename for manifest-driven reconstruction."""
    index = defaultdict(list)
    mapped_style_folders = set()
    total = 0
    for style_dir in sorted(Path(wikiart_root).iterdir()):
        if not style_dir.is_dir() or normalize_style_name(style_dir.name) not in reverse_map:
            continue
        mapped_style_folders.add(style_dir.name)
        for image_path in sorted(style_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                index[image_path.name].append(image_path)
                total += 1
    return index, total, mapped_style_folders


def validate_source_counts(total_images, mapped_styles, args):
    errors = []
    if args.expected_total_images and total_images != args.expected_total_images:
        errors.append(
            f"expected {args.expected_total_images} mapped images, found {total_images}"
        )
    if args.expected_style_folders and len(mapped_styles) != args.expected_style_folders:
        errors.append(
            f"expected {args.expected_style_folders} mapped style folders, "
            f"found {len(mapped_styles)}"
        )
    if errors and not args.allow_incomplete_source:
        raise RuntimeError(
            "WikiArt source validation failed: " + "; ".join(errors) + ". "
            "Use the released 81,444-image/27-style source. "
            "--allow_incomplete_source is only for smoke tests."
        )
    for error in errors:
        print(f"[WARN] {error}")


def load_released_selection(manifest_path, wikiart_root, reverse_map, args):
    """Resolve the exact released 50 candidates per family from their filenames."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index, total_images, mapped_styles = source_image_index(wikiart_root, reverse_map)
    validate_source_counts(total_images, mapped_styles, args)

    selected_by_category = {}
    for category in CATEGORY_MAPPING:
        category_info = manifest.get("categories", {}).get(category)
        if not category_info:
            raise RuntimeError(f"Selection manifest is missing category: {category}")
        filenames = category_info.get("selected_files", [])
        if len(filenames) != args.per_category:
            raise RuntimeError(
                f"{category} contains {len(filenames)} manifest entries; "
                f"expected {args.per_category}."
            )
        resolved = []
        for filename in filenames:
            matches = index.get(filename, [])
            if len(matches) != 1:
                raise RuntimeError(
                    f"Manifest filename must resolve exactly once: {filename} "
                    f"(found {len(matches)})."
                )
            source_path = matches[0]
            mapped_category, _ = reverse_map[normalize_style_name(source_path.parent.name)]
            if mapped_category != category:
                raise RuntimeError(
                    f"Manifest category mismatch for {filename}: {mapped_category} != {category}."
                )
            resolved.append(str(source_path))
        selected_by_category[category] = resolved
    return manifest, selected_by_category, total_images, mapped_styles


def main():
    parser = argparse.ArgumentParser(
        description="WikiArt Category-wise Style Selector "
                    "(with quality filters, artist caps, sub-style coverage)"
    )
    parser.add_argument("--wikiart_root", type=str, required=True,
                        help="WikiArt images 根目录 (包含 style 子文件夹)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--per_category", type=int, default=50,
                        help="每类选多少张候选 (default: 50, 人工审核后保留 40)")
    parser.add_argument("--min_size", type=int, default=768,
                        help="最短边最小像素 (default: 768, crop 到 512 后更清晰)")
    parser.add_argument("--max_aspect_ratio", type=float, default=3.0,
                        help="最大长短边比 (default: 3.0, 排除极端长条图)")
    parser.add_argument("--min_file_kb", type=float, default=50.0,
                        help="最小文件大小 KB (default: 50, 排除缩略图)")
    parser.add_argument("--max_per_artist", type=int, default=3,
                        help="每个 artist 最多选几张 (default: 3)")
    parser.add_argument("--min_per_substyle", type=int, default=2,
                        help="每个 WikiArt 子风格至少选几张 (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--copy", action="store_true",
                        help="复制文件而非创建 symlink (未启用 --resize_to 时生效)")
    parser.add_argument("--resize_to", type=int, default=512,
                        help="输出图像尺寸, 0 表示不 resize (default: 512). "
                             "使用与 run_stylepfc.py load_img 一致的 CenterCrop+Resize 预处理")
    parser.add_argument(
        "--selection_manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST,
        help="Released 50-candidate manifest used for exact reconstruction."
    )
    parser.add_argument(
        "--resample", action="store_true",
        help="Ignore the released manifest and perform a new stratified sample."
    )
    parser.add_argument(
        "--expected_total_images", type=int, default=EXPECTED_SOURCE_IMAGES,
        help=f"Expected mapped source-image count (default: {EXPECTED_SOURCE_IMAGES}; 0 disables)."
    )
    parser.add_argument(
        "--expected_style_folders", type=int, default=EXPECTED_SOURCE_STYLE_FOLDERS,
        help=f"Expected mapped style-folder count (default: {EXPECTED_SOURCE_STYLE_FOLDERS}; 0 disables)."
    )
    parser.add_argument(
        "--allow_incomplete_source", action="store_true",
        help="Permit a partial source for smoke tests; invalid for benchmark reconstruction."
    )
    parser.add_argument(
        "--fallback_min_size", type=int, default=0,
        help="Optional explicit short-edge fallback; 0 disables threshold fallback."
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. 构建映射
    reverse_map = build_reverse_mapping(CATEGORY_MAPPING)

    # 2. 扫描 WikiArt
    print(f"Scanning WikiArt: {args.wikiart_root}")
    print(f"Quality filters: min_size={args.min_size}px, "
          f"max_aspect={args.max_aspect_ratio}, "
          f"min_file={args.min_file_kb}KB, "
          f"max_per_artist={args.max_per_artist}, "
          f"min_per_substyle={args.min_per_substyle}")
    print()

    released_manifest = None
    selected_from_manifest = None
    if args.resample:
        pool, unmapped, mapped_styles = scan_wikiart(args.wikiart_root, reverse_map)
        source_total = sum(
            len(paths)
            for category_pool in pool.values()
            for artist_pool in category_pool.values()
            for paths in artist_pool.values()
        )
        validate_source_counts(source_total, mapped_styles, args)
    else:
        released_manifest, selected_from_manifest, source_total, mapped_styles = (
            load_released_selection(
                args.selection_manifest, args.wikiart_root, reverse_map, args
            )
        )
        pool = {}
        unmapped = set()
        print(f"Reconstructing released candidates from: {args.selection_manifest}")

    if unmapped:
        print(f"[INFO] 以下 WikiArt styles 未映射到任何大类 (可手动补充):")
        for s in sorted(unmapped):
            print(f"  - {s}")
        print()

    # 3. 每类采样
    output_root = require_new_output_directory(args.output_dir)

    log = {
        "version": "v2",
        "seed": args.seed,
        "per_category": args.per_category,
        "min_size": args.min_size,
        "max_aspect_ratio": args.max_aspect_ratio,
        "min_file_kb": args.min_file_kb,
        "max_per_artist": args.max_per_artist,
        "min_per_substyle": args.min_per_substyle,
        "resize_to": args.resize_to if args.resize_to and args.resize_to > 0 else None,
        "source_image_count": source_total,
        "mapped_style_folder_count": len(mapped_styles),
        "selection_mode": "resampled" if args.resample else "released_manifest",
        "selection_manifest": str(args.selection_manifest) if not args.resample else None,
        "categories": {},
    }

    all_filter_stats = {}

    header = (f"{'Category':<20} {'SubStyles':>9} {'Scanned':>8} "
              f"{'PassQual':>8} {'Selected':>8} {'Artists':>8}")
    print(header)
    print("-" * len(header))

    for cat in CATEGORY_MAPPING.keys():
        substyle_pool = pool.get(cat, {})

        if selected_from_manifest is not None:
            selected = selected_from_manifest[cat]
            released_category = released_manifest["categories"][cat]
            fstats = {
                "total_scanned": source_total,
                "passed_quality": None,
                "rejected": {},
                "n_selected": len(selected),
                "n_unique_artists": released_category.get("n_unique_artists"),
                "substyle_coverage": released_category.get("substyle_coverage", {}),
            }
            n_substyles = released_category.get("n_substyles", 0)
        else:
            selected, fstats = stratified_sample(
                substyle_pool,
                n=args.per_category,
                min_size=args.min_size,
                max_aspect_ratio=args.max_aspect_ratio,
                min_file_kb=args.min_file_kb,
                max_per_artist=args.max_per_artist,
                min_per_substyle=args.min_per_substyle,
                rng=rng,
                fallback_min_size=args.fallback_min_size,
            )
            n_substyles = len(substyle_pool)

        # 输出到子目录
        cat_dir = output_root / cat
        cat_dir.mkdir(exist_ok=True)

        for img_path in selected:
            dst = cat_dir / Path(img_path).name
            if args.resize_to and args.resize_to > 0:
                dst = dst.with_suffix(".jpg")
                center_crop_and_resize(img_path, str(dst), args.resize_to)
            elif args.copy:
                shutil.copy2(img_path, dst)
            else:
                try:
                    dst.symlink_to(img_path)
                except OSError:
                    shutil.copy2(img_path, dst)

        passed_display = "manifest" if fstats["passed_quality"] is None else fstats["passed_quality"]
        print(f"{cat:<20} {n_substyles:>9} {fstats['total_scanned']:>8} "
              f"{str(passed_display):>8} {fstats['n_selected']:>8} "
              f"{fstats['n_unique_artists']:>8}")

        # 记录日志
        log["categories"][cat] = {
            "wikiart_styles": CATEGORY_MAPPING[cat],
            "n_substyles": n_substyles,
            "n_selected": fstats["n_selected"],
            "n_unique_artists": fstats["n_unique_artists"],
            "substyle_coverage": fstats["substyle_coverage"],
            "selected_files": [Path(p).name for p in selected],
        }
        all_filter_stats[cat] = fstats

        if fstats["n_selected"] != args.per_category:
            raise RuntimeError(
                f"{cat}: selected {fstats['n_selected']} images; expected {args.per_category}."
            )

        for substyle, count in fstats["substyle_coverage"].items():
            if count < args.min_per_substyle:
                print(f"  WARNING: {substyle} only has {count} images "
                      f"(< {args.min_per_substyle})")

    # 4. 保存日志
    log_path = output_root / "selection_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    stats_path = output_root / "filter_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_filter_stats, f, indent=2, ensure_ascii=False)

    total = sum(c["n_selected"] for c in log["categories"].values())
    print(f"\nDone. {total} images selected across {len(CATEGORY_MAPPING)} categories.")
    print(f"Selection log: {log_path}")
    print(f"Filter stats:  {stats_path}")
    print(f"\nNext step: run review_candidates.py to generate HTML gallery for manual review.")


if __name__ == "__main__":
    main()
