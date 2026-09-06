"""
Review Candidates — HTML Gallery Generator
===========================================
为 select_wikiart_styles.py 输出的候选图像生成 HTML gallery，
方便人工审核标记需要剔除的图像。

工作流:
  1. select_wikiart_styles.py --per_category 50  → 生成 50 张/类候选
  2. python review_candidates.py --input_dir ./candidates_50  → 生成 review.html
  3. 浏览器打开 review.html，勾选需要剔除的图像
  4. 点击"导出"按钮 → 下载 rejected.json
  5. python review_candidates.py --apply-rejected ./candidates_50 --rejected rejected.json --keep 40
     → 剔除标记图像，保留前 40 张到最终目录

人工审核标准（checklist）:
  - 风格标签是否正确（图像风格与所属类别明显不符 → 剔除）
  - 有无水印/大面积遮挡（面积 > 5% → 剔除）
  - 是否为摄影照片而非绘画（WikiArt 偶尔混入 → 剔除）
  - 是否为纯文字/书法作品（无绘画内容 → 剔除）
"""

import json
import argparse
import shutil
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_REJECTION_DIR = SCRIPT_ROOT / "manifests"
EXPECTED_CATEGORIES = [
    "C1_Impressionist",
    "C2_Expressionist",
    "C3_Geometric",
    "C4_Decorative",
    "C5_Realistic",
    "C6_Modern",
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Style Image Review - {category}</title>
<style>
body {{ font-family: system-ui; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
.info {{ background: #fff3cd; padding: 12px; border-radius: 8px; margin-bottom: 20px; }}
.checklist {{ background: #d4edda; padding: 12px; border-radius: 8px; margin-bottom: 20px; }}
.checklist li {{ margin: 4px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
.card {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: relative; }}
.card.rejected {{ opacity: 0.4; border: 3px solid red; }}
.card img {{ width: 100%; height: 280px; object-fit: cover; cursor: pointer; }}
.card .label {{ padding: 8px 12px; font-size: 13px; word-break: break-all; color: #555; }}
.card .reason {{ margin: 0 12px 12px; width: calc(100% - 24px); padding: 6px; }}
.card .reject-btn {{ position: absolute; top: 8px; right: 8px; background: red; color: white;
  border: none; border-radius: 50%; width: 32px; height: 32px; cursor: pointer; font-size: 18px;
  display: flex; align-items: center; justify-content: center; opacity: 0.7; }}
.card .reject-btn:hover {{ opacity: 1; }}
.card.rejected .reject-btn {{ background: green; }}
.toolbar {{ position: sticky; top: 0; background: white; padding: 12px; border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; display: flex; gap: 12px; align-items: center; z-index: 100; }}
.toolbar button {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }}
.toolbar .export {{ background: #007bff; color: white; }}
.toolbar .count {{ font-size: 14px; color: #666; }}
</style>
</head>
<body>
<h1>{category} — Human Review ({total} candidates)</h1>

<div class="checklist">
<strong>Review Checklist (reject if any applies):</strong>
<ul>
<li>Style label incorrect (image style doesn't match category)</li>
<li>Watermark or occlusion > 5% area</li>
<li>Photography (not a painting)</li>
<li>Pure text / calligraphy (no painting content)</li>
</ul>
</div>

<div class="toolbar">
  <button class="export" onclick="exportRejected()">Export rejected.json</button>
  <span class="count">Rejected: <span id="reject-count">0</span> / {total}</span>
  <span class="count">Remaining: <span id="remain-count">{total}</span></span>
</div>

<div class="grid">
{cards}
</div>

<script>
const rejected = new Map();
function toggle(fname, card) {{
  if (rejected.has(fname)) {{
    rejected.delete(fname);
    card.classList.remove('rejected');
  }} else {{
    const reason = card.querySelector('.reason').value;
    if (!reason) {{
      alert('Select a predefined rejection reason first.');
      return;
    }}
    rejected.set(fname, reason);
    card.classList.add('rejected');
  }}
  document.getElementById('reject-count').textContent = rejected.size;
  document.getElementById('remain-count').textContent = {total} - rejected.size;
}}
function exportRejected() {{
  const data = JSON.stringify({{
    category: "{category}",
    rejected: Array.from(rejected, ([filename, reason]) => ({{filename, reason}})),
    remaining: {total} - rejected.size
  }}, null, 2);
  const blob = new Blob([data], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'rejected_{category}.json';
  a.click();
}}
</script>
</body>
</html>"""

CARD_TEMPLATE = """<div class="card" id="card-{idx}">
  <img src="{img_path}" onclick="toggle('{fname}', this.parentElement)">
  <button class="reject-btn" onclick="toggle('{fname}', this.parentElement)">&times;</button>
  <div class="label">{fname}</div>
  <select class="reason">
    <option value="">Select rejection reason</option>
    <option value="incorrect_style_label">Incorrect style label</option>
    <option value="watermark_or_occlusion">Watermark or occlusion &gt; 5%</option>
    <option value="photograph_not_artwork">Photograph rather than artwork</option>
    <option value="text_only_or_unreadable">Text-only, unreadable, or corrupted</option>
  </select>
</div>"""


def generate_gallery(input_dir):
    """为每个类别生成 HTML review gallery"""
    root = Path(input_dir)
    cat_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("C"))
    category_names = [path.name for path in cat_dirs]
    if category_names != EXPECTED_CATEGORIES:
        raise RuntimeError(
            f"Expected the six StyleFamilyBench categories, found: {category_names}"
        )

    for cat_dir in cat_dirs:
        imgs = sorted(
            f for f in cat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

        cards = []
        for idx, img in enumerate(imgs):
            cards.append(CARD_TEMPLATE.format(
                idx=idx,
                img_path=img.name,
                fname=img.name,
            ))

        html = HTML_TEMPLATE.format(
            category=cat_dir.name,
            total=len(imgs),
            cards="\n".join(cards),
        )

        out_path = cat_dir / "review.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"  {cat_dir.name}: {len(imgs)} images -> {out_path}")

    print(f"\nDone. Open each review.html in a browser to review.")
    print("After review, export rejected.json and run:")
    print(f"  python review_candidates.py apply {input_dir} "
          f"--rejected <rejected.json> --keep 40")


def require_new_output_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {path}. Use a new directory to avoid "
            "mixing files from different reviews."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def rejection_entries(data, source_path):
    """Accept both the released filename list and the reason-aware gallery schema."""
    parsed = {}
    for entry in data.get("rejected", []):
        if isinstance(entry, str):
            filename = entry
            reason = "released_manual_review"
        elif isinstance(entry, dict):
            filename = entry.get("filename", "")
            reason = entry.get("reason", "")
        else:
            raise RuntimeError(f"Invalid rejection entry in {source_path}: {entry!r}")
        if not filename or not reason:
            raise RuntimeError(f"Incomplete rejection entry in {source_path}: {entry!r}")
        if filename in parsed:
            raise RuntimeError(f"Duplicate rejection filename in {source_path}: {filename}")
        parsed[filename] = reason
    return parsed


def apply_rejected(input_dir, rejected_files, keep_n, output_dir):
    """根据 rejected.json 剔除图像，保留 keep_n 张到输出目录"""
    root = Path(input_dir)
    out_root = Path(output_dir) if output_dir else root.parent / (root.name + "_final")
    out_root = require_new_output_directory(out_root)

    # 合并所有 rejected files
    all_rejected = {}  # category -> {filename: reason}
    for rfile in rejected_files:
        with open(rfile, "r", encoding="utf-8") as f:
            data = json.load(f)
        cat = data["category"]
        if cat in all_rejected:
            raise RuntimeError(f"More than one rejection manifest was provided for {cat}.")
        all_rejected[cat] = rejection_entries(data, rfile)

    cat_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("C"))
    category_names = [path.name for path in cat_dirs]
    if category_names != EXPECTED_CATEGORIES:
        raise RuntimeError(
            f"Expected the six StyleFamilyBench categories, found: {category_names}"
        )

    review_log = {}

    for cat_dir in cat_dirs:
        cat = cat_dir.name
        if cat not in all_rejected:
            raise RuntimeError(f"No rejection manifest was provided for {cat}.")
        rejected_details = all_rejected[cat]
        rejected_set = set(rejected_details)

        imgs = sorted(
            f for f in cat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

        image_names = {f.name for f in imgs}
        unknown = sorted(rejected_set - image_names)
        if unknown:
            raise RuntimeError(f"{cat} rejection manifest contains unknown files: {unknown}")
        kept = [f for f in imgs if f.name not in rejected_set]
        if len(kept) != keep_n:
            raise RuntimeError(
                f"{cat} has {len(kept)} images after review; exactly {keep_n} are required. "
                "Update the rejection manifest instead of silently truncating the set."
            )
        final = kept

        out_cat = out_root / cat
        out_cat.mkdir(exist_ok=True)
        for f in final:
            shutil.copy2(f, out_cat / f.name)

        review_log[cat] = {
            "total_candidates": len(imgs),
            "rejected": len(rejected_set),
            "kept": len(final),
            "rejected_files": sorted(rejected_set),
            "rejection_reasons": {
                filename: rejected_details[filename] for filename in sorted(rejected_details)
            },
        }

        print(f"  {cat}: {len(imgs)} candidates - {len(rejected_set)} rejected = "
              f"{len(final)} kept")

    log_path = out_root / "review_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(review_log, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Final images: {out_root}")
    print(f"Review log: {log_path}")
    print(f"\nNext step: run rename_to_digits.py --input_dir {out_root} --output_dir <benchmark_dir>")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML gallery for reviewing style image candidates, "
                    "or apply rejection to produce final selection."
    )
    subparsers = parser.add_subparsers(dest="command")

    # gallery command
    gallery_p = subparsers.add_parser("gallery", help="Generate HTML review galleries")
    gallery_p.add_argument("input_dir", type=str, help="Candidates directory from select_wikiart_styles.py")

    # apply command
    apply_p = subparsers.add_parser("apply", help="Apply rejection and produce final selection")
    apply_p.add_argument("input_dir", type=str, help="Candidates directory")
    apply_p.add_argument("--rejected", type=str, nargs="*", default=[],
                         help="One or more rejected_*.json files")
    apply_p.add_argument(
        "--rejected_dir", type=Path, default=DEFAULT_REJECTION_DIR,
        help="Directory containing rejected_*.json (defaults to released manifests)."
    )
    apply_p.add_argument("--keep", type=int, default=40,
                         help="Keep top N images per category after rejection (default: 40)")
    apply_p.add_argument("--output_dir", type=str, default=None,
                         help="Output directory (default: <input_dir>_final)")

    args = parser.parse_args()

    if args.command == "gallery":
        generate_gallery(args.input_dir)
    elif args.command == "apply":
        rejected_files = list(args.rejected)
        if not rejected_files and args.rejected_dir:
            rejected_files.extend(
                str(path) for path in sorted(args.rejected_dir.glob("rejected_*.json"))
            )
        if not rejected_files:
            raise RuntimeError("No rejection manifests were found.")
        apply_rejected(args.input_dir, rejected_files, args.keep, args.output_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Step 1: Generate review galleries")
        print("  python review_candidates.py gallery ./candidates_50")
        print()
        print("  # Step 2: After browser review, apply rejection")
        print("  python review_candidates.py apply ./candidates_50 \\")
        print("      --rejected rejected_C1_Impressionist.json rejected_C2_Expressionist.json \\")
        print("      --keep 40")


if __name__ == "__main__":
    main()
