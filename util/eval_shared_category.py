#!/usr/bin/env python
"""Evaluate category outputs with reusable prepared style eval folders."""

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
METRICS = ["artfid", "fid", "lpips", "lpips_gray", "cfsd", "histo_loss"]
CSV_HEADER = [
    "method",
    "group",
    "category",
    "content_dir",
    "style_dir",
    "output_dir",
    "image_count",
    "metric_name",
    "metric_value",
    "command",
    "timestamp",
    "status",
    "error",
    "log_path",
]
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def image_files(path):
    if not path.is_dir():
        return []
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith(".")
        and ".tmp." not in p.name
        and not p.name.endswith(".tmp.png")
    )


def parse_categories(sty_eval_root, categories_text):
    if categories_text:
        return [item.strip() for item in categories_text.split(",") if item.strip()]
    return sorted(path.name for path in sty_eval_root.iterdir() if path.is_dir())


def command_text(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(cmd, cwd, log_path):
    output_lines = []
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n$ {}\n".format(command_text(cmd)))
        log.flush()
        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            output_lines.append(line)
        return_code = process.wait()
        log.write("\n[exit code] {}\n".format(return_code))
    return return_code, "".join(output_lines)


def extract_metric(pattern, text):
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def parse_artfid_output(text):
    return {
        "artfid": extract_metric(r"ArtFID:\s*({})".format(FLOAT_PATTERN), text),
        "fid": extract_metric(r"(?<![A-Za-z])FID:\s*({})".format(FLOAT_PATTERN), text),
        "lpips": extract_metric(r"LPIPS:\s*({})".format(FLOAT_PATTERN), text),
        "lpips_gray": extract_metric(r"LPIPS_gray:\s*({})".format(FLOAT_PATTERN), text),
        "cfsd": extract_metric(r"CFSD:\s*({})".format(FLOAT_PATTERN), text),
    }


def parse_histogan_output(text):
    return {"histo_loss": extract_metric(r"color matching loss:\s*({})".format(FLOAT_PATTERN), text)}


def write_csv_header(path, append):
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CSV_HEADER).writeheader()


def append_metric_rows(path, base_row, metrics):
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        for metric_name in METRICS:
            row = dict(base_row)
            row["metric_name"] = metric_name
            row["metric_value"] = metrics.get(metric_name, "")
            writer.writerow(row)


def validate_category(args, category, output_dir, content_dir, style_dir):
    output_images = image_files(output_dir)
    content_images = image_files(content_dir)
    style_images = image_files(style_dir)
    if not output_dir.is_dir():
        return False, "output directory not found", 0
    if not content_dir.is_dir():
        return False, "content eval directory not found", len(output_images)
    if not style_dir.is_dir():
        return False, "shared style eval directory not found", len(output_images)
    if len(output_images) != len(content_images):
        return False, "output count {} != content eval count {}".format(len(output_images), len(content_images)), len(output_images)
    if len(output_images) != len(style_images):
        return False, "output count {} != style eval count {}".format(len(output_images), len(style_images)), len(output_images)
    if not args.allow_partial and len(output_images) != args.expected_count:
        return False, "output count {} != expected {}".format(len(output_images), args.expected_count), len(output_images)

    output_stems = [path.stem for path in output_images]
    content_stems = [path.stem for path in content_images]
    style_stems = [path.stem for path in style_images]
    if output_stems != content_stems:
        missing = sorted(set(content_stems) - set(output_stems))[:5]
        extra = sorted(set(output_stems) - set(content_stems))[:5]
        return False, "output/content filenames are not aligned; missing={}, extra={}".format(missing, extra), len(output_images)
    if output_stems != style_stems:
        missing = sorted(set(style_stems) - set(output_stems))[:5]
        extra = sorted(set(output_stems) - set(style_stems))[:5]
        return False, "output/style filenames are not aligned; missing={}, extra={}".format(missing, extra), len(output_images)
    return True, "", len(output_images)


def evaluate_category(args, category):
    output_dir = args.output_root / category
    style_dir = args.sty_eval_root / category
    result_dir = args.result_root / category
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "eval.log"
    timestamp = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n===== eval {} / {} {} =====\n".format(args.method, category, timestamp))

    ok, error, image_count = validate_category(args, category, output_dir, args.cnt, style_dir)
    artfid_cmd = [
        sys.executable,
        "eval_artfid.py",
        "--sty",
        style_dir,
        "--cnt",
        args.cnt,
        "--tar",
        output_dir,
        "--mode",
        args.mode,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--content_metric",
        args.content_metric,
    ]
    histogan_cmd = [sys.executable, "eval_histogan.py", "--sty", style_dir, "--tar", output_dir]
    commands = "{} ; {}".format(command_text(artfid_cmd), command_text(histogan_cmd))
    base_row = {
        "method": args.method,
        "group": args.group,
        "category": category,
        "content_dir": str(args.cnt),
        "style_dir": str(style_dir),
        "output_dir": str(output_dir),
        "image_count": image_count,
        "metric_name": "",
        "metric_value": "",
        "command": commands,
        "timestamp": timestamp,
        "status": "failed" if not ok else "",
        "error": error,
        "log_path": str(log_path),
    }
    if not ok:
        append_metric_rows(args.output_csv, base_row, {})
        return {"category": category, "status": "failed", "error": error, "metrics": {}, "image_count": image_count}

    artfid_code, artfid_output = run_command(artfid_cmd, args.eval_dir, log_path)
    histogan_code, histogan_output = run_command(histogan_cmd, args.eval_dir, log_path)
    metrics = {}
    metrics.update(parse_artfid_output(artfid_output))
    metrics.update(parse_histogan_output(histogan_output))
    missing = [metric for metric in METRICS if not metrics.get(metric)]
    errors = []
    if artfid_code != 0:
        errors.append("eval_artfid exit {}".format(artfid_code))
    if histogan_code != 0:
        errors.append("eval_histogan exit {}".format(histogan_code))
    if missing:
        errors.append("missing metrics: {}".format("|".join(missing)))
    status = "failed" if errors else "completed"
    base_row["status"] = status
    base_row["error"] = "; ".join(errors)
    append_metric_rows(args.output_csv, base_row, metrics)
    metrics_json = {
        "method": args.method,
        "group": args.group,
        "category": category,
        "status": status,
        "metrics": metrics,
        "content_dir": str(args.cnt),
        "style_dir": str(style_dir),
        "output_dir": str(output_dir),
        "image_count": image_count,
        "error": base_row["error"],
        "log_path": str(log_path),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics_json


def write_paper_table(path, categories, rows):
    completed = {row["category"]: row for row in rows if row.get("status") == "completed"}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", "Metric"] + categories + ["Overall"])
        for metric in METRICS:
            values = []
            line = [rows[0]["method"] if rows else "", metric]
            for category in categories:
                value = completed.get(category, {}).get("metrics", {}).get(metric, "")
                if value == "":
                    line.append("")
                    continue
                number = float(value)
                values.append(number)
                line.append("{:.4f}".format(number))
            line.append("{:.4f}".format(sum(values) / len(values)) if values else "")
            writer.writerow(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate method outputs with shared category eval inputs.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--group", default="sty_benchmark_40")
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--cnt", default=Path("data/cnt"), type=Path)
    parser.add_argument("--sty_eval_root", default=Path("data/sty_benchmark_40_eval"), type=Path)
    parser.add_argument("--eval_dir", default=Path("evaluation"), type=Path)
    parser.add_argument("--result_root", required=True, type=Path)
    parser.add_argument("--output_csv", default=None, type=Path)
    parser.add_argument("--categories", default="")
    parser.add_argument("--expected_count", default=800, type=int)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mode", default="art_fid", choices=["art_fid", "art_fid_inf"])
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--content_metric", default="lpips", choices=["lpips", "vgg", "alexnet", "ssim", "ms-ssim"])
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.cnt = args.cnt.resolve()
    args.sty_eval_root = args.sty_eval_root.resolve()
    args.eval_dir = args.eval_dir.resolve()
    args.result_root = args.result_root.resolve()
    args.output_csv = (args.result_root / "per_category_metrics_long.csv") if args.output_csv is None else args.output_csv.resolve()
    return args


def main():
    args = parse_args()
    if not args.cnt.is_dir():
        raise SystemExit("content directory not found: {}".format(args.cnt))
    if not args.sty_eval_root.is_dir():
        raise SystemExit("shared style eval root not found: {}".format(args.sty_eval_root))
    if not args.eval_dir.is_dir():
        raise SystemExit("evaluation directory not found: {}".format(args.eval_dir))

    args.result_root.mkdir(parents=True, exist_ok=True)
    categories = parse_categories(args.sty_eval_root, args.categories)
    write_csv_header(args.output_csv, args.append)
    start = time.time()
    rows = []
    for category in categories:
        print("\n[eval] {} / {}".format(args.method, category))
        row = evaluate_category(args, category)
        rows.append(row)
        if row["status"] == "completed":
            print("completed: ArtFID={}, histo_loss={}".format(row["metrics"].get("artfid", ""), row["metrics"].get("histo_loss", "")))
        else:
            print("failed: {}".format(row["error"]))
    summary = {
        "method": args.method,
        "group": args.group,
        "categories": categories,
        "elapsed_seconds": round(time.time() - start, 2),
        "completed": sum(1 for row in rows if row.get("status") == "completed"),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
        "output_csv": str(args.output_csv),
    }
    (args.result_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_paper_table(args.result_root / "paper_table.csv", categories, rows)
    print("\nResults: {}".format(args.output_csv))
    print("Table:   {}".format(args.result_root / "paper_table.csv"))
    print("Summary: {}".format(args.result_root / "summary.json"))


if __name__ == "__main__":
    main()
