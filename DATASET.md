# StyleFamilyBench: Dataset Preparation and Evaluation Protocol

StylePFC is training-free and does not require a training dataset. For
category-wise evaluation, we use **StyleFamilyBench**, a fixed benchmark built
from COCO content images and WikiArt style references.

StyleFamilyBench is designed to expose performance variations that can be hidden
when heterogeneous styles are evaluated as one mixed set. It contains six style
families and evaluates every method independently on each family.

## Benchmark summary

| Item | Count | Source |
| --- | ---: | --- |
| Content images | 20 | COCO |
| Style families | 6 | WikiArt labels grouped by art-historical similarity |
| Style references per family | 40 | Artist-balanced WikiArt selection |
| Total style references | 240 | 6 families x 40 images |
| Content-style pairs per family | 800 | 20 content x 40 style images |
| Content-style pairs per method | 4,800 | 6 families x 800 pairs |

All methods must use the same 4,800 content-style pairs. Metrics are computed
separately for each family. Because every family contains the same number of
pairs, the overall result is the arithmetic mean of the six family-level
results.

## Data sources and usage terms

Obtain the source images from their original providers:

- Reuse the exact 20 evaluation images in
  [StyleID `data/cnt`](https://github.com/jiwoogit/StyleID/tree/main/data/cnt);
  their underlying source is [COCO](https://cocodataset.org/#download).
- [`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart) is the
  81,444-image, 27-style source used by the released construction manifests.

The source datasets and benchmark images are not committed to this repository.
Users are responsible for complying with the terms, licenses, and attribution
requirements of COCO, WikiArt, and the individual artworks. This repository does
not grant additional rights to redistribute the source images.

## Style-family taxonomy

The 27 WikiArt style labels are grouped into the following six benchmark
families. `Art_Nouveau` and `Art_Nouveau_Modern` are treated as aliases used by
different WikiArt exports, not as two distinct styles.

| Family | Constituent WikiArt labels |
| --- | --- |
| `C1_Impressionist` | `Impressionism`, `Post_Impressionism`, `Pointillism`, `Fauvism` |
| `C2_Expressionist` | `Expressionism`, `Abstract_Expressionism`, `Color_Field_Painting`, `Action_painting` |
| `C3_Geometric` | `Cubism`, `Analytical_Cubism`, `Synthetic_Cubism`, `Art_Nouveau` (`Art_Nouveau_Modern`) |
| `C4_Decorative` | `Baroque`, `Rococo`, `Mannerism_Late_Renaissance`, `Symbolism` |
| `C5_Realistic` | `Realism`, `Romanticism`, `High_Renaissance`, `Early_Renaissance`, `Northern_Renaissance` |
| `C6_Modern` | `Minimalism`, `Contemporary_Realism`, `New_Realism`, `Pop_Art`, `Naive_Art_Primitivism`, `Ukiyo_e` |

## Expected raw WikiArt layout

The selection procedure expects a label-preserving WikiArt export. If the
downloaded dataset uses Parquet or another storage format, first export it to one
directory per fine-grained style label while preserving artist information in
the metadata or file names.

```text
wikiart/
└── images/
    ├── Impressionism/
    │   ├── artist-name_image-id.jpg
    │   └── ...
    ├── Abstract_Expressionism/
    │   └── ...
    ├── Cubism/
    │   └── ...
    └── ...
```

## Benchmark construction

The complete reconstruction scripts are in [`dataset_tools/`](dataset_tools/).
Install their optional dependency first:

```bash
pip install -r dataset_tools/requirements.txt
```

The tools write raw data and intermediate images only below the repository-local
`data/` directory, which is excluded from version control. They refuse to merge
results into a non-empty output directory; use a new output path for a rebuild.

### 0. Export the WikiArt source

The default command downloads a pinned `huggan/wikiart` revision
(`d559852d2b232e0fcf195e775866964f0564f2b5`), verifies that the full source has
81,444 images and 27 style labels, and exports it into label directories:

```bash
python dataset_tools/download_wikiart.py
```

Default locations:

```text
data/.cache/huggingface/       # repository-local download cache
data/raw/wikiart/images/       # exported style-label directories
data/raw/wikiart/export_summary.json
```

The resolved dataset fingerprint, pinned revision, label list, and per-label
counts are written to `export_summary.json`. `--max_samples` and
`--allow_incomplete` are debugging options and must not be used to build the
benchmark.

### 1. Build the candidate pools

Map each source style label to its family using the taxonomy above. Apply the
following image-only filters before inspecting any model output:

| Filter | Setting |
| --- | ---: |
| Minimum short edge | 768 pixels |
| Maximum long-to-short edge ratio | 3:1 |
| Minimum file size | 50 KB |
| Grayscale rejection | RGB mean range `< 5` and standard-deviation range `< 3` |
| Maximum images per artist | 3 |
| Minimum images per constituent style | 2, when enough valid images exist |
| Random seed | 42 |

Select 50 candidates for each family. Center-crop each selected image to a
square, resize it to `512 x 512` with Lanczos resampling, and retain a provenance
record containing the original path, WikiArt label, artist, image dimensions,
and the applied filter settings.

If a constituent style does not contain two valid images, keep the available
images and record the exception rather than silently replacing them with images
from another label.

To reconstruct the exact 50 candidates per family used in the released
benchmark, run:

```bash
python dataset_tools/select_wikiart_styles.py \
  --wikiart_root data/raw/wikiart/images \
  --output_dir data/stylefamilybench_work/candidates_50 \
  --selection_manifest dataset_tools/manifests/selection_log.json \
  --per_category 50 \
  --min_size 768 \
  --max_aspect_ratio 3.0 \
  --min_file_kb 50 \
  --max_per_artist 3 \
  --min_per_substyle 2 \
  --resize_to 512 \
  --seed 42
```

The released manifest path is the default, but it is shown explicitly here for
clarity. The script validates the full source, resolves every released filename
exactly once, and fails if any candidate is missing or belongs to the wrong
family.

To create a new candidate set under the same automatic protocol, add
`--resample`. This produces a different benchmark version and must not be
reported as the released StyleFamilyBench selection:

```bash
python dataset_tools/select_wikiart_styles.py \
  --wikiart_root data/raw/wikiart/images \
  --output_dir data/stylefamilybench_work/resampled_candidates_50 \
  --resample \
  --seed 42
```

### 2. Manually review the candidates

Review all 50 candidates in each family without consulting outputs or metrics
from StylePFC or any comparison method. Reject an image only for a predefined
input-quality reason, including:

- an evidently incorrect style label;
- a visible watermark or obstruction covering a substantial image region;
- a photographic image incorrectly included as an artwork;
- a text-only image or an unreadable/corrupted image.

Retain exactly 40 style references per family and store the rejection decisions
in a machine-readable review log.

To inspect a new candidate set manually, generate the six local HTML galleries:

```bash
python dataset_tools/review_candidates.py gallery \
  data/stylefamilybench_work/candidates_50
```

For exact reconstruction, apply the six released rejection manifests included
in the repository:

```bash
python dataset_tools/review_candidates.py apply \
  data/stylefamilybench_work/candidates_50 \
  --rejected_dir dataset_tools/manifests \
  --keep 40 \
  --output_dir data/stylefamilybench_work/candidates_40
```

The apply step requires exactly 40 surviving images in every family. It fails on
missing manifests, unknown filenames, duplicate categories, or an incorrect
survivor count instead of silently truncating the selection.

### 3. Finalize names and provenance

Sort the retained images deterministically and save them as `00.png` through
`39.png`. Preserve the complete new-name-to-source-name mapping. A reproducible
release should retain the following metadata even when the source images cannot
be redistributed:

```text
selection_log.json   # source path, label, artist, and selected candidate
filter_stats.json    # counts and rejection reasons for automatic filters
review_log.json      # manual decisions and predefined reasons
name_mapping.json    # final benchmark name -> original source name
```

Finalize the six-by-40 benchmark:

```bash
python dataset_tools/rename_to_digits.py \
  --input_dir data/stylefamilybench_work/candidates_40 \
  --output_dir data/sty \
  --expected_per_category 40
```

This writes both `name_mapping.json` and `benchmark_manifest.json`. The latter
records the final dimensions, color mode, source filename, file SHA-256, and
decoded RGB-pixel SHA-256 for every generated PNG.

The selection criteria must be fixed before evaluating any method. Selecting or
removing references according to generated outputs or metric values constitutes
data snooping and invalidates the comparison.

## Final StylePFC data layout

Place the 20 fixed content images and the six finalized style directories under
the repository-local `data/` directory:

```text
StylePFC/
└── data/
    ├── cnt/
    │   ├── 00.png
    │   ├── ...
    │   └── 19.png
    └── sty/
        ├── C1_Impressionist/
        │   ├── 00.png
        │   ├── ...
        │   └── 39.png
        ├── C2_Expressionist/
        ├── C3_Geometric/
        ├── C4_Decorative/
        ├── C5_Realistic/
        └── C6_Modern/
```

Input files may use PNG, JPEG, BMP, or WebP. The runner converts images to RGB
and resizes them to `512 x 512`; the curated benchmark uses PNG to avoid adding
another lossy-compression stage.

## Run StylePFC by family

When `--sty` points to the category root `data/sty`, `run_stylepfc.py`
automatically discovers all six category subdirectories and runs them
independently. It creates a separate output and precomputed-feature directory
for each family because the style files reuse the stems `00` through `39`.

Run all six families from the repository root:

```bash
python run_stylepfc.py \
  --cnt data/cnt \
  --sty data/sty \
  --output_path output/stylefamilybench \
  --precomputed precomputed/stylefamilybench
```

A complete run produces 800 images per family and 4,800 images in total.
Output names follow the shared convention:

```text
{content_stem}_stylized_{style_stem}.png
```

The runner also writes `pair_manifest.csv`, `run_metadata.json`, and
`pair_progress.jsonl` to each output directory for traceability and safe resume.

## Validate the reconstructed benchmark

After placing the fixed 20 StyleID/COCO content images in `data/cnt`, validate
the complete benchmark before running any method:

```bash
python dataset_tools/verify_stylefamilybench.py \
  --content_dir data/cnt \
  --style_dir data/sty \
  --released_mapping dataset_tools/manifests/name_mapping.json \
  --released_pixels dataset_tools/manifests/stylefamilybench_pixels.json
```

The validator is read-only. It checks the 20 content names, six family names,
40 style names per family, `512 x 512` RGB decoding, the released filename
mapping, and all checksums generated during finalization. A valid reconstruction
reports 20 content images, 240 style images, and 4,800 total pairs.

The released pixel hashes are computed from decoded `512 x 512` RGB values. They
remain stable when different Pillow or PNG encoder versions produce different
lossless file bytes for the same pixels.

## Prepare evaluation references

ArtFID and the other image-level metrics require content and style reference
folders whose file names match the generated outputs. Prepare the references for
each family independently:

```bash
python util/prepare_flat_eval_refs.py \
  --cnt data/cnt \
  --sty data/sty/C1_Impressionist \
  --cnt-eval data/stylefamilybench_eval/C1_Impressionist/cnt \
  --sty-eval data/stylefamilybench_eval/C1_Impressionist/sty \
  --size 512 \
  --expected-count 800
```

Then evaluate the corresponding generated directory:

```bash
python evaluation/eval_artfid.py \
  --sty data/stylefamilybench_eval/C1_Impressionist/sty \
  --cnt data/stylefamilybench_eval/C1_Impressionist/cnt \
  --tar output/stylefamilybench/C1_Impressionist

python evaluation/eval_histogan.py \
  --sty data/stylefamilybench_eval/C1_Impressionist/sty \
  --tar output/stylefamilybench/C1_Impressionist
```

Repeat this procedure for all six families and report each metric independently.
The evaluation utilities can report ArtFID, FID, LPIPS, grayscale LPIPS, CFSD,
and color-matching loss.

## Comparison-method protocol

For a fair comparison, every external method must:

1. use the same 20 content images and the same 240 style references;
2. evaluate all 4,800 content-style pairs without subsampling;
3. use the same center-crop and `512 x 512` input resolution;
4. save one directory per style family;
5. follow the `{content_stem}_stylized_{style_stem}.png` naming convention;
6. use the same prepared references and evaluation implementation;
7. report family-level metrics before the six-family arithmetic mean.

Do not merge the six style directories into a single pool for the primary
evaluation. A mixed-pool result may be reported only as an additional sanity
check and must not replace the family-wise benchmark.
