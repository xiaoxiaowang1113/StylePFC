# StylePFC

Training-free reference-image style transfer built on Stable Diffusion and
[StyleID](https://github.com/jiwoogit/StyleID). The default method combines
CA-AdaIN, Style-Mixing Self-Attention (SMSA), and Phase-Preserving Fourier
Correction (PFC).

## Usage

1. Setup
2. Run StylePFC
3. Evaluation

## Setup

Create the environment:

```bash
conda env create -f environment.yaml
conda activate stylepfc
```

The runner downloads the Stable Diffusion v1.4 checkpoint on the first run and
caches it inside the project. For an offline run, pass a local checkpoint with
`--ckpt /path/to/sd-v1-4.ckpt` (or place it at
`models/ldm/stable-diffusion-v1/model.ckpt`). Accept the model license on the
[Stable Diffusion v1.4 page](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
before downloading.

## Data layout

Place the dataset under `data/` as shown below. The image files are not included
in this repository; see [DATASET.md](DATASET.md) for construction details.

```text
data/
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

The runner converts images to RGB, center-crops them, and resizes them to
`512 x 512`.

## Inference

The defaults correspond to the full method: CA-AdaIN, Style-Mixing
Self-Attention (SMSA), and Phase-Preserving Fourier Correction (PFC). With the
dataset layout above, run the following command from the repository root. The
default `--sty data/sty` is a category root: the runner discovers all six
subdirectories, processes them independently, and writes each result under
`output/<category>`.

```bash
python run_stylepfc.py --cnt data/cnt --sty data/sty
```

For a single content-style pair, the optional Diffusers implementation can be
run with:

```bash
python diffusers_implementation/run_stylepfc_diffusers.py \
  --cnt_fn data/cnt/00.png \
  --sty_fn data/sty/C1_Impressionist/00.png \
  --save_dir results/example
```

To save DDIM inversion features, optionally add
`--precomputed precomputed/stylefamilybench`; a separate cache is created for
each category.

## Utilities

The scripts in `util/` are only used for evaluation and input preparation; they
are not needed for the normal six-category inference command.

- `prepare_flat_eval_refs.py` creates the flat 800-image content/style reference
  folders required by the ArtFID and HistoGAN scripts.
- `copy_inputs.py` creates the same kind of pairwise evaluation copies under
  `<input>_eval` and supports category subdirectories.
- `copy_inputs_flat.py` is the legacy helper for a single flat style directory.
- `eval_shared_category.py` provides the shared category-wise evaluation and
  metric-recording workflow.

For example, prepare the evaluation folders for one category with:

```bash
python util/prepare_flat_eval_refs.py \
  --cnt data/cnt \
  --sty data/sty/C1_Impressionist \
  --cnt-eval data/stylefamilybench_eval/C1_Impressionist/cnt \
  --sty-eval data/stylefamilybench_eval/C1_Impressionist/sty \
  --size 512 \
  --expected-count 800
```

To prepare all categories with the older pair-copy helper, run:

```bash
python util/copy_inputs.py --cnt data/cnt --sty data/sty
```

Replace `C1_Impressionist` with the other category names when using
`prepare_flat_eval_refs.py`.

## Evaluation

After preparing the flat evaluation references, run the two evaluation scripts
for each category. For example:

```bash
python evaluation/eval_artfid.py --sty data/stylefamilybench_eval/C1_Impressionist/sty --cnt data/stylefamilybench_eval/C1_Impressionist/cnt --tar output/C1_Impressionist
python evaluation/eval_histogan.py --sty data/stylefamilybench_eval/C1_Impressionist/sty --tar output/C1_Impressionist
```

Replace `C1_Impressionist` with `C2_Expressionist` through `C6_Modern` for the
remaining categories.

## Citation



