# StylePFC Diffusers implementation

This directory provides the Stable Diffusion v1.5/Diffusers counterpart of the
main Stable Diffusion v1.4/CompVis implementation. Both paths implement the same
full StylePFC configuration:

- CA-AdaIN in the terminal inversion latent;
- Style-Mixing Self-Attention (SMSA) with aligned content-side key/value banks
  and a shared softmax; and
- Phase-Preserving Fourier Correction (PFC) on timestep-matched predicted-clean
  estimates.

The framework-specific attention hooks differ because CompVis and Diffusers use
different UNet APIs. The algorithm, default coefficients, RGB-space Fourier
correction, VAE re-encoding, and predicted-clean latent write-back are the same.

## Run with Stable Diffusion v1.5

From this directory, run:

```bash
python run_stylepfc_diffusers.py \
  --cnt_fn data/content.png \
  --sty_fn data/style.png \
  --save_dir results/example
```

The default configuration is:

```text
sd_version      = 1.5
init_mode       = ca_adain
attn_mode       = smsa
pfc_steps       = 2,5,8
lambda_amp      = 0.5
fusion_alpha    = 0.2
fusion_beta     = 1.0
gamma           = 0.75
tau_c           = 0.4
tau_s           = 1.5
content_weight  = 0.4
style_weight    = 0.6
```

The final image is saved losslessly as `stylized_image.png`. Intermediate
inversion and denoising trajectories are also saved as PNG files.

## Component controls

Use `--without_pfc`, `--without_attn_injection`, or
`--without_init_adain` to disable PFC, attention injection, or terminal-latent
initialization. `--attn_mode styleid` retains the original StyleID attention
replacement path, and `--init_mode adain` retains ordinary AdaIN for comparison.

The optional SD 2.x model choices are inherited from the upstream Diffusers
adapter, but the public StylePFC reproduction path in this directory defaults to
SD v1.5.

## Acknowledgement

This adapter is based on the Diffusers implementation released with
[StyleID](https://github.com/jiwoogit/StyleID). See the repository-level README
and LICENSE for acknowledgements and citation information.
