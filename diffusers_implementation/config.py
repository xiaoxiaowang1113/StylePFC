import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Run StylePFC with the Diffusers Stable Diffusion implementation."
    )

    # StyleID query protection and attention injection.
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--T", type=float, default=1.5)
    parser.add_argument(
        "--attn_mode", choices=("smsa", "styleid"), default="smsa"
    )
    parser.add_argument("--tau_c", type=float, default=0.4)
    parser.add_argument("--tau_s", type=float, default=1.5)
    parser.add_argument("--adain_eps", type=float, default=1e-6)
    parser.add_argument("--without_attn_injection", action="store_true")
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[7, 8, 9, 10, 11]
    )

    # Content-aware terminal-latent initialization.
    parser.add_argument(
        "--init_mode", choices=("ca_adain", "adain", "content"), default="ca_adain"
    )
    parser.add_argument("--style_weight", type=float, default=0.6)
    parser.add_argument("--content_weight", type=float, default=0.4)
    parser.add_argument("--without_init_adain", action="store_true")

    # Phase-Preserving Fourier Correction (PFC). Step indices are one-based
    # reverse-DDIM loop indices.
    parser.add_argument("--pfc_steps", type=str, default="2,5,8")
    parser.add_argument("--lambda_amp", type=float, default=0.5)
    parser.add_argument("--fusion_alpha", type=float, default=0.2)
    parser.add_argument("--fusion_beta", type=float, default=1.0)
    parser.add_argument("--without_pfc", action="store_true")

    # Diffusion model. The public Diffusers path targets SD v1.5 by default.
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument(
        "--sd_version",
        choices=("1.5", "2.0", "2.1-base", "2.1"),
        default="1.5",
    )

    parser.add_argument("--cnt_fn", type=str, required=True)
    parser.add_argument("--sty_fn", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    return parser.parse_args()
