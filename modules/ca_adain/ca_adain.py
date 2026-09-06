import torch


def ca_adain(inv_cnt_latent, inv_style_latent, content_weight=0.4, style_weight=0.6, eps=1e-4):
    """
    Content-Aware AdaIN (AttenST-style initialization).
    inv_cnt_latent: [B,C,H,W] content trajectory latent at T step
    inv_style_latent: [B,C,H,W] style trajectory latent at T step
    """
    if inv_cnt_latent.shape != inv_style_latent.shape:
        raise ValueError("CA-AdaIN expects content/style latents with identical shapes.")

    cnt_mean = inv_cnt_latent.mean(dim=(2, 3), keepdim=True)
    cnt_std = inv_cnt_latent.std(dim=(2, 3), keepdim=True, unbiased=False) + eps

    sty_mean = inv_style_latent.mean(dim=(2, 3), keepdim=True)
    sty_std = inv_style_latent.std(dim=(2, 3), keepdim=True, unbiased=False) + eps

    std_mix = style_weight * sty_std + content_weight * cnt_std
    mean_mix = style_weight * sty_mean + content_weight * cnt_mean

    norm_cnt = (inv_cnt_latent - cnt_mean) / cnt_std
    latent_cs = norm_cnt * std_mix + mean_mix
    return latent_cs
