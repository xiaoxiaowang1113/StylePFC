"""Phase-Preserving Fourier Correction (PFC) building blocks.

The public Phase-Preserving Fourier Correction path operates on RGB images
decoded from timestep-matched predicted-clean latents. The sampler is
responsible for VAE decoding/encoding and for writing the encoded candidate
back to the current predicted-clean latent.
"""

import torch
import torch.nn.functional as F


PFC_MODES = ("paper", "dynamic_content", "clean_triplet")


def latent_adain(content, style, eps=1e-5):
    """Match per-channel mean and standard deviation without spatial shuffling."""
    content_mean = content.mean(dim=[0, 2, 3], keepdim=True)
    content_std = content.std(dim=[0, 2, 3], keepdim=True) + eps
    style_mean = style.mean(dim=[0, 2, 3], keepdim=True)
    style_std = style.std(dim=[0, 2, 3], keepdim=True) + eps
    normalized = (content - content_mean) / content_std
    return normalized * style_std + style_mean


def _resize_like(source, reference):
    if source.shape[-2:] == reference.shape[-2:]:
        return source
    return F.interpolate(
        source,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _fft_dtype(tensor):
    # CUDA FFT does not support every half-precision shape used by this project.
    if tensor.dtype in (torch.float32, torch.float64):
        return tensor.dtype
    return torch.float32


def weighted_writeback(candidate, reference, fusion_alpha=0.2, fusion_beta=1.0):
    """Write an encoded PFC candidate back to a predicted-clean latent."""
    return fusion_alpha * candidate + fusion_beta * reference


def _rgb_phase_fusion(predicted_clean_image, content_image, lambda_amp=0.5):
    """Construct an RGB PFC candidate using content phase and mixed amplitude."""
    device = predicted_clean_image.device
    dtype = predicted_clean_image.dtype
    content = _resize_like(content_image.to(device=device), predicted_clean_image)
    work_dtype = _fft_dtype(predicted_clean_image)
    current = predicted_clean_image.to(dtype=work_dtype)
    content = content.to(dtype=work_dtype)

    fft_current = torch.fft.fft2(current, dim=(-2, -1))
    fft_content = torch.fft.fft2(content, dim=(-2, -1))
    amplitude = (
        (1.0 - lambda_amp) * torch.abs(fft_current)
        + lambda_amp * torch.abs(fft_content)
    )
    phase = torch.angle(fft_content)

    reconstructed = torch.fft.ifft2(
        amplitude * torch.exp(1j * phase),
        dim=(-2, -1),
    ).real
    aligned = latent_adain(reconstructed, current)
    return aligned.to(dtype=dtype, device=device)


# Phase-Preserving Fourier Correction (PFC)
# This is the paper's main correction module. It combines the amplitude of the
# current stylization prediction with the timestep-aligned content amplitude,
# preserves the content phase, and returns an RGB candidate for VAE write-back.
def phase_preserving_fourier_correction(
    predicted_clean_image,
    content_clean_image,
    lambda_amp=0.5,
):
    """Correct decoded, timestep-matched predicted-clean RGB predictions."""
    return _rgb_phase_fusion(
        predicted_clean_image=predicted_clean_image,
        content_image=content_clean_image,
        lambda_amp=lambda_amp,
    )


def paper_phase_fusion(predicted_clean_image, content_image, lambda_amp=0.5):
    """Legacy comparison mode using the fixed original RGB content image."""
    return _rgb_phase_fusion(
        predicted_clean_image=predicted_clean_image,
        content_image=content_image,
        lambda_amp=lambda_amp,
    )


def clean_triplet_phase_fusion(
    predicted_clean,
    content_clean,
    style_clean,
    lambda_amp=0.5,
    fusion_alpha=0.2,
    fusion_beta=1.0,
):
    """Legacy comparison mode operating directly on three clean-latent estimates."""
    device = predicted_clean.device
    dtype = predicted_clean.dtype
    content = _resize_like(content_clean.to(device=device), predicted_clean)
    style = _resize_like(style_clean.to(device=device), predicted_clean)
    work_dtype = _fft_dtype(predicted_clean)
    current = predicted_clean.to(dtype=work_dtype)
    content = content.to(dtype=work_dtype)
    style = style.to(dtype=work_dtype)

    fft_content = torch.fft.fft2(content, dim=(-2, -1))
    fft_style = torch.fft.fft2(style, dim=(-2, -1))
    amplitude = (
        (1.0 - lambda_amp) * torch.abs(fft_content)
        + lambda_amp * torch.abs(fft_style)
    )
    phase = torch.angle(fft_content)

    reconstructed = torch.fft.ifft2(
        amplitude * torch.exp(1j * phase),
        dim=(-2, -1),
    ).real
    aligned = latent_adain(reconstructed, current)
    written = weighted_writeback(
        aligned,
        current,
        fusion_alpha=fusion_alpha,
        fusion_beta=fusion_beta,
    )
    return written.to(dtype=dtype, device=device)


def latent_phase_fusion(
    x_latent,
    x_c_latent,
    x_s_latent,
    lambda_amp=0.5,
    fusion_alpha=0.2,
    fusion_beta=1.0,
):
    """Backward-compatible latent-space PFC used by older diagnostic scripts."""
    device = x_latent.device
    dtype = x_latent.dtype
    x_c = _resize_like(x_c_latent, x_latent)
    x_s = _resize_like(x_s_latent, x_latent)

    fft_content = torch.fft.fft2(x_c, dim=(-2, -1))
    fft_style = torch.fft.fft2(x_s, dim=(-2, -1))
    amplitude = (
        (1.0 - lambda_amp) * torch.abs(fft_content)
        + lambda_amp * torch.abs(fft_style)
    )
    phase = torch.angle(fft_content)

    reconstructed = torch.fft.ifft2(
        amplitude * torch.exp(1j * phase),
        dim=(-2, -1),
    ).real
    aligned = latent_adain(reconstructed, x_latent)
    written = weighted_writeback(
        aligned,
        x_latent,
        fusion_alpha=fusion_alpha,
        fusion_beta=fusion_beta,
    )
    return written.to(dtype=dtype, device=device)
