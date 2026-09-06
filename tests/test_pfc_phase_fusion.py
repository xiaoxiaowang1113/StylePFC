import torch

from modules.pfc import (
    PFC_MODES,
    clean_triplet_phase_fusion,
    phase_preserving_fourier_correction,
    paper_phase_fusion,
    weighted_writeback,
)


def test_supported_modes_include_dynamic_content_and_comparisons():
    assert PFC_MODES == ("paper", "dynamic_content", "clean_triplet")


def test_dynamic_content_preserves_identical_rgb_predictions():
    image = torch.randn(1, 3, 8, 8, dtype=torch.float32)

    result = phase_preserving_fourier_correction(image, image, lambda_amp=0.5)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert torch.allclose(result, image, atol=1e-5, rtol=1e-5)


def test_paper_mode_preserves_identical_images():
    image = torch.randn(1, 3, 8, 8, dtype=torch.float32)

    result = paper_phase_fusion(image, image, lambda_amp=0.5)

    assert torch.allclose(result, image, atol=1e-5, rtol=1e-5)


def test_clean_triplet_preserves_identical_clean_latents():
    latent = torch.randn(1, 4, 8, 8, dtype=torch.float32)

    result = clean_triplet_phase_fusion(
        predicted_clean=latent,
        content_clean=latent,
        style_clean=latent,
        lambda_amp=0.5,
        fusion_alpha=1.0,
        fusion_beta=0.0,
    )

    assert torch.allclose(result, latent, atol=1e-5, rtol=1e-5)


def test_weighted_writeback_uses_requested_weights():
    candidate = torch.ones(1, 4, 2, 2)
    reference = torch.full_like(candidate, 2.0)

    result = weighted_writeback(
        candidate,
        reference,
        fusion_alpha=0.25,
        fusion_beta=0.75,
    )

    assert torch.allclose(result, torch.full_like(candidate, 1.75))
