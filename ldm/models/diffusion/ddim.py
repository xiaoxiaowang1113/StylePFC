"""SAMPLING ONLY - FPD with configurable DDIM step sets."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    extract_into_tensor,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)
from modules.pfc import (
    PFC_MODES,
    clean_triplet_phase_fusion,
    phase_preserving_fourier_correction,
    paper_phase_fusion,
    weighted_writeback,
)


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor) and attr.device != torch.device("cuda"):
            attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True, strength=1.0):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
            strength=strength,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)))

        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        self.register_buffer("ddim_sqrt_alphas", np.sqrt(ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    def make_negative_prompt_schedule(self, negative_prompt_schedule, negative_prompt_alpha, total_steps):
        if negative_prompt_schedule == "linear":
            negative_prompt_schedule = np.flip(np.linspace(0, 1, total_steps))
        elif negative_prompt_schedule == "constant":
            negative_prompt_schedule = np.flip(np.ones(total_steps))
        elif negative_prompt_schedule == "exp":
            negative_prompt_schedule = np.exp(-6 * np.linspace(0, 1, total_steps))
        else:
            raise NotImplementedError
        return negative_prompt_schedule * negative_prompt_alpha

    @staticmethod
    def _normalize_fpd_steps(fpd_steps):
        if fpd_steps is None:
            return {2, 5, 8}
        return {int(step) for step in fpd_steps}

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        negative_conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        injected_features=None,
        strength=1.0,
        callback_ddim_timesteps=None,
        negative_prompt_alpha=1.0,
        negative_prompt_schedule="constant",
        style_img=None,
        style_guidance=0.0,
        content_guidance=0.0,
        start_step=9999,
        content_latents=None,
        style_latents=None,
        content_clean_latents=None,
        style_clean_latents=None,
        content_reference_image=None,
        fpd_mode=None,
        fusion_alpha=0.2,
        fusion_beta=1.0,
        lambda_amp=0.5,
        fpd_steps=None,
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            elif conditioning.shape[0] != batch_size:
                print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose, strength=strength)
        channels, height, width = shape
        size = (batch_size, channels, height, width)
        print(f"Data shape for DDIM sampling is {size}, eta {eta}")

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            negative_conditioning=negative_conditioning,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            injected_features=injected_features,
            callback_ddim_timesteps=callback_ddim_timesteps,
            negative_prompt_alpha=negative_prompt_alpha,
            negative_prompt_schedule=negative_prompt_schedule,
            style_img=style_img,
            style_guidance=style_guidance,
            content_guidance=content_guidance,
            start_step=start_step,
            content_latents=content_latents,
            style_latents=style_latents,
            content_clean_latents=content_clean_latents,
            style_clean_latents=style_clean_latents,
            content_reference_image=content_reference_image,
            fpd_mode=fpd_mode,
            fusion_alpha=fusion_alpha,
            fusion_beta=fusion_beta,
            lambda_amp=lambda_amp,
            fpd_steps=fpd_steps,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        negative_conditioning=None,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        injected_features=None,
        callback_ddim_timesteps=None,
        negative_prompt_alpha=1.0,
        negative_prompt_schedule="constant",
        style_img=None,
        style_guidance=1.0,
        content_guidance=1.0,
        start_step=9999,
        content_latents=None,
        style_latents=None,
        content_clean_latents=None,
        style_clean_latents=None,
        content_reference_image=None,
        fpd_mode=None,
        fusion_alpha=0.2,
        fusion_beta=1.0,
        lambda_amp=0.5,
        fpd_steps=None,
    ):
        device = self.model.betas.device
        batch_size = shape[0]
        img = torch.randn(shape, device=device) if x_T is None else x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        callback_ddim_timesteps_list = (
            np.flip(make_ddim_timesteps("uniform", callback_ddim_timesteps, self.ddpm_num_timesteps))
            if callback_ddim_timesteps is not None
            else np.flip(self.ddim_timesteps)
        )
        negative_prompt_alpha_schedule = self.make_negative_prompt_schedule(
            negative_prompt_schedule,
            negative_prompt_alpha,
            total_steps,
        )
        style_loss = None
        legacy_latent_mode = fpd_mode is None
        if not legacy_latent_mode and fpd_mode not in PFC_MODES:
            raise ValueError(
                f"Unsupported PFC mode: {fpd_mode}. Expected one of {PFC_MODES}."
            )
        fpd_steps_set = self._normalize_fpd_steps(fpd_steps)
        displayed_mode = "legacy_latent" if legacy_latent_mode else fpd_mode
        print(f"PFC mode: {displayed_mode}; active steps: {sorted(fpd_steps_set)}")

        if fpd_mode == "dynamic_content":
            required_content_indices = [
                i
                for i in range(total_steps)
                if total_steps - i - 1 < start_step and i + 1 in fpd_steps_set
            ]
            if required_content_indices and content_clean_latents is None:
                raise ValueError(
                    "PFC mode 'dynamic_content' requires content_clean_latents."
                )
            if required_content_indices and len(content_clean_latents) <= max(
                required_content_indices
            ):
                raise ValueError(
                    "PFC mode 'dynamic_content' content clean trajectory is too short: "
                    f"length={len(content_clean_latents)}, required loop index="
                    f"{max(required_content_indices)}."
                )

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            if index >= start_step:
                continue

            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            injected_features_i = injected_features[i] if injected_features is not None and len(injected_features) > 0 else None
            negative_prompt_alpha_i = negative_prompt_alpha_schedule[i]

            step_id = i + 1
            if legacy_latent_mode:
                fpd_inputs_ready = (
                    content_latents is not None and style_latents is not None
                )
            elif fpd_mode == "paper":
                fpd_inputs_ready = content_reference_image is not None
            elif fpd_mode == "dynamic_content":
                fpd_inputs_ready = content_clean_latents is not None
            else:
                fpd_inputs_ready = (
                    content_clean_latents is not None
                    and style_clean_latents is not None
                )
            apply_fpd = fpd_inputs_ready and step_id in fpd_steps_set
            x_c_latent = (
                content_latents[i] if apply_fpd and legacy_latent_mode else None
            )
            x_s_latent = (
                style_latents[i] if apply_fpd and legacy_latent_mode else None
            )
            x_c_clean = (
                content_clean_latents[i]
                if apply_fpd and fpd_mode in ("dynamic_content", "clean_triplet")
                else None
            )
            x_s_clean = (
                style_clean_latents[i]
                if apply_fpd and fpd_mode == "clean_triplet"
                else None
            )

            img, pred_x0 = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                negative_conditioning=negative_conditioning,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                injected_features=injected_features_i,
                negative_prompt_alpha=negative_prompt_alpha_i,
                style_loss=style_loss,
                style_guidance_scale=style_guidance,
                style_img=style_img,
                content_guidance_scale=content_guidance,
                apply_fpd=apply_fpd,
                fpd_mode=fpd_mode,
                content_reference_image=content_reference_image,
                x_c_latent=x_c_latent,
                x_s_latent=x_s_latent,
                x_c_clean=x_c_clean,
                x_s_clean=x_s_clean,
                fpd_loop_index=i,
                fusion_alpha=fusion_alpha,
                fusion_beta=fusion_beta,
                lambda_amp=lambda_amp,
            )

            if step in callback_ddim_timesteps_list:
                if callback:
                    callback(i)
                if img_callback:
                    img_callback(pred_x0, img, step)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        negative_conditioning=None,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        injected_features=None,
        negative_prompt_alpha=1.0,
        style_guidance_scale=1.0,
        style_loss=None,
        style_img=None,
        content_guidance_scale=1.0,
        apply_fpd=False,
        fpd_mode=None,
        content_reference_image=None,
        x_c_latent=None,
        x_s_latent=None,
        x_c_clean=None,
        x_s_clean=None,
        fpd_loop_index=None,
        fusion_alpha=0.2,
        fusion_beta=1.0,
        lambda_amp=0.5,
    ):
        batch_size, *_, device = *x.shape, x.device

        if negative_conditioning is not None:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([negative_conditioning, unconditional_conditioning])
            e_t_negative, e_t_uncond = self.model.apply_model(
                x_in,
                t_in,
                c_in,
                injected_features=injected_features,
            ).chunk(2)

            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(
                x_in,
                t_in,
                c_in,
                injected_features=injected_features,
            ).chunk(2)

            e_t_tilde = negative_prompt_alpha * e_t_uncond + (1 - negative_prompt_alpha) * e_t_negative
            e_t = e_t_tilde + unconditional_guidance_scale * (e_t - e_t_tilde)
        elif c is not None:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(
                x_in,
                t_in,
                c_in,
                injected_features=injected_features,
            ).chunk(2)
            if unconditional_guidance_scale != 1:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        else:
            e_t = self.model.apply_model(
                x,
                t,
                unconditional_conditioning,
                injected_features=injected_features,
            )

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if apply_fpd:
            if fpd_mode is None:
                if x_c_latent is None or x_s_latent is None:
                    raise RuntimeError("Legacy latent PFC requires content/style latents.")
                pred_x0 = self._latent_phase_fusion(
                    x_latent=pred_x0,
                    x_c_latent=x_c_latent,
                    x_s_latent=x_s_latent,
                    lambda_amp=lambda_amp,
                    fusion_alpha=fusion_alpha,
                    fusion_beta=fusion_beta,
                )
            elif fpd_mode in ("paper", "dynamic_content"):
                if fpd_mode == "paper":
                    reference_image = content_reference_image
                    fusion_fn = paper_phase_fusion
                else:
                    if not torch.is_tensor(x_c_clean) or not x_c_clean.is_floating_point():
                        raise TypeError(
                            "PFC mode 'dynamic_content' dynamic content clean latent must "
                            "be a floating-point tensor at loop index "
                            f"{fpd_loop_index}."
                        )
                    if x_c_clean.shape != pred_x0.shape:
                        raise ValueError(
                            "PFC mode 'dynamic_content' dynamic content clean latent shape "
                            f"mismatch at loop index {fpd_loop_index}: "
                            f"content={tuple(x_c_clean.shape)}, "
                            f"generated={tuple(pred_x0.shape)}."
                        )
                    x_c_clean = x_c_clean.to(
                        device=pred_x0.device,
                        dtype=pred_x0.dtype,
                    )
                    reference_image = self.model.decode_first_stage(x_c_clean)
                    # Phase-Preserving Fourier Correction (PFC) preserves the
                    # timestep-aligned content phase, mixes Fourier amplitudes,
                    # and returns the corrected RGB candidate for VAE write-back.
                    fusion_fn = phase_preserving_fourier_correction

                predicted_clean_image = self.model.decode_first_stage(pred_x0)
                aligned_image = fusion_fn(
                    predicted_clean_image=predicted_clean_image,
                    **(
                        {"content_image": reference_image}
                        if fpd_mode == "paper"
                        else {"content_clean_image": reference_image}
                    ),
                    lambda_amp=lambda_amp,
                )
                aligned_latent = self.model.get_first_stage_encoding(
                    self.model.encode_first_stage(aligned_image)
                )
                aligned_latent = aligned_latent.to(
                    dtype=pred_x0.dtype,
                    device=pred_x0.device,
                )
                pred_x0 = weighted_writeback(
                    aligned_latent,
                    pred_x0,
                    fusion_alpha=fusion_alpha,
                    fusion_beta=fusion_beta,
                )
            elif fpd_mode == "clean_triplet" and x_c_clean is not None and x_s_clean is not None:
                pred_x0 = clean_triplet_phase_fusion(
                    predicted_clean=pred_x0,
                    content_clean=x_c_clean,
                    style_clean=x_s_clean,
                    lambda_amp=lambda_amp,
                    fusion_alpha=fusion_alpha,
                    fusion_beta=fusion_beta,
                )
            else:
                raise RuntimeError(f"Missing required inputs for PFC mode '{fpd_mode}'.")

        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @staticmethod
    def _latent_adain(content, style, eps=1e-5):
        content_mean = content.mean(dim=[0, 2, 3], keepdim=True)
        content_std = content.std(dim=[0, 2, 3], keepdim=True) + eps
        style_mean = style.mean(dim=[0, 2, 3], keepdim=True)
        style_std = style.std(dim=[0, 2, 3], keepdim=True) + eps
        normalized = (content - content_mean) / content_std
        return normalized * style_std + style_mean

    def _latent_phase_fusion(
        self,
        x_latent,
        x_c_latent,
        x_s_latent,
        lambda_amp=0.5,
        fusion_alpha=0.2,
        fusion_beta=1.0,
    ):
        device = x_latent.device
        dtype = x_latent.dtype
        _, _, height, width = x_latent.shape

        if x_c_latent.shape[-2:] != (height, width):
            x_c = F.interpolate(x_c_latent, size=(height, width), mode="bilinear", align_corners=False)
        else:
            x_c = x_c_latent

        if x_s_latent.shape[-2:] != (height, width):
            x_s = F.interpolate(x_s_latent, size=(height, width), mode="bilinear", align_corners=False)
        else:
            x_s = x_s_latent

        fft_content = torch.fft.fft2(x_c, dim=(-2, -1))
        fft_style = torch.fft.fft2(x_s, dim=(-2, -1))

        amp_content = torch.abs(fft_content)
        amp_style = torch.abs(fft_style)
        phase_content = torch.angle(fft_content)

        amp_mix = lambda_amp * amp_style + (1.0 - lambda_amp) * amp_content
        fft_mix = amp_mix * torch.exp(1j * phase_content)
        fused = torch.fft.ifft2(fft_mix, dim=(-2, -1)).real
        fused = self._latent_adain(fused, x_latent)
        z_hat = fusion_alpha * fused + fusion_beta * x_latent
        return z_hat.to(dtype=dtype, device=device)

    @torch.no_grad()
    def encode_ddim(
        self,
        img,
        num_steps,
        conditioning=None,
        unconditional_conditioning=None,
        unconditional_guidance_scale=1.0,
        end_step=999,
        callback_ddim_timesteps=None,
        img_callback=None,
    ):
        print(f"Running DDIM inversion with {num_steps} timesteps")
        if num_steps == 999:
            total = 999
            stride = total // num_steps
            iterator = tqdm(range(0, total, stride), desc="DDIM Inversion", total=num_steps)
            steps = list(range(0, total + stride, stride))
        else:
            total = self.ddpm_num_timesteps
            stride = total // num_steps
            time_steps = range(1, total, stride)
            iterator = tqdm(time_steps, desc="DDIM Inversion", total=num_steps)
            steps = list(range(1, total + stride, stride))
            steps[-1] = 999

        callback_ddim_timesteps_list = (
            np.flip(make_ddim_timesteps("uniform", callback_ddim_timesteps, self.ddpm_num_timesteps))
            if callback_ddim_timesteps is not None
            else np.flip(self.ddim_timesteps)
        )
        for i, t in enumerate(iterator):
            if i > end_step:
                break
            img, pred_x0 = self.reverse_ddim(
                img,
                t,
                t_next=steps[i + 1],
                c=conditioning,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=unconditional_guidance_scale,
            )
            if t in callback_ddim_timesteps_list and img_callback:
                img_callback(pred_x0, img, t)
        return img, pred_x0

    @torch.no_grad()
    def reverse_ddim(
        self,
        x,
        t,
        t_next,
        c=None,
        quantize_denoised=False,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        batch_size, *_, device = *x.shape, x.device
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        if c is None:
            e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(x, t_tensor, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_tensor] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod
        a_t = torch.full((batch_size, 1, 1, 1), alphas[t], device=device)
        a_next = torch.full((batch_size, 1, 1, 1), alphas[t_next], device=device)
        sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[t], device=device)

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        dir_xt = (1.0 - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * pred_x0 + dir_xt
        return x_next, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @torch.no_grad()
    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
    ):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return x_dec
