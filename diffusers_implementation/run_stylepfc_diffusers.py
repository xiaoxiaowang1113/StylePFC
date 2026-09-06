import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import get_args
from stable_diffusion import (
    attention_op,
    decode_latent,
    encode_latent,
    get_text_embedding,
    get_unet_layers,
    load_stable_diffusion,
    style_mixing_self_attention,
)
from utils import denormalize, normalize, save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ca_adain import ca_adain
from modules.pfc import (
    phase_preserving_fourier_correction,
    weighted_writeback,
)


def parse_step_indices(text):
    if not text.strip():
        return set()
    steps = {int(item.strip()) for item in text.split(",") if item.strip()}
    if any(step <= 0 for step in steps):
        raise ValueError("PFC step indices must be positive, one-based integers.")
    return steps


def adain(content, style, eps=1e-4):
    content_mean = content.mean(dim=(2, 3), keepdim=True)
    content_std = content.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    style_mean = style.mean(dim=(2, 3), keepdim=True)
    style_std = style.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    return ((content - content_mean) / content_std) * style_std + style_mean

# class for obtain and override the features
class style_transfer_module():
           
    def __init__(self,
        unet, vae, text_encoder, tokenizer, scheduler, cfg, style_transfer_params = None,
    ):  
        
        style_transfer_params_default = {
            'gamma': 0.75,
            'tau': 1.5,
            'injection_layers': [7, 8, 9, 10, 11]
        }
        if style_transfer_params is not None:
            style_transfer_params_default.update(style_transfer_params)
        self.style_transfer_params = style_transfer_params_default
        
        self.unet = unet # SD unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.cfg = cfg

        self.attn_features = {} # where to save key value (attention block feature)
        self.attn_features_modify = {} # where to save key value to modify (attention block feature)

        self.cur_t = None
        
        # Get residual and attention block in decoder
        # [0 ~ 11], total 12 layers
        resnet, attn = get_unet_layers(unet)
        
        # where to inject key and value
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']
    
        
        for i in qkv_injection_layer_num:
            self.attn_features["layer{}_attn".format(i)] = {}
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__get_query_key_value("layer{}_attn".format(i)))
        
        # Modify hook (if you change query key value)
        for i in qkv_injection_layer_num:
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__modify_self_attn_qkv("layer{}_attn".format(i)))
        
        # triggers for obtaining or modifying features
        
        self.trigger_get_qkv = False # if set True --> save attn qkv in self.attn_features
        self.trigger_modify_qkv = False # if set True --> save attn qkv by self.attn_features_modify
        
        self.modify_num = None # ignore
        self.modify_num_sa = None # ignore
        
    def get_text_condition(self, text):
        if text is None:
            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            device = next(self.text_encoder.parameters()).device
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device)
            )[0]
            return {'encoder_hidden_states': uncond_embeddings}

        device = next(self.text_encoder.parameters()).device
        text_embeddings, uncond_embeddings = get_text_embedding(
            text, self.text_encoder, self.tokenizer, device=device
        )
        text_cond = [text_embeddings, uncond_embeddings]
        denoise_kwargs = {
            'encoder_hidden_states': torch.cat(text_cond)
        }
        return denoise_kwargs

    def _apply_dynamic_content_pfc(self, predicted_clean, content_clean):
        predicted_clean_image = decode_latent(predicted_clean, self.vae)
        content_clean_image = decode_latent(
            content_clean.to(device=predicted_clean.device, dtype=predicted_clean.dtype),
            self.vae,
        )
        # Phase-Preserving Fourier Correction (PFC) operates on decoded,
        # timestep-matched predicted-clean RGB images before VAE write-back.
        aligned_image = phase_preserving_fourier_correction(
            predicted_clean_image=predicted_clean_image,
            content_clean_image=content_clean_image,
            lambda_amp=self.cfg.lambda_amp,
        )
        aligned_latent = encode_latent(aligned_image, self.vae).to(
            device=predicted_clean.device, dtype=predicted_clean.dtype
        )
        return weighted_writeback(
            aligned_latent,
            predicted_clean,
            fusion_alpha=self.cfg.fusion_alpha,
            fusion_beta=self.cfg.fusion_beta,
        )

    def reverse_process(
        self,
        input,
        denoise_kwargs,
        content_clean_by_timestep=None,
        pfc_steps=None,
    ):
        pred_images = []
        pred_latents = []
        pfc_steps = set() if pfc_steps is None else set(pfc_steps)

        # Reverse diffusion process
        for step_index, t in enumerate(tqdm(self.scheduler.timesteps), start=1):
            # setting t (for saving time step)
            self.cur_t = t.item()

            with torch.no_grad():
                # For text condition on stable diffusion
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    input = torch.cat([input] * bs)

                noisy_residual = self.unet(
                    input, t.to(input.device), **denoise_kwargs
                ).sample

                # For text condition on stable diffusion
                if noisy_residual.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noisy_residual.chunk(2)
                    noisy_residual = noise_pred_uncond
                    input, _ = input.chunk(2)

                step_output = self.scheduler.step(noisy_residual, t, input)
                pred_original_sample = step_output.pred_original_sample
                prev_noisy_sample = step_output.prev_sample

                apply_pfc = step_index in pfc_steps
                if apply_pfc:
                    if content_clean_by_timestep is None:
                        raise RuntimeError(
                            "Phase-Preserving Fourier Correction requires the "
                            "content predicted-clean trajectory."
                        )
                    content_clean = content_clean_by_timestep.get(int(t.item()))
                    if content_clean is None:
                        raise RuntimeError(
                            "Missing the content predicted-clean latent for DDIM timestep "
                            f"{int(t.item())}."
                        )
                    corrected = self._apply_dynamic_content_pfc(
                        pred_original_sample, content_clean
                    )

                    # Preserve the scheduler's direction/noise term while replacing
                    # only its predicted-clean component, matching the SD1.4 sampler.
                    timesteps = self.scheduler.timesteps
                    if step_index < len(timesteps):
                        previous_timestep = int(timesteps[step_index].item())
                        alpha_previous = self.scheduler.alphas_cumprod[
                            previous_timestep
                        ].to(device=input.device, dtype=input.dtype)
                    else:
                        alpha_previous = self.scheduler.final_alpha_cumprod.to(
                            device=input.device, dtype=input.dtype
                        )
                    prev_noisy_sample = prev_noisy_sample + alpha_previous.sqrt() * (
                        corrected - pred_original_sample
                    )
                    pred_original_sample = corrected

                input = prev_noisy_sample
                pred_latents.append(pred_original_sample)
                pred_images.append(decode_latent(pred_original_sample, self.vae))

        return pred_images, pred_latents

    ## Inversion (https://github.com/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)
    def invert_process(self, input, denoise_kwargs):
        pred_images = []
        pred_latents = []
        pred_clean_by_timestep = {}

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = list(reversed(self.scheduler.timesteps))
        num_inference_steps = len(self.scheduler.timesteps)
        train_timesteps = self.scheduler.config.num_train_timesteps
        step_size = train_timesteps // num_inference_steps
        cur_latent = input.clone()

        with torch.no_grad():
            for i in tqdm(range(0, num_inference_steps)):
                t = timesteps[i]
                self.cur_t = t.item()

                # For text condition on stable diffusion
                if 'encoder_hidden_states' in denoise_kwargs.keys():
                    bs = denoise_kwargs['encoder_hidden_states'].shape[0]
                    cur_latent = torch.cat([cur_latent] * bs)

                # Predict the noise residual
                noise_pred = self.unet(cur_latent, t.to(cur_latent.device), **denoise_kwargs).sample

                # For text condition on stable diffusion
                if noise_pred.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond
                    cur_latent, _ = cur_latent.chunk(2)

                current_t = max(0, t.item() - step_size)
                next_t = t.item()
                alpha_t = self.scheduler.alphas_cumprod[current_t].to(
                    device=cur_latent.device, dtype=cur_latent.dtype
                )
                alpha_t_next = self.scheduler.alphas_cumprod[next_t].to(
                    device=cur_latent.device, dtype=cur_latent.dtype
                )

                if self.cfg.sd_version == "2.1":
                    beta_t = 1 - alpha_t
                    pred_original_sample = alpha_t.sqrt() * cur_latent - beta_t.sqrt() * noise_pred
                    pred_epsilon = alpha_t.sqrt() * noise_pred + beta_t.sqrt() * cur_latent
                    pred_sample_direction = (1 - alpha_t_next).sqrt() * pred_epsilon
                    cur_latent = alpha_t_next.sqrt() * pred_original_sample + pred_sample_direction
                else:
                    # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
                    pred_original_sample = (
                        cur_latent - (1 - alpha_t).sqrt() * noise_pred
                    ) / alpha_t.sqrt()
                    cur_latent = (cur_latent - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred

                pred_clean_by_timestep[int(t.item())] = (
                    pred_original_sample.detach().clone()
                )
                pred_latents.append(cur_latent)
                pred_images.append(decode_latent(cur_latent, self.vae))

        return pred_images, pred_latents, pred_clean_by_timestep
        
    # ============================ hook operations ===============================
    
    # save key value in self.original_kv[name]
    def __get_query_key_value(self, name):
        def hook(model, input, output):
            
            if self.trigger_get_qkv:
                    
                _, query, key, value, _ = attention_op(model, input[0])
                
                self.attn_features[name][int(self.cur_t)] = (query.detach(), key.detach(), value.detach())
            
        return hook

    
    def __modify_self_attn_qkv(self, name):
        def hook(model, input, output):
            if self.trigger_modify_qkv:
                _, q_cs, k_cs, v_cs, _ = attention_op(model, input[0])
                q_c, k_s, v_s, k_c, v_c = self.attn_features_modify[name][
                    int(self.cur_t)
                ]

                if self.cfg.attn_mode == "smsa":
                    # Style-Mixing Self-Attention (SMSA) uses a shared softmax
                    # over the style and aligned-content branches.
                    return style_mixing_self_attention(
                        model,
                        input[0],
                        q_content=q_c,
                        k_style=k_s,
                        v_style=v_s,
                        k_content=k_c,
                        v_content=v_c,
                        gamma=self.style_transfer_params['gamma'],
                        tau_s=self.cfg.tau_s,
                        tau_c=self.cfg.tau_c,
                        eps=self.cfg.adain_eps,
                    )

                # Original StyleID attention injection fallback.
                q_hat_cs = q_c * self.style_transfer_params['gamma'] + q_cs * (1 - self.style_transfer_params['gamma'])
                k_cs, v_cs = k_s, v_s
                _, _, _, _, modified_output = attention_op(model, input[0], key=k_cs, value=v_cs, query=q_hat_cs, temperature=self.style_transfer_params['tau'])
                return modified_output

        return hook
    
    
if __name__ == "__main__":
    cfg = get_args()

    save_dir = cfg.save_dir
    style_image_bgr = cv2.imread(cfg.sty_fn)
    content_image_bgr = cv2.imread(cfg.cnt_fn)
    if style_image_bgr is None:
        raise FileNotFoundError(f"Cannot read style image: {cfg.sty_fn}")
    if content_image_bgr is None:
        raise FileNotFoundError(f"Cannot read content image: {cfg.cnt_fn}")
    style_image = style_image_bgr[:, :, ::-1]
    content_image = content_image_bgr[:, :, ::-1]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + '/intermediate', exist_ok=True)

    ddim_steps = cfg.ddim_steps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pfc_steps = set() if cfg.without_pfc else parse_step_indices(cfg.pfc_steps)
    if any(step > ddim_steps for step in pfc_steps):
        raise ValueError(
            f"PFC steps {sorted(pfc_steps)} exceed ddim_steps={ddim_steps}."
        )

    style_text = None
    content_text = None

    style_transfer_params = {
        'gamma': cfg.gamma,
        'tau': cfg.T,
        'injection_layers': cfg.layers,
    }

    # Get SD modules
    vae, tokenizer, text_encoder, unet, scheduler = load_stable_diffusion(
        sd_version=cfg.sd_version, precision_t=dtype, device=device
    )
    scheduler.set_timesteps(ddim_steps)

    # Init style transfer module
    unet_wrapper = style_transfer_module(unet, vae, text_encoder, tokenizer, scheduler, cfg, style_transfer_params=style_transfer_params)

    # Get style image tokens
    denoise_kwargs = unet_wrapper.get_text_condition(style_text)

    unet_wrapper.trigger_get_qkv = True # get attention features (key, value)
    unet_wrapper.trigger_modify_qkv = False

    style_latent = encode_latent(normalize(style_image).to(device=vae.device, dtype=dtype), vae)

    # invert process
    print("Invert style image...")
    images, latents, _ = unet_wrapper.invert_process(
        style_latent, denoise_kwargs=denoise_kwargs
    )
    style_latent = latents[-1]

    # save image?
    images = [denormalize(input)[0] for input in images]
    image_last = images[-1]
    images = np.concatenate(images, axis=1)

    save_image(images, os.path.join(save_dir, "intermediate/inversion_style.png"))
    save_image(image_last, os.path.join(save_dir, "intermediate/latent_style.png"))

    # ================= IMPORTANT =================
    # save key value from style image
    style_features = copy.deepcopy(unet_wrapper.attn_features)
    # =============================================
    # Get content image tokens
    denoise_kwargs = unet_wrapper.get_text_condition(content_text)

    unet_wrapper.trigger_get_qkv = True
    unet_wrapper.trigger_modify_qkv = False

    content_latent = encode_latent(normalize(content_image).to(device=vae.device, dtype=dtype), vae)

    # invert process
    print("Invert content image...")
    images, latents, content_clean_by_timestep = unet_wrapper.invert_process(
        content_latent, denoise_kwargs=denoise_kwargs
    )
    content_latent = latents[-1]

    # save image?
    images = [denormalize(input)[0] for input in images]
    image_last = images[-1]
    images = np.concatenate(images, axis=1)

    save_image(images, os.path.join(save_dir, "intermediate/inversion_content.png"))
    save_image(image_last, os.path.join(save_dir, "intermediate/latent_content.png"))

    # ================= IMPORTANT =================
    # save res feature from content image
    content_features = copy.deepcopy(unet_wrapper.attn_features)
    # =============================================
    # ================= IMPORTANT =================
    # Set modify features
    for layer_name in style_features.keys():
        unet_wrapper.attn_features_modify[layer_name] = {}
        for t in scheduler.timesteps:
            t = t.item()
            q_c, k_c, v_c = content_features[layer_name][t]
            _, k_s, v_s = style_features[layer_name][t]
            unet_wrapper.attn_features_modify[layer_name][t] = (
                q_c,
                k_s,
                v_s,
                k_c,
                v_c,
            )
    # =============================================
    
    unet_wrapper.trigger_get_qkv = False
    unet_wrapper.trigger_modify_qkv = not cfg.without_attn_injection # modify attn feature (key value)
    
    
    # Generate style transferred image
    denoise_kwargs = unet_wrapper.get_text_condition(content_text)
    
    if cfg.without_init_adain or cfg.init_mode == "content":
        latent_cs = content_latent
    elif cfg.init_mode == "ca_adain":
        latent_cs = ca_adain(
            content_latent,
            style_latent,
            content_weight=cfg.content_weight,
            style_weight=cfg.style_weight,
        )
    else:
        latent_cs = adain(content_latent, style_latent)

    # reverse process
    print(
        "Style transfer... "
        f"init={cfg.init_mode}, attention={cfg.attn_mode}, "
        "Phase-Preserving Fourier Correction (PFC) "
        f"steps={sorted(pfc_steps)}"
    )
    images, latents = unet_wrapper.reverse_process(
        latent_cs,
        denoise_kwargs=denoise_kwargs,
        content_clean_by_timestep=content_clean_by_timestep,
        pfc_steps=pfc_steps,
    )

    # save image
    images = [denormalize(input)[0] for input in images]
    image_last = images[-1]
    images = np.concatenate(images, axis=1)

    save_image(images, os.path.join(save_dir, "reverse_stylized.png"))
    save_image(image_last, os.path.join(save_dir, "stylized_image.png"))
