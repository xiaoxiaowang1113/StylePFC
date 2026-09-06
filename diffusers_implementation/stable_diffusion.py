
import math

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


# From "https://huggingface.co/blog/stable_diffusion"
def load_stable_diffusion(sd_version='2.1', precision_t=torch.float32, device="cuda"):
    model_keys = {
        "1.5": "runwayml/stable-diffusion-v1-5",
        "2.0": "stabilityai/stable-diffusion-2-base",
        "2.1-base": "stabilityai/stable-diffusion-2-1-base",
        "2.1": "stabilityai/stable-diffusion-2-1",
    }
    if sd_version not in model_keys:
        raise ValueError(f"Unsupported Stable Diffusion version: {sd_version}")
    model_key = model_keys[sd_version]
        
    # Create model
    pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)
    
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    # import xformer
    # unet.enable_xformers_memory_efficient_attention()
    
    del pipe
    
    # Use DDIM scheduler
    scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)
    
    return vae, tokenizer, text_encoder, unet, scheduler

def decode_latent(latents, vae):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    return image

def encode_latent(images, vae):
    # encode the image with vae
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mode()
    latents = 0.18215 * latents
    return latents

def get_text_embedding(text, text_encoder, tokenizer, device="cuda"):
    # TODO currently, hard-coding for stable diffusion
    with torch.no_grad():

        prompt = [text]
        batch_size = len(prompt)
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(device))[0].to(device)
        max_length = text_input.input_ids.shape[-1]
        # print(max_length, text_input.input_ids)
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(device)
    
    return text_embeddings, uncond_embeddings



def get_unet_layers(unet):
    
    layer_num = [i for i in range(12)]
    resnet_layers = []
    attn_layers = []
    
    for idx, ln in enumerate(layer_num):
        up_block_idx = idx // 3
        layer_idx = idx % 3
        
        resnet_layers.append(getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx])
        if up_block_idx > 0:
            attn_layers.append(getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx])
        else:
            attn_layers.append(None)
        
    return resnet_layers, attn_layers
        
        
        

# Diffusers attention code for getting query, key, value and attention map
def attention_op(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0):
    residual = hidden_states
    
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, None)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    if query is None:
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    if key is None:
        key = attn.to_k(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
    if value is None:
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

    
    if key.shape[0] != query.shape[0]:
        key, value = key[:query.shape[0]], value[:query.shape[0]]

    # apply temperature scaling
    query = query * temperature # same as applying it on qk matrix

    if attention_probs is None:
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

    batch_heads, img_len, txt_len = attention_probs.shape
    
    # h = w = int(img_len ** 0.5)
    # attention_probs_return = attention_probs.reshape(batch_heads // attn.heads, attn.heads, h, w, txt_len)
    
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    
    return attention_probs, query, key, value, hidden_states


def attention_bank_adain(content, style, eps=1e-6):
    """Align head-folded attention banks over their token dimension."""
    content_mean = content.mean(dim=1, keepdim=True)
    content_std = content.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    style_mean = style.mean(dim=1, keepdim=True)
    style_std = style.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return ((content - content_mean) / content_std) * style_std + style_mean


def _match_head_batch(tensor, reference):
    tensor = tensor.to(device=reference.device, dtype=reference.dtype)
    if tensor.shape[0] == reference.shape[0]:
        return tensor
    if reference.shape[0] % tensor.shape[0] != 0:
        raise ValueError(
            "Cached attention bank cannot be expanded to the current head batch: "
            f"cached={tensor.shape[0]}, current={reference.shape[0]}."
        )
    return tensor.repeat(reference.shape[0] // tensor.shape[0], 1, 1)


# Style-Mixing Self-Attention (SMSA)
# The style branch and the style-aligned content branch produce separate logits.
# Their logits are concatenated before one shared softmax, so both branches
# compete for query-specific probability mass before their values are aggregated.
def style_mixing_self_attention(
    attn,
    hidden_states,
    q_content,
    k_style,
    v_style,
    k_content,
    v_content,
    gamma=0.75,
    tau_s=1.5,
    tau_c=0.4,
    eps=1e-6,
):
    """Apply Style-Mixing Self-Attention through the Diffusers attention API."""
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, None)

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    else:
        batch_size = hidden_states.shape[0]

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    q_current = attn.head_to_batch_dim(attn.to_q(hidden_states))
    q_content = _match_head_batch(q_content, q_current)
    q = gamma * q_content + (1.0 - gamma) * q_current

    k_style = _match_head_batch(k_style, q)
    v_style = _match_head_batch(v_style, q)
    k_content = attention_bank_adain(_match_head_batch(k_content, q), k_style, eps)
    v_content = attention_bank_adain(_match_head_batch(v_content, q), v_style, eps)

    scale = getattr(attn, "scale", 1.0 / math.sqrt(q.shape[-1]))
    style_logits = torch.bmm(q, k_style.transpose(1, 2)) * scale * tau_s
    content_logits = torch.bmm(q, k_content.transpose(1, 2)) * scale * tau_c
    attention_probs = torch.cat([style_logits, content_logits], dim=-1).softmax(dim=-1)
    values = torch.cat([v_style, v_content], dim=1)
    hidden_states = torch.bmm(attention_probs, values)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch_size, channel, height, width
        )
    if attn.residual_connection:
        hidden_states = hidden_states + residual
    return hidden_states / attn.rescale_output_factor
