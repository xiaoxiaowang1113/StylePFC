from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pickle
import os

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def align_attention_bank_statistics(cnt_feat, sty_feat, eps=1e-6):
    """Align a content attention bank to style statistics over its tokens."""
    # Attention banks are shaped as (heads, tokens, dim), so alignment happens
    # over tokens before Style-Mixing Self-Attention performs shared normalization.
    cnt_mean = cnt_feat.mean(dim=1, keepdim=True)
    cnt_std = cnt_feat.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    sty_mean = sty_feat.mean(dim=1, keepdim=True)
    sty_std = sty_feat.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.attn = None
        self.q = None
        self.k = None
        self.v = None

    def _repeat_injected(self, tensor, batch_size, reference):
        if tensor is None:
            return None
        # Inversion banks are cached once per sample and expanded to match the CFG batch.
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        return torch.cat([tensor] * batch_size, dim=0)

    def forward(self,
                x,
                context=None,
                mask=None,
                q_injected=None,
                k_injected=None,
                v_injected=None,
                k_style_injected=None,
                k_content_injected=None,
                v_style_branch_injected=None,
                v_content_injected=None,
                injection_config=None,):
        self.attn = None
        h = self.heads
        b = x.shape[0]
        attn_matrix_scale = 1.0
        gamma = 0.0
        tau_s = 1.0
        tau_c = 1.0
        k_align = True
        v_align = True
        smsa_mode = None
        direct_add_lambda = 0.5
        adain_eps = 1e-6
        if injection_config is not None:
            attn_matrix_scale = injection_config.get("T", 1.0)
            gamma = injection_config.get("gamma", 0.0)
            tau_s = injection_config.get("tau_s", attn_matrix_scale)
            tau_c = injection_config.get("tau_c", attn_matrix_scale)
            k_align = injection_config.get("k_align", True)
            v_align = injection_config.get("v_align", True)
            smsa_mode = injection_config.get("smsa_mode")
            direct_add_lambda = injection_config.get("direct_add_lambda", 0.5)
            adain_eps = injection_config.get("adain_eps", 1e-6)

        q_current = self.to_q(x)
        q_current = rearrange(q_current, "b n (h d) -> (b h) n d", h=h)

        if q_injected is None:
            q = q_current
        else:
            # Preserve the cached content query before applying the
            # Style-Mixing Self-Attention routing logic.
            q_in = self._repeat_injected(q_injected, b, q_current)
            q = q_in * gamma + q_current * (1.0 - gamma)

        context = default(context, x)

        use_smsa = (
            smsa_mode is not None
            and q_injected is not None
            and k_style_injected is not None
            and k_content_injected is not None
            and v_style_branch_injected is not None
            and v_content_injected is not None
        )

        direct_add_sims = None
        if use_smsa:
            k_style = self._repeat_injected(k_style_injected, b, q)
            k_content = self._repeat_injected(k_content_injected, b, q)
            v_style = self._repeat_injected(v_style_branch_injected, b, q)
            v_content = self._repeat_injected(v_content_injected, b, q)

            if smsa_mode == "direct_replacement":
                # Q^c retrieves only the raw style value bank.
                k = k_style
                v = v_style
                sim = (
                    einsum("b i d, b j d -> b i j", q, k_style)
                    * self.scale
                )
            elif smsa_mode == "direct_addition":
                # The two raw branches are normalized independently below.
                sim_style = (
                    einsum("b i d, b j d -> b i j", q, k_style)
                    * self.scale
                )
                sim_content = (
                    einsum("b i d, b j d -> b i j", q, k_content)
                    * self.scale
                )
                direct_add_sims = (sim_style, sim_content)
                sim = torch.cat([sim_style, sim_content], dim=-1)
                k = torch.cat([k_style, k_content], dim=1)
                v = torch.cat([v_style, v_content], dim=1)
            else:
                # Feature alignment happens inside attention so ablations can
                # switch it on and off per run.
                k_content_use = (
                    align_attention_bank_statistics(k_content, k_style, eps=adain_eps)
                    if k_align
                    else k_content
                )
                v_content_use = (
                    align_attention_bank_statistics(v_content, v_style, eps=adain_eps)
                    if v_align
                    else v_content
                )

                # Style-Mixing Self-Attention (SMSA) concatenates the two score
                # matrices before one shared softmax. The branches therefore
                # compete for probability mass for every query.
                sim_style = (
                    einsum("b i d, b j d -> b i j", q, k_style)
                    * self.scale
                    * tau_s
                )
                sim_content = (
                    einsum("b i d, b j d -> b i j", q, k_content_use)
                    * self.scale
                    * tau_c
                )
                sim = torch.cat([sim_style, sim_content], dim=-1)
                k = torch.cat([k_style, k_content_use], dim=1)
                v = torch.cat([v_style, v_content_use], dim=1)
        else:
            # Standard self-attention fallback used when SMSA banks are absent.
            if k_injected is None:
                k = self.to_k(context)
                k = rearrange(k, "b m (h d) -> (b h) m d", h=h)
            else:
                k = self._repeat_injected(k_injected, b, q_current)

            if v_injected is None:
                v = self.to_v(context)
                v = rearrange(v, "b m (h d) -> (b h) m d", h=h)
            else:
                v = self._repeat_injected(v_injected, b, q_current)

            sim = einsum("b i d, b j d -> b i j", q, k)
            if q_injected is not None or k_injected is not None:
                sim *= attn_matrix_scale
            sim *= self.scale

        self.q = q
        self.k = k
        self.v = v

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            if direct_add_sims is not None:
                max_neg_value = -torch.finfo(sim.dtype).max
                branch_mask = repeat(mask, "b j -> (b h) () j", h=h)
                sim_style, sim_content = direct_add_sims
                sim_style.masked_fill_(~branch_mask, max_neg_value)
                sim_content.masked_fill_(~branch_mask, max_neg_value)
                direct_add_sims = (sim_style, sim_content)
            elif use_smsa and smsa_mode != "direct_replacement":
                mask = torch.cat([mask, mask], dim=-1)
            if direct_add_sims is None:
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                sim.masked_fill_(~mask, max_neg_value)

        if direct_add_sims is not None:
            sim_style, sim_content = direct_add_sims
            attn_style = sim_style.softmax(dim=-1)
            attn_content = sim_content.softmax(dim=-1)
            self.attn = torch.cat(
                [direct_add_lambda * attn_style, attn_content], dim=-1
            )
            out = (
                direct_add_lambda
                * einsum("b i j, b j d -> b i d", attn_style, v_style)
                + einsum(
                    "b i j, b j d -> b i d", attn_content, v_content
                )
            )
        else:
            attn = sim.softmax(dim=-1)
            self.attn = attn
            out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
    def forward(self,
                x,
                context=None,
                self_attn_q_injected=None,
                self_attn_k_injected=None,
                self_attn_v_injected=None,
                self_attn_k_style_injected=None,
                self_attn_k_content_injected=None,
                self_attn_v_style_branch_injected=None,
                self_attn_v_content_injected=None,
                injection_config=None,
                ):
        return checkpoint(self._forward, (x,
                                          context,
                                          self_attn_q_injected,
                                          self_attn_k_injected,
                                          self_attn_v_injected,
                                          self_attn_k_style_injected,
                                          self_attn_k_content_injected,
                                          self_attn_v_style_branch_injected,
                                          self_attn_v_content_injected,
                                          injection_config,), self.parameters(), self.checkpoint)

    def _forward(self,
                 x,
                 context=None,
                 self_attn_q_injected=None,
                 self_attn_k_injected=None,
                 self_attn_v_injected=None,
                 self_attn_k_style_injected=None,
                 self_attn_k_content_injected=None,
                 self_attn_v_style_branch_injected=None,
                 self_attn_v_content_injected=None,
                 injection_config=None):
        x_ = self.attn1(self.norm1(x),
                       q_injected=self_attn_q_injected,
                       k_injected=self_attn_k_injected,
                       v_injected=self_attn_v_injected,
                       k_style_injected=self_attn_k_style_injected,
                       k_content_injected=self_attn_k_content_injected,
                       v_style_branch_injected=self_attn_v_style_branch_injected,
                       v_content_injected=self_attn_v_content_injected,
                       injection_config=injection_config,)
        x = x_ + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self,
                x,
                context=None,
                self_attn_q_injected=None,
                self_attn_k_injected=None,
                self_attn_v_injected=None,
                self_attn_k_style_injected=None,
                self_attn_k_content_injected=None,
                self_attn_v_style_branch_injected=None,
                self_attn_v_content_injected=None,
                injection_config=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x,
                      context=context,
                      self_attn_q_injected=self_attn_q_injected,
                      self_attn_k_injected=self_attn_k_injected,
                      self_attn_v_injected=self_attn_v_injected,
                      self_attn_k_style_injected=self_attn_k_style_injected,
                      self_attn_k_content_injected=self_attn_k_content_injected,
                      self_attn_v_style_branch_injected=self_attn_v_style_branch_injected,
                      self_attn_v_content_injected=self_attn_v_content_injected,
                      injection_config=injection_config)

            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
