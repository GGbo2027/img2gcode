import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn
from tqdm import tqdm
from .transformer import VisionTransformer_1, LayerNormFp32, QuickGELU



# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs) + x

# FiLM
class FiLM(nn.Module):
    def __init__(self, feature_dim, context_dim):
        super().__init__()
        self.to_film_params = nn.Sequential(
            nn.Linear(context_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

    def forward(self, x, context):
        # x: [B, C, L], context: [B, ctx_dim] or [B, 1, ctx_dim]
        if context.dim() == 3:
            context = context.squeeze(1)
        
        # Get gamma and beta
        film_params = self.to_film_params(context)  # [B, feature_dim*2]
        gamma, beta = film_params.chunk(2, dim=1)   #  [B, feature_dim]
        
        # Adjust gamma and beta to match x's channel dimension
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)    # [B, C, 1]
        
        return gamma * x + beta

class PreNormFiLM(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, context, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, context, *args, **kwargs) + x

# sinusoidal positional embeds
class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules
class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class ViTConfig:
    def __init__(
        self,
        image_size=64,
        patch_size=4,
        width=256,
        layers=8,
        heads=8,
        mlp_ratio=4.0,
        ls_init_value=None,
        pos_embed_type='learnable',
        no_ln_pre=False,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.ls_init_value = ls_init_value
        self.pos_embed_type = pos_embed_type
        self.no_ln_pre = no_ln_pre

class ViT(nn.Module):
    def __init__(self, config: ViTConfig, embed_dim: int, quick_gelu: bool = False):
        super().__init__()
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if config.ls_init_value is not None else nn.LayerNorm

        self.vit = VisionTransformer_1(
            image_size=config.image_size,
            patch_size=config.patch_size,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            ls_init_value=config.ls_init_value,
            pos_embed_type=config.pos_embed_type,
            no_ln_pre=config.no_ln_pre,
            act_layer=act_layer,
            norm_layer=norm_layer,
            output_dim=embed_dim,      
            pool_type='none',         
            output_tokens=True,       
        )
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # tokens: All token，includes cls and patches，[B, 1+P, D]
        _, tokens = self.vit(x)

        cls_tok    = tokens[:, 0, :]             # [B, D]

        patch_mean = tokens[:, 1:, :].mean(dim=1)  # [B, D]

        fused = torch.cat([cls_tok, patch_mean], dim=1)  # [B, 2*D]
        out   = self.fuse_proj(fused)                     # [B, D]

        if normalize:
            out = F.normalize(out, dim=-1)
        return out



# -- Conditional 1D U-Net with FiLM --
def extract(a, t, x_shape):
    # t: [B], long
    b, *_ = t.shape
    out = a.gather(0, t)  # a: [T], so gather(0)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# 1D U-Net with learned mask head
class Unet1DConditional(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2, 4, 6, 8),
        channels=3,
        dropout=0.,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=32,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=16,
        attn_heads=4,
        context_dim=None,
    ):
        super().__init__()
        assert context_dim is not None, "context_dim must be provided"
        self.channels = channels
        self.self_condition = self_condition
        in_ch = channels * (2 if self_condition else 1)

        init_dim = init_dim or dim
        self.init_conv = nn.Conv1d(in_ch, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embedding
        time_dim = dim * 4
        sinu = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta) if not (learned_sinusoidal_cond or random_fourier_features) else RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
        fourier_dim = dim if not (learned_sinusoidal_cond or random_fourier_features) else (learned_sinusoidal_dim + 1)
        self.time_mlp = nn.Sequential(
            sinu,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        resnet = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)
        self.context_proj = nn.Linear(context_dim, dims[-1])

        self.downs = nn.ModuleList()
        for ind, (di, dout) in enumerate(in_out):
            is_last = ind == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                resnet(di, di),
                resnet(di, di),
                #Residual(PreNorm(di, LinearAttention(di))),
                Residual(PreNorm(di, Attention(di, dim_head=attn_dim_head, heads=attn_heads))),
                Residual(PreNormFiLM(di, FiLM(di, dims[-1]))),
                Downsample(di, dout) if not is_last else nn.Conv1d(di, dout, 3, padding=1)
            ]))

        # middle
        mid = dims[-1]
        self.mid_block1 = resnet(mid, mid)
        self.mid_attn = Residual(PreNorm(mid, Attention(mid, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = resnet(mid, mid)
        
        self.ups = nn.ModuleList()
        for ind, (di, dout) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                resnet(dout + di, dout),
                resnet(dout + di, dout),
                #Residual(PreNorm(dout, LinearAttention(dout))),
                Residual(PreNorm(dout, Attention(dout, dim_head=attn_dim_head, heads=attn_heads))),
                Residual(PreNormFiLM(dout, FiLM(dout, dims[-1]))),
                Upsample(dout, di) if not is_last else nn.Conv1d(dout, di, 3, padding=1)
            ]))

        out_ch = channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim or out_ch
        self.final_res = resnet(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)
        self.mask_head = nn.Sequential(
            nn.Conv1d(init_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, time, z):
        x0 = self.init_conv(x)
        #print(f"Input shape: {x0.shape}, Time shape: {time.shape}, Context shape: {z.shape}")
        skip = x0.clone()
        t_emb = self.time_mlp(time)
        z = z.unsqueeze(1) if z.dim() == 2 else z
        ctx = self.context_proj(z)

        h = []
        # down
        for b1, b2, attn, film, down in self.downs:
            x0 = b1(x0, t_emb); h.append(x0)
            x0 = b2(x0, t_emb)
            x0 = attn(x0)
            x0 = film(x0, ctx) 
            h.append(x0)
            x0 = down(x0)
        # mid
        x0 = self.mid_block1(x0, t_emb)
        x0 = self.mid_attn(x0)
        x0 = self.mid_block2(x0, t_emb)
        # up
        for b1, b2, attn, film, up in self.ups:
            x0 = torch.cat([x0, h.pop()], dim=1)
            x0 = b1(x0, t_emb)
            x0 = torch.cat([x0, h.pop()], dim=1)
            x0 = b2(x0, t_emb)
            x0 = attn(x0)
            x0 = film(x0, ctx)  
            x0 = up(x0)
        # final
        x0 = torch.cat([x0, skip], dim=1)
        x0 = self.final_res(x0, t_emb)
        out = self.final_conv(x0)
        mask = self.mask_head(x0)
        return out, mask

class GaussianDiffusion1DConditional(nn.Module):
    def __init__(self, model, *, seq_length, timesteps=1000, sampling_timesteps=None,
                 objective='pred_noise', beta_schedule='cosine', ddim_sampling_eta=0., normalize=False):
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        assert objective in {'pred_noise','pred_x0','pred_v'}
        self.objective = objective

        # create betas
        if beta_schedule=='linear':
            betas = linear_beta_schedule(timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)
        alphas = 1 - betas
        acp = torch.cumprod(alphas, 0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.)
        self.num_timesteps = timesteps
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        # register buffers
        def reg(n, v): self.register_buffer(n, v.to(torch.float32))
        reg('betas', betas)
        reg('alphas_cumprod', acp)
        reg('alphas_cumprod_prev', acp_prev)
        reg('sqrt_alphas_cumprod', torch.sqrt(acp))
        reg('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - acp))
        reg('posterior_variance', betas * (1 - acp_prev) / (1 - acp))
        reg('posterior_mean_coef1', betas * torch.sqrt(acp_prev) / (1 - acp))
        reg('posterior_mean_coef2', (1 - acp_prev) * torch.sqrt(alphas) / (1 - acp))

        # loss weight
        snr = acp / (1 - acp)
        if objective == 'pred_noise':
            lw = torch.ones_like(snr)
        elif objective == 'pred_x0':
            lw = snr
        else:
            lw = snr / (snr + 1)
        reg('loss_weight', lw)

        # normalize
        if normalize:
            self.normalize = lambda x: 2*x - 1
            self.unnormalize = lambda x: (x + 1)/2
        else:
            self.normalize = lambda x: x
            self.unnormalize = lambda x: x

    def q_sample(self, x_start, t, noise=None, mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
              extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        if mask is not None:
            x_t = x_t * mask + x_start * (1 - mask)
        return x_t

    def p_losses(self, x_start, t, z, noise=None, mask=None):
        # mask: [B, L] -> [B,1,L]
        if mask is not None:
            mask = mask.unsqueeze(1)

        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start, t, noise, mask)
        model_out, mask_pred = self.model(x_noisy, t, z)

        # choose target for diffusion loss
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            target = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                      extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start)

        loss_seq = F.mse_loss(model_out, target, reduction='none')
        channel_weights = torch.tensor([1.35, 1.35, 0.3], device=loss_seq.device).view(1, 3, 1)
        loss_seq = loss_seq * channel_weights
        
        if mask is not None:
            loss_seq = loss_seq * mask
        loss_seq = loss_seq.mean(dim=[1,2])
        loss_seq = (loss_seq * extract(self.loss_weight, t, loss_seq.shape)).mean()

        # BCE loss for mask prediction
        loss_mask = 0.0
        if mask is not None:
            loss_mask = F.binary_cross_entropy(mask_pred, mask, reduction='mean')

        # total loss
        return loss_seq + loss_mask

    @torch.no_grad()
    def p_sample(self, x, t, z, mask=None):
        # mask: [B, L] -> [B,1,L]
        if mask is not None:
            mask = mask.unsqueeze(1)

        # compute posterior mean/variance
        model_out, mask_pred = self.model(x, t, z)
        # reconstruct x_start
        if self.objective == 'pred_noise':
            x_start = (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * x -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * model_out
            )
        else:
            x_start = model_out.clamp(-1,1)

        mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        var = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x) if (t[0] > 0).item() else torch.zeros_like(x)
        x_next = mean + var.sqrt() * noise

        # choose which mask to apply: predicted or given
        m = mask_pred if mask is None else mask
        if m is not None:
            x_next = x_next * m + x * (1 - m)
        return x_next

    @torch.no_grad()
    def sample(self, batch_size, z):
        # initial noise
        shape = (batch_size, self.model.channels, self.seq_length)
        x = torch.randn(shape, device=self.betas.device)

        for t_int in tqdm(reversed(range(self.num_timesteps)), desc='sampling'):
            # make t tensor
            t_tensor = torch.full((batch_size,), t_int,
                                  device=x.device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, z)
        return self.unnormalize(x)

# Wrapper
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, vit, vit_embed_dim, unet_dim, seq_length, diffusion_kwargs):
        super().__init__()
        self.vit = vit
        
        unet_kwargs = {
            'channels': diffusion_kwargs.get('channels', 3),
            'dim_mults': diffusion_kwargs.get('dim_mults', (1, 2, 2, 4, 6, 8)),
            'dropout': diffusion_kwargs.get('dropout', 0.),
            'self_condition': diffusion_kwargs.get('self_condition', False),
            'learned_variance': diffusion_kwargs.get('learned_variance', False),
            'learned_sinusoidal_cond': diffusion_kwargs.get('learned_sinusoidal_cond', False),
            'random_fourier_features': diffusion_kwargs.get('random_fourier_features', False),
            'attn_dim_head': diffusion_kwargs.get('attn_dim_head', 16),
            'attn_heads': diffusion_kwargs.get('attn_heads', 4),
        }
        
        # initialize UNet
        self.unet = Unet1DConditional(
            dim=unet_dim,
            context_dim=vit_embed_dim,
            **unet_kwargs
        )
        
        # initialize diffusion model
        self.diffusion = GaussianDiffusion1DConditional(
            model=self.unet,
            seq_length=seq_length,
            timesteps=diffusion_kwargs.get('timesteps', 1000),
            sampling_timesteps=diffusion_kwargs.get('sampling_timesteps', None),
            objective=diffusion_kwargs.get('objective', 'pred_noise'),
            beta_schedule=diffusion_kwargs.get('beta_schedule', 'cosine'),
            normalize=diffusion_kwargs.get('normalize', False)
        )

    def forward(self, x_seq, cond_img, mask):
        # x_seq: [B,C,L], cond_img: [B,3,H,W], mask: [B,L]
        z = self.vit(cond_img).unsqueeze(1)  # [B,1,embed]
        t = torch.randint(0, self.diffusion.num_timesteps,
                          (x_seq.size(0),), device=x_seq.device)
        return self.diffusion.p_losses(x_seq, t, z, mask=mask)

    @torch.no_grad()
    def sample(self, batch_size, cond_img):
        z = self.vit(cond_img).unsqueeze(1)
        return self.diffusion.sample(batch_size, z)
