import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from transformers import Dinov2Model, Dinov2Config
from torch.nn import Module
from collections import namedtuple
from tqdm import tqdm

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



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.fn(x_norm, *args, **kwargs)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half = dim // 2
        self.weights = nn.Parameter(torch.randn(half), requires_grad=not is_random)

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)

# ----------------------------
# Residual & Attention Blocks
# ----------------------------
class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.LayerNorm(dim_out)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x.transpose(1,2)).transpose(1,2)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.drop(self.act(x))

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out*2))
                    if time_emb_dim is not None else None)
        self.block1 = Block(dim, dim_out, dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = (nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity())

    def forward(self, x, t_emb=None):
        scale_shift = None
        if self.mlp is not None and t_emb is not None:
            t = self.mlp(t_emb).reshape(x.size(0), -1, 1)
            scale_shift = t.chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden*3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [t.view(b, self.heads, -1, n) for t in qkv]
        q = q * self.scale
        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = out.reshape(b, -1, n)
        return self.to_out(out)


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Config, Dinov2Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2Config

class DINOv2Encoder(nn.Module):
    def __init__(self, finetune_last_n_layers=0):
        super().__init__()
        cfg = Dinov2Config.from_pretrained('facebook/dinov2-small')
        cfg.output_hidden_states = False
        cfg.return_dict = True
        self.encoder = Dinov2Model.from_pretrained('facebook/dinov2-small', config=cfg)
        
        # 直接使用模型的原始维度，移除embed_dim参数
        self.hidden_size = cfg.hidden_size
        
        for param in self.parameters():
            param.requires_grad = False
        if finetune_last_n_layers > 0:
            if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
                blocks = self.encoder.encoder.layer
                num_layers = len(blocks)
                for i in range(num_layers - finetune_last_n_layers, num_layers):
                    if i >= 0:
                        for param in blocks[i].parameters():
                            param.requires_grad = True
                        print(f"Unfreezing transformer block {i}")

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        outputs = self.encoder(pixel_values=x)
        tokens = outputs.last_hidden_state
        patch_tokens = tokens[:, 1:, :]  # 移除 CLS token，只保留 patch tokens
        
        # 对所有 patch tokens 进行平均池化，得到一个 token
        pooled_token = torch.mean(patch_tokens, dim=1, keepdim=True)  # [B, 1, hidden_size]
        
        # 归一化并返回
        return F.normalize(pooled_token, dim=-1)






# ----------------------------
# Helper for diffusion
# ----------------------------
def extract(a, t, shape):
    return a.gather(0, t).reshape(t.size(0), *((1,) * (len(shape)-1)))

def linear_beta_schedule(T):
    scale = 1000 / T
    return torch.linspace(scale*1e-4, scale*0.02, T, dtype=torch.float64)

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    a2 = torch.cos(((x/T)+s)/(1+s)*math.pi*0.5)**2
    a2 = a2 / a2[0]
    betas = 1 - (a2[1:]/a2[:-1])
    return torch.clip(betas, 0, 0.999)

class Unet1DConditional(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 6, 8),
        channels=3,
        dropout=0.,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=32,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=8,
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

        self.downs = nn.ModuleList()
        for ind, (di, dout) in enumerate(in_out):
            is_last = ind == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                resnet(di, di),
                resnet(di, di),
                Residual(PreNorm(di, Attention(di, dim_head=attn_dim_head, heads=attn_heads))),
                Residual(
                    FiLM(di, context_dim)  # 替换CrossAttention为FiLM
                ),
                Downsample(di, dout) if not is_last else nn.Conv1d(di, dout, 3, padding=1)
            ]))

        # middle
        mid = dims[-1]
        self.mid_block1 = resnet(mid, mid)
        self.mid_attn = Residual(PreNorm(mid, Attention(mid, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_film = Residual(FiLM(mid, context_dim))  
        self.mid_block2 = resnet(mid, mid)

        self.ups = nn.ModuleList()
        for ind, (di, dout) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(nn.ModuleList([
                resnet(dout + di, dout),
                resnet(dout + di, dout),
                Residual(PreNorm(dout, Attention(dout, dim_head=attn_dim_head, heads=attn_heads))),
                Residual(
                    FiLM(dout, context_dim)  
                ),
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
        skip = x0.clone()
        t_emb = self.time_mlp(time)

        h = []
        # down
        for b1, b2, attn, film, down in self.downs:
            x0 = b1(x0, t_emb); h.append(x0)
            x0 = b2(x0, t_emb)
            x0 = attn(x0)
            x0 = film(x0, z)
            h.append(x0)
            x0 = down(x0)

        # mid
        x0 = self.mid_block1(x0, t_emb)
        x0 = self.mid_attn(x0)
        x0 = self.mid_film(x0, z)  
        x0 = self.mid_block2(x0, t_emb)
        # up
        for b1, b2, attn, film, up in self.ups:
            x0 = torch.cat([x0, h.pop()], dim=1)
            x0 = b1(x0, t_emb)
            x0 = torch.cat([x0, h.pop()], dim=1)
            x0 = b2(x0, t_emb)
            x0 = attn(x0)
            x0 = film(x0, z) 
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

        #b, c, n = x_start.shape
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


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, unet_dim=64, seq_length=128, finetune_last_n_layers=2, diffusion_kwargs=None):
        super().__init__()
        diffusion_kwargs = diffusion_kwargs or {}
        self.encoder = DINOv2Encoder(finetune_last_n_layers=finetune_last_n_layers)

        unet_kwargs = {
            'channels': diffusion_kwargs.get('channels', 3),
            'dim_mults': diffusion_kwargs.get('dim_mults', (1,2,4,6,8)),
            'dropout': diffusion_kwargs.get('dropout', 0.),
            'self_condition': diffusion_kwargs.get('self_condition', False),
            'learned_variance': diffusion_kwargs.get('learned_variance', False),
            'learned_sinusoidal_cond': diffusion_kwargs.get('learned_sinusoidal_cond', False),
            'random_fourier_features': diffusion_kwargs.get('random_fourier_features', False),
            'attn_dim_head': diffusion_kwargs.get('attn_dim_head', 8),
            'attn_heads': diffusion_kwargs.get('attn_heads', 4),
        }
        self.unet = Unet1DConditional(
            dim=unet_dim,
            context_dim=384,
            **unet_kwargs
        )

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
        """
        x_seq: [B, C, L]
        cond_img: [B, 3, H, W]
        mask: [B, L]
        """
        z = self.encoder(cond_img)  # [B, 1+P, embed_dim]
        t = torch.randint(0, self.diffusion.num_timesteps, (x_seq.size(0),), device=x_seq.device)
        return self.diffusion.p_losses(x_seq, t, z, mask=mask)

    @torch.no_grad()
    def sample(self, batch_size, cond_img):
        z = self.encoder(cond_img)  # [B, 1+P, embed_dim]
        return self.diffusion.sample(batch_size, z)
