from __future__ import annotations
import torch
sqrt  = torch.sqrt     # 用 torch 版，避免 math→Tensor 报错
expm1 = torch.expm1
import math
from functools import wraps
import numpy as np
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from tqdm import tqdm
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
import inspect 
from typing import Callable, Tuple
from typing import Callable, Optional



# helpers

def exists(val):
    return val is not None

def identity(t):
    return t

def is_lambda(f):
    return inspect.isfunction(f) and f.__name__ == "<lambda>"

def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

def l2norm(t):
    return F.normalize(t, dim = -1)

class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        factor = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(
    dim,
    dim_out = None,
    factor = 1
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = factor, p2 = factor),
        nn.Conv2d(dim * (factor ** 2), default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, normalize_dim = 2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim = normalize_dim) * scale * (x.shape[normalize_dim] ** 0.5)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out, normalize_dim = 1)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim = 1)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim, normalize_dim = 1)
        )

    def forward(self, x):
        residual = x

        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out) + residual

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 8, dropout = 0.1):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = RMSNorm(dim, scale = False)
        dim_hidden = dim * mult

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_hidden * 2),
            Rearrange('b d -> b 1 d')
        )

        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias = False),
            nn.SiLU()
        )

        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias = False)
        )

    def forward(self, x, t):
        x = self.norm(x)
        x = self.proj_in(x)

        scale, shift = self.to_scale_shift(t).chunk(2, dim = -1)
        x = x * (scale + 1) + shift

        return self.proj_out(x)

# vit

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        time_cond_dim,
        depth,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        dropout = 0.,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim = dim, mult = ff_mult, cond_dim = time_cond_dim, dropout = dropout)
            ]))

    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x, t) + x

        return x






def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))



def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def logsnr_schedule_cosine(t, logsnr_min = -15, logsnr_max = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner

def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner

class UViT2DTensor(nn.Module):
    """
    Vision-Transformer-to-Tensor 2D diffusion backbone
    使用ViT前后的适度上下采样来提高特征提取能力
    优化用于64x64单通道图像，并保留更多细节信息
    """
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 1,
        vit_depth: int = 8,
        vit_dropout: float = 0.2,
        attn_dim_head: int = 32,
        attn_heads: int = 8,
        ff_mult: int = 4,
        learned_sinusoidal_dim: int = 16,
        max_tensor_len: int = 165,
        feature_dim: int = 3,
        init_img_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        final_img_itransform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        patch_size: int = 4,
        dual_patchnorm: bool = False,
    ) -> None:
        super().__init__()

        # ---------------- 基本属性 ----------------
        self.max_tensor_len = max_tensor_len
        self.feature_dim = feature_dim

        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        input_channels = channels
        init_dim = default(init_dim, dim)

        # ---------------- 自定义残差块 ----------------
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.norm1 = nn.GroupNorm(8, channels)
                self.act1 = nn.SiLU()
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.norm2 = nn.GroupNorm(8, channels)
                self.act2 = nn.SiLU()
                
            def forward(self, x):
                residual = x
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.act1(x)
                x = self.conv2(x)
                x = self.norm2(x)
                x = residual + x
                return self.act2(x)

        # ---------------- patch embedding ----------------
        if patch_size > 1:
            if not dual_patchnorm:
                self.init_conv = nn.Conv2d(
                    channels, init_dim, kernel_size=patch_size, stride=patch_size
                )
            else:
                input_channels = channels * (patch_size**2)
                self.init_conv = nn.Sequential(
                    Rearrange("b c (h p1) (w p2) -> b h w (c p1 p2)", p1=patch_size, p2=patch_size),
                    nn.LayerNorm(input_channels),
                    nn.Linear(input_channels, init_dim),
                    nn.LayerNorm(init_dim),
                    Rearrange("b h w c -> b c h w"),
                )
        else:
            self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=7, padding=3)

        # ---------------- 通道维度规划 ----------------
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ---------------- 时间嵌入 ----------------
        time_dim = dim * 4
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---------------- 编码器（ResNet + Attention） ----------------
        self.down_blocks = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.down_blocks.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    LinearAttention(dim_in),
                    nn.Conv2d(dim_in, dim_out, kernel_size=1)  # 只改变通道数，不改变分辨率
                ])
            )

        # ---------------- ViT中间块及其上下采样 ----------------
        mid_dim = dims[-1]
        
        # ViT前的下采样，减小特征图尺寸，但仅下采样一次以保留更多细节
        self.pre_vit_downsample = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.SiLU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),  # 保持32x32，增强特征
            nn.SiLU(),
        )
        
        # ViT处理块
        self.vit = Transformer(
            dim=mid_dim,
            time_cond_dim=time_dim,
            depth=vit_depth,
            dim_head=attn_dim_head,
            heads=attn_heads,
            ff_mult=ff_mult,
            dropout=vit_dropout,
        )
        
        # ViT后的上采样，恢复特征图尺寸，对应单次下采样
        self.post_vit_upsample = nn.Sequential(
            nn.ConvTranspose2d(mid_dim, mid_dim, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.SiLU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),  # 增强特征融合
            nn.SiLU(),
        )

        #decoder
        self.up_blocks = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out):
            self.up_blocks.append(
                nn.ModuleList([
                    nn.Conv2d(dim_out, dim_in, kernel_size=1),  # 只改变通道数，不改变分辨率
                    ResnetBlock(dim_in * 2, dim_in, time_emb_dim=time_dim),
                    ResnetBlock(dim_in * 2, dim_in, time_emb_dim=time_dim),
                    LinearAttention(dim_in),
                ])
            )

        #output head
        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_dim)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(init_dim, init_dim*2, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(init_dim*2),  # 添加残差连接保留细节
            nn.Conv2d(init_dim*2, init_dim*4, 3, padding=1, stride=2),  # 64x64 -> 32x32
            nn.SiLU(),
            ResidualBlock(init_dim*4),  # 添加残差连接保留细节
            nn.Conv2d(init_dim*4, init_dim*8, 3, padding=1, stride=2),  # 32x32 -> 16x16
            nn.SiLU(),
            ResidualBlock(init_dim*8),  # 添加残差连接保留细节
            nn.AdaptiveAvgPool2d((8, 8)), 
            nn.Flatten(),
            nn.Linear(init_dim*8*64, 768),  # 64=8x8
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.SiLU()
        )

        # ---------------- 张量生成器 ----------------
        self.tensor_generator = nn.Sequential(
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, max_tensor_len * feature_dim),
        )

        # ---------------- 掩码生成器 ----------------
        self.mask_generator = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, max_tensor_len),
        )
        nn.init.zeros_(self.mask_generator[-1].bias)  # 初始化为零，有助于训练稳定性

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        """
        Args:
            x    : (B, C, H, W)  输入图像张量，数值范围 [0,1], 预期尺寸为64x64x1
            time : (B,)          连续时间 t ∈ [0,1]

        Returns:
            raw_tensor  : (B, max_len, feature_dim)
            mask_tensor : (B, max_len, 1)  未经过Sigmoid的logits
        """
        # -------- 输入预处理 --------
        x = self.init_img_transform(x)       # 用户自定义预处理
        x = self.init_conv(x)                # patch / conv
        residual = x.clone()                 # 用于最后 skip-connect

        # -------- 时间嵌入 --------
        t = self.time_mlp(time)

        # -------- 编码器 --------
        skip = []
        for block1, block2, attn, conv_down in self.down_blocks:
            x = block1(x, t)
            skip.append(x)
            x = block2(x, t)
            x = attn(x)
            skip.append(x)
            x = conv_down(x)  # 只改变通道数，不改变分辨率

        # -------- ViT预处理 --------
        # 只进行一次下采样，保留更多细节：64x64 -> 32x32
        x_before_vit = x.clone()  # 保存ViT前的特征，用于额外skip连接
        x = self.pre_vit_downsample(x)  
        
        # -------- ViT --------
        x = rearrange(x, "b c h w -> b h w c")
        x, ps = pack([x], "b * c")
        x = self.vit(x, t)
        (x,) = unpack(x, ps, "b * c")
        x = rearrange(x, "b h w c -> b c h w")
        
        # -------- ViT后处理 --------
        x = self.post_vit_upsample(x)  # 32x32 -> 64x64
        
        # 添加额外的残差连接，进一步保留细节
        x = x + x_before_vit

        # -------- 解码器 --------
        for conv_up, block1, block2, attn in self.up_blocks:
            x = conv_up(x)  # 只改变通道数，不改变分辨率
            x = torch.cat((x, skip.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, skip.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

        # -------- 输出头 --------
        x = torch.cat((x, residual), dim=1)
        x = self.final_res_block(x, t)

        # -------- 特征提取与张量生成 --------
        features = self.feature_extractor(x)
        
        raw_tensor = self.tensor_generator(features).view(
            -1, self.max_tensor_len, self.feature_dim
        )
        mask_tensor = self.mask_generator(features).unsqueeze(-1)

        return raw_tensor, mask_tensor


    @property
    def device(self):
        return next(self.parameters()).device




class TensorDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        image_size: int,
        max_tensor_len: int = 165,
        feature_dim: int = 3,
        channels: int = 1,
        pred_objective: str = "v",
        noise_schedule: Callable[[torch.Tensor], torch.Tensor] = logsnr_schedule_cosine,
        noise_d: Optional[float] = None,
        noise_d_low: Optional[float] = None,
        noise_d_high: Optional[float] = None,
        num_sample_steps: int = 250,
        clip_sample_denoised: bool = True,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
    ) -> None:
        super().__init__()
        assert pred_objective in {"v", "eps"}, "pred_objective must be 'v' or 'eps'"
        self.model = model
        self.channels = channels
        self.image_size = image_size
        self.max_tensor_len = max_tensor_len
        self.feature_dim = feature_dim
        self.pred_objective = pred_objective
        self._debug_printed = False

        # build log_snr schedule
        self.log_snr = noise_schedule
        if noise_d is not None:
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)
        if noise_d_low is not None or noise_d_high is not None:
            assert noise_d_low is not None and noise_d_high is not None, \
                "noise_d_low and noise_d_high must both be provided"
            self.log_snr = logsnr_schedule_interpolated(
                self.log_snr, image_size, noise_d_low, noise_d_high
            )

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def load_npy_image(self, npy_path: str) -> torch.Tensor:
        image_data = torch.from_numpy(np.load(npy_path)).float()
        if image_data.ndim == 3:
            image_data = image_data.unsqueeze(0)
        if image_data.shape[1] != self.channels:
            image_data = image_data.permute(0, 3, 1, 2)
        if image_data.max() > 1.0:
            image_data = image_data / 255.0
        _, _, h, w = image_data.shape
        if (h, w) != (self.image_size, self.image_size):
            image_data = F.interpolate(
                image_data, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False
            )
        return image_data.to(self.device)

    def q_sample(
        self,
        x_start: torch.Tensor,
        times: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ):
        noise = noise if noise is not None else torch.randn_like(x_start)
        log_snr = self.log_snr(times)
        log_snr_pad = right_pad_dims_to(x_start, log_snr)
        alpha = sqrt(log_snr_pad.sigmoid())
        sigma = sqrt((-log_snr_pad).sigmoid())
        return x_start * alpha + noise * sigma, log_snr, noise

    def p_mean_variance(
        self,
        img: torch.Tensor,
        x: torch.Tensor,
        time: torch.Tensor,
        time_next: torch.Tensor,
    ):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        sq_alpha, sq_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        sq_sigma, sq_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()
        alpha, sigma, alpha_nxt = map(sqrt, (sq_alpha, sq_sigma, sq_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred_tensor, _ = self.model(img, batch_log_snr)

        if self.pred_objective == "v":
            x_start = alpha * x - sigma * pred_tensor
        else:
            x_start = (x - sigma * pred_tensor) / alpha
        x_start = x_start.clamp(-1.0, 1.0)

        model_mean = alpha_nxt * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = sq_sigma_next * c
        return model_mean, posterior_variance

    def forward(
        self,
        *,
        img_tensor: Optional[torch.Tensor] = None,
        target_tensor: Optional[torch.Tensor] = None,
    ):
        img = img_tensor
        
        if not self._debug_printed:
            # 打印real shape
            img_shape = img_tensor.shape if img_tensor is not None else None
            tgt_shape = target_tensor.shape if target_tensor is not None else None
            print(f"[TensorDiffusion] img_tensor.shape={img_shape}, "
                  f"target_tensor.shape={tgt_shape}")
            branch = "training branch" if target_tensor is not None else "sampling branch"
            print(f"[TensorDiffusion] → 走 {branch}")
            self._debug_printed = True
        B, C, H, W = img.shape
        assert (H, W) == (self.image_size, self.image_size), \
            f"Input image must be {self.image_size}x{self.image_size}"

        # sampling mode
        if target_tensor is None:
            return self.p_sample_loop(img)

        # training mode
        times = torch.rand(B, device=self.device)
        noise = torch.randn_like(target_tensor)

        x_noised, log_snr, noise_used = self.q_sample(target_tensor, times, noise=noise)
        pred_tensor, _ = self.model(img, log_snr)

        padded_log_snr = right_pad_dims_to(x_noised, log_snr)
        if self.pred_objective == "v":
            alpha = sqrt(padded_log_snr.sigmoid())
            sigma = sqrt((-padded_log_snr).sigmoid())
            target = alpha * noise_used - sigma * target_tensor
        else:
            target = noise_used

        loss = F.mse_loss(pred_tensor, target, reduction="mean")
        return {"loss": loss, "tensor_mse": loss}

    @torch.no_grad()
    def p_sample(
        self,
        img: torch.Tensor,
        x: torch.Tensor,
        time: torch.Tensor,
        time_next: torch.Tensor,
    ):
        model_mean, model_var = self.p_mean_variance(img, x, time, time_next)
        if time_next == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, img: torch.Tensor):
        B = img.shape[0]
        x = torch.randn(B, self.max_tensor_len, self.feature_dim, device=self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)
        for i in range(self.num_sample_steps):
            t, t_next = steps[i], steps[i + 1]
            x = self.p_sample(img, x, t, t_next)
        return x.clamp(-1.0, 1.0)


