"""
Utility functions for CSDNet — clustering and EMA helpers.
"""
import torch
import torch.nn.functional as F
from inspect import isfunction


def exists(val):
    return val is not None

def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets=None):
    """Single iteration of hard cluster assignment + centroid update."""
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]
    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)
    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0
    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means
"""
CSDNet attention modules.

- IGPA: Iterative Global Prototype Attention (Eq.14)
  Refines cluster prototypes via center_iter, projects them to global K/V.

- IGA: Iterative Grouped Aggregation (Eq.18)
  Local-window self-attention + cross-attention to IGPA's global prototypes.

- LWI: Local Window Interaction
  Patch-based local self-attention for fine detail.

- Attention: standard multi-head self-attention.
- ConvFFN: feed-forward network with depthwise conv.
- PreNorm: LayerNorm → fn wrapper.
"""
import torch.nn as nn
from einops import rearrange



# ═════════════════════════════════════════════════════════════
# IGPA — Iterative Global Prototype Attention
# ═════════════════════════════════════════════════════════════

class IGPA(nn.Module):
    """Projects cluster prototypes to global K/V. Q comes from IGA."""
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        if self.training:
            x_global = center_iter(F.normalize(normed_x, dim=-1),
                                   F.normalize(x_means, dim=-1))
        else:
            x_global = x_means
        k = self.to_k(x_global)
        v = self.to_v(x_global)
        k = rearrange(k, 'n (h d) -> h n d', h=self.heads)
        v = rearrange(v, 'n (h d) -> h n d', h=self.heads)
        return k, v, x_global.detach()


# ═════════════════════════════════════════════════════════════
# IGA — Iterative Grouped Aggregation
# ═════════════════════════════════════════════════════════════

class IGA(nn.Module):
    """
    Grouped attention: local-window self-attention within groups
    + cross-attention to global prototypes from IGPA.
    """
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        B, N, _ = normed_x.shape

        q = self.to_q(normed_x)
        k = self.to_k(normed_x)
        v = self.to_v(normed_x)

        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        gs = min(N, self.group_size)
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        paded_q = torch.cat((q, torch.flip(q[:, N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d",
                            ng=ng, h=self.heads)

        paded_k = torch.cat((k, torch.flip(k[:, N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2, 2*gs, gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d", h=self.heads)

        paded_v = torch.cat((v, torch.flip(v[:, N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2, 2*gs, gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d", h=self.heads)

        # Grouped local attention
        out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)

        # Global prototype attention
        k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
        v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)
        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)

        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)
        return out


# ═════════════════════════════════════════════════════════════
# LWI — Local Window Interaction
# ═════════════════════════════════════════════════════════════

def patch_divide(x, step, ps):
    """Divide feature map into overlapping patches."""
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    for i in range(0, h + step - ps, step):
        top = i; down = i + ps
        if down > h: top, down = h - ps, h
        for j in range(0, w + step - ps, step):
            left = j; right = j + ps
            if right > w: left, right = w - ps, w
            crop_x.append(x[:, :, top:down, left:right])
    nh = (h + step - ps - 1) // step + 1
    nw = (w + step - ps - 1) // step + 1
    crop_x = torch.stack(crop_x, 1)
    return crop_x.contiguous(), nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches with overlapping blending."""
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    count  = torch.zeros((b, h, w), device=x.device)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i; down = i + ps
        if down > h: top, down = h - ps, h
        for j in range(0, w + step - ps, step):
            left = j; right = j + ps
            if right > w: left, right = w - ps, w
            output[:, :, top:down, left:right] += crop_x[:, index]
            count[:, top:down, left:right] += 1
            index += 1
    return output / count.unsqueeze(1)


class Attention(nn.Module):
    """Standard multi-head self-attention."""
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      (q, k, v))
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LWI(nn.Module):
    """
    Patch-based local window interaction.
    Splits feature map into overlapping patches, applies
    self-attention per patch, then merges with blending.
    """
    def __init__(self, dim, qk_dim, mlp_dim, heads):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads, qk_dim))
        self.ff   = PreNorm(dim, ConvFFN(dim, mlp_dim))

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        crop_x = self.attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c -> b n c h w', n=n, w=pw)

        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


# ═════════════════════════════════════════════════════════════
# ConvFFN & PreNorm
# ═════════════════════════════════════════════════════════════

class dwconv(nn.Module):
    """Depthwise convolution applied after (B,N,C) → (B,C,H,W) reshape."""
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
        self.hidden_features = hidden_features
        self.depthwise_conv = nn.Conv2d(
            hidden_features, hidden_features,
            kernel_size, padding=kernel_size // 2,
            groups=hidden_features)

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features,
                                    x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    """Feed-forward network: Linear → GELU → dwconv → Linear."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class PreNorm(nn.Module):
    """LayerNorm applied before fn."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
"""
Frequency Structure Encoder (FSE).

Decomposes local 3×3 patches into DCT frequency bands, encodes low and high
frequencies through separate MLP branches, then fuses them via a cross-frequency
interaction layer (Φ_cross) with residual connection.
"""
import math


class FrequencyStructureEncoder(nn.Module):
    """
    Lightweight structure descriptor extractor.

    1. 3×3 sliding window → DCT projection → 9 frequency coefficients
    2. Split into low-frequency (4 bases) and high-frequency (5 bases) groups
    3. Each group encoded independently → half the structure dimension
    4. Cross-frequency interaction (Φ_cross): 2-layer MLP with residual
    5. LayerNorm + learnable scale (≥ 0.5)

    Args:
        in_channels:   input feature channels
        structure_dim: output descriptor dimension (default 16)
        kernel_size:   DCT window size (3 → 3×3 → 9 coefficients)
    """
    LOW_FREQ_INDICES  = [0, 1, 3, 4]       # (0,0), (0,1), (1,0), (1,1)
    HIGH_FREQ_INDICES = [2, 5, 6, 7, 8]

    def __init__(self, in_channels, structure_dim=16, kernel_size=3):
        super().__init__()
        self.in_channels   = in_channels
        self.structure_dim = structure_dim
        self.kernel_size   = kernel_size
        self.padding       = kernel_size // 2

        dct_basis = self._build_dct_basis(kernel_size)
        self.register_buffer('dct_basis', dct_basis)

        n_low  = len(self.LOW_FREQ_INDICES)
        n_high = len(self.HIGH_FREQ_INDICES)
        half_dim = structure_dim // 2

        # Independent frequency branch encoders
        self.low_freq_mlp = nn.Sequential(
            nn.Linear(n_low * in_channels, structure_dim),
            nn.GELU(),
            nn.Linear(structure_dim, half_dim),
        )
        self.high_freq_mlp = nn.Sequential(
            nn.Linear(n_high * in_channels, structure_dim),
            nn.GELU(),
            nn.Linear(structure_dim, half_dim),
        )

        # Cross-frequency interaction (Φ_cross): 2-layer MLP with residual
        self.cross_freq_proj = nn.Sequential(
            nn.Linear(structure_dim, structure_dim),
            nn.GELU(),
            nn.Linear(structure_dim, structure_dim),
        )

        self.norm      = nn.LayerNorm(structure_dim)
        self.scale_raw = nn.Parameter(torch.tensor(0.5))
        self._init_weights()

    def _build_dct_basis(self, k):
        """Construct 2D DCT basis matrix (k², k²) via Kronecker product."""
        dct_1d = torch.zeros(k, k)
        for i in range(k):
            for j in range(k):
                if i == 0:
                    dct_1d[i, j] = math.sqrt(1.0 / k)
                else:
                    dct_1d[i, j] = math.sqrt(2.0 / k) * math.cos(
                        math.pi * i * (2 * j + 1) / (2 * k))
        return torch.kron(dct_1d, dct_1d)

    def _init_weights(self):
        for mlp in [self.low_freq_mlp, self.high_freq_mlp, self.cross_freq_proj]:
            for m in mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size

        # Unfold → DCT projection → 9 frequency coefficients per pixel
        patches = F.unfold(x, kernel_size=k, padding=self.padding, stride=1)
        patches = patches.permute(0, 2, 1).reshape(B, H * W, C, k * k)
        freq_features = torch.matmul(patches, self.dct_basis.t())

        # Group into low and high frequency
        low  = freq_features[:, :, :, self.LOW_FREQ_INDICES].reshape(B, H * W, -1)
        high = freq_features[:, :, :, self.HIGH_FREQ_INDICES].reshape(B, H * W, -1)

        # Independent branch encoding
        low_enc  = self.low_freq_mlp(low)     # (B, N, half_dim)
        high_enc = self.high_freq_mlp(high)   # (B, N, half_dim)

        # Cross-frequency interaction with residual
        combined = torch.cat([low_enc, high_enc], dim=-1)
        structure_encoding = combined + self.cross_freq_proj(combined)

        # Normalize and scale
        structure_encoding = self.norm(structure_encoding)
        scale = 0.5 + F.softplus(self.scale_raw)
        structure_encoding = structure_encoding * scale

        structure_encoding = structure_encoding.permute(0, 2, 1).reshape(
            B, self.structure_dim, H, W)
        return structure_encoding
"""
CSDNet: Content-Structure Dual-aware Network for Image Super-Resolution.

Core architecture:
  - FSE (Frequency Structure Encoder): extracts local structure descriptors via DCT
  - DAB (Dynamic Aggregation Block): core transformer block with dual-aware routing
  - StructureFusionModule: fuses content features with structure descriptors

Each DAB contains:
  IGPA → prototype refinement → global K/V
  IGA  → grouped local attention + cross-attention to global prototypes
  StructureFusion → content + α · structure
  LWI  → patch-based local window interaction
  ConvFFN → feed-forward network
"""






# ═════════════════════════════════════════════════════════
# Structure Fusion Module
# ═════════════════════════════════════════════════════════

class StructureFusionModule(nn.Module):
    """
    Fuses content features x with FSE structure descriptors s via
    a learnable mixing weight α (sigmoid-gated, lower bound 0.01).
    """
    def __init__(self, dim, structure_dim):
        super().__init__()
        self.norm_s = nn.LayerNorm(structure_dim)
        self.proj_x = nn.Linear(dim, dim, bias=False)
        self.proj_s = nn.Linear(structure_dim, dim, bias=False)
        self.fusion_weight_raw = nn.Parameter(torch.tensor(-2.0))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x_flat, s_flat):
        s_normed  = self.norm_s(s_flat)
        content   = self.proj_x(x_flat)
        structure = self.proj_s(s_normed)
        alpha     = 0.01 + torch.sigmoid(self.fusion_weight_raw)
        fused     = content + alpha * structure
        return self.post_norm(fused)


# ═════════════════════════════════════════════════════════
# DAB — Dynamic Aggregation Block
# ═════════════════════════════════════════════════════════

class DAB(nn.Module):
    """
    One transformer block of CSDNet.

    Flow:
      1. IGPA: cluster prototypes → global K/V
      2. Dual-aware routing: content similarity + β · structure similarity
      3. IGA: local grouped attention + global prototype cross-attention
      4. Structure-guided fusion (FSE descriptors injected via α)
      5. LWI: patch-based local window interaction
      6. ConvFFN: feed-forward with depthwise conv

    Block-specific FSE can be shared across blocks (share_fse='all')
    or independent per block (share_fse='none').

    Args:
        dim:              feature dimension
        qk_dim:           Q/K projection dimension
        mlp_dim:          FFN hidden dimension
        heads:            number of attention heads
        n_iter:           prototype refinement iterations
        num_tokens:       number of cluster prototypes
        group_size:       IGA grouping size
        patch_size:       LWI patch size
        ema_decay:        prototype EMA decay
        use_structure:    enable structure-aware routing + fusion
        structure_dim:    FSE output dimension
        structure_kernel: FSE DCT window size
        shared_fse:       pre-constructed FSE module (for sharing)
    """
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128, patch_size=16,
                 ema_decay=0.999,
                 use_structure=True, structure_dim=16,
                 structure_kernel=3, shared_fse=None):
        super().__init__()

        self.n_iter     = n_iter
        self.ema_decay  = ema_decay
        self.num_tokens = num_tokens
        self.use_structure = use_structure
        self.patch_size = patch_size

        self.norm = nn.LayerNorm(dim)
        self.mlp  = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.igpa = IGPA(dim, qk_dim, heads)
        self.iga  = IGA(dim, qk_dim, heads, group_size)
        self.lwi  = LWI(dim, qk_dim, mlp_dim, heads)

        # Content prototype (EMA-updated)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))

        if use_structure:
            self.register_buffer('structure_means',
                                 torch.randn(num_tokens, structure_dim))
            if shared_fse is not None:
                self.fse = shared_fse
            else:
                self.fse = FrequencyStructureEncoder(
                    dim, structure_dim, structure_kernel)
            self.structure_fusion = StructureFusionModule(dim, structure_dim)
            self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)

        # Initialize or retrieve prototypes
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat(
                (x, torch.flip(x[:, N-pad_n:N, :], dims=[-2])), dim=-2)
            x_means = torch.mean(
                rearrange(paded_x, 'b (cnt n) c -> cnt (b n) c',
                          cnt=self.num_tokens), dim=-2).detach()
        else:
            x_means = self.means.detach()

        # Iterative prototype refinement (no grad for warm-up iterations)
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(
                        F.normalize(x, dim=-1),
                        F.normalize(x_means, dim=-1))

        # IGPA: prototypes → global K/V
        k_global, v_global, x_means = self.igpa(x, x_means)

        # Compute token → prototype assignment
        with torch.no_grad():
            x_scores = torch.einsum(
                'b i c, j c -> b i j',
                F.normalize(x, dim=-1),
                F.normalize(x_means, dim=-1))

            if self.use_structure:
                # Extract structure descriptors
                feat_2d = rearrange(residual, 'b (h w) c -> b c h w', h=h, w=w)
                s_map   = self.fse(feat_2d)
                s_flat  = rearrange(s_map, 'b c h w -> b (h w) c')

                # Dual-aware routing: add β · structure similarity
                s_scores = torch.einsum(
                    'b i c, j c -> b i j',
                    F.normalize(s_flat, dim=-1),
                    F.normalize(self.structure_means, dim=-1))
                x_scores = x_scores + self.beta.detach() * s_scores

            x_belong_idx = torch.argmax(x_scores, dim=-1)
            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1,
                                    index=idx).unsqueeze(-1)

        # IGA: grouped attention
        y = self.iga(x, idx_last, k_global, v_global)

        # Structure-guided fusion
        if self.use_structure:
            s_flat = F.normalize(s_flat, dim=-1)
            y = self.structure_fusion(y, s_flat)

        y = rearrange(y, 'b (h w) c -> b c h w', h=h).contiguous()
        y = self.lwi(y, self.patch_size)

        x_flat = rearrange(y, 'b c h w -> b (h w) c')
        x = residual + x_flat
        x = self.mlp(x, x_size=(h, w)) + x

        # EMA update prototypes
        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    if self.use_structure:
                        self.structure_means.data.copy_(
                            s_flat.mean(dim=0, keepdim=True).expand(
                                self.num_tokens, -1))
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, new_means, self.ema_decay)
                    if self.use_structure:
                        s_mean_new = s_flat.mean(dim=0, keepdim=True)
                        ema_inplace(self.structure_means, s_mean_new,
                                    self.ema_decay)

        return rearrange(x, 'b (h w) c -> b c h w', h=h)


# ═════════════════════════════════════════════════════════
# CSDNet
# ═════════════════════════════════════════════════════════

class CSDNet(nn.Module):
    """
    Content-Structure Dual-aware Network for Image Super-Resolution.

    8 DAB blocks with shared or independent FSE, PixelShuffle upsampler,
    global residual connection with bilinear upsampled base.

    Args:
        upscale:          SR scale factor (2, 3, or 4)
        dim:              feature dimension (default 40)
        block_num:        number of DAB blocks (default 8)
        qk_dim:           Q/K projection dimension (default 36)
        mlp_dim:          FFN hidden dimension (default 96)
        heads:            attention heads (default 4)
        n_iters:          per-block clustering iterations
        num_tokens:       prototypes per block
        group_size:       IGA group size per block
        patch_size:       LWI patch size per block
        use_structure:    enable FSE + structure routing + fusion
        structure_dim:    FSE output dimension (default 16)
        structure_kernel: DCT window size (default 3)
        share_fse:        'all' | 'quarter' | 'none'
    """
    def __init__(self, upscale=4, dim=40, block_num=8, qk_dim=36,
                 mlp_dim=96, heads=4,
                 n_iters=(5, 5, 5, 5, 5, 5, 5, 5),
                 num_tokens=(16, 32, 64, 128, 16, 32, 64, 128),
                 group_size=(256, 128, 64, 32, 256, 128, 64, 32),
                 patch_size=(16, 20, 24, 28, 16, 20, 24, 28),
                 use_structure=True, structure_dim=16, structure_kernel=3,
                 share_fse='all'):
        super().__init__()

        self.dim         = dim
        self.block_num   = block_num
        self.upscale     = upscale
        self.use_structure = use_structure
        self.share_fse   = share_fse

        self.first_conv = nn.Conv2d(3, dim, 3, 1, 1)

        # Shared FSE modules
        shared_fses = self._create_shared_fse(
            use_structure, share_fse, dim, structure_dim, structure_kernel)

        # Build DAB blocks
        self.blocks    = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        for i in range(block_num):
            if use_structure and share_fse != 'none':
                if share_fse == 'all':
                    s_fse = shared_fses[0]
                elif share_fse == 'half':
                    s_fse = shared_fses[0] if i < 4 else shared_fses[1]
                elif share_fse == 'quarter':
                    s_fse = shared_fses[i // 2]
                else:
                    s_fse = None
            else:
                s_fse = None

            self.blocks.append(DAB(
                dim, qk_dim, mlp_dim, heads,
                n_iter=n_iters[i] if i < len(n_iters) else 3,
                num_tokens=num_tokens[i] if i < len(num_tokens) else 8,
                group_size=group_size[i] if i < len(group_size) else 128,
                patch_size=patch_size[i] if i < len(patch_size) else 16,
                use_structure=use_structure,
                structure_dim=structure_dim,
                structure_kernel=structure_kernel,
                shared_fse=s_fse,
            ))
            self.mid_convs.append(nn.Conv2d(dim, dim, 3, 1, 1))

        if use_structure and share_fse != 'none':
            self.shared_fse_modules = nn.ModuleList(shared_fses)

        # Upsampler
        if upscale == 4:
            self.upconv1 = nn.Conv2d(dim, dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(dim, dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale in (2, 3):
            self.upconv = nn.Conv2d(dim, dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.last_conv = nn.Conv2d(dim, 3, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _create_shared_fse(self, use_structure, share_fse, dim,
                           structure_dim, structure_kernel):
        if not use_structure or share_fse == 'none':
            return []
        n = {'all': 1, 'half': 2, 'quarter': 4}.get(share_fse, 0)
        return [FrequencyStructureEncoder(dim, structure_dim, structure_kernel)
                for _ in range(n)]

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(self.block_num):
            residual = self.blocks[i](x)
            x = x + self.mid_convs[i](residual)
        return x

    def forward(self, x):
        if x.max() > 1.0:
            x = x / 255.0

        base = F.interpolate(x, scale_factor=self.upscale,
                             mode='bilinear', align_corners=False)
        x = self.first_conv(x)
        x = self.forward_features(x) + x

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))

        out = base + self.last_conv(out)
        return out * 255.0
