"""
Contrastive learning models for jet image classification.

SimCLREncoder: backbone + projection head, trained with NT-Xent loss
MoCo: momentum contrast with queue of negatives
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights


class SwinEncoder(nn.Module):
    """Standard Swin-T backbone adapted for 3-channel jet images."""

    def __init__(self, embed_dim=256, pretrained=False):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        swin = swin_t(weights=weights)
        self.swin_features = swin.features
        self.swin_norm = swin.norm
        self.pool = nn.AdaptiveAvgPool2d(1)
        swin_out_dim = 768
        self.projector = nn.Sequential(
            nn.Linear(swin_out_dim, swin_out_dim),
            nn.ReLU(),
            nn.Linear(swin_out_dim, embed_dim),
        )

    def forward(self, x):
        h = self.swin_features(x)       # (B, H, W, C)
        h = self.swin_norm(h)
        h = h.permute(0, 3, 1, 2)       # (B, C, H, W)
        h = self.pool(h).flatten(1)      # (B, 768)
        z = self.projector(h)
        return h, z


class CustomSwinEncoder(nn.Module):
    """Custom lightweight Swin for 128x128 jet images.

    Adapted from arXiv:2602.00141: 3 stages (2,2,4 blocks),
    patch_size=4, window_size=4, ~86% fewer params than Swin-T.
    128x128 -> patch(4) -> 32x32 -> merge -> 16x16 -> merge -> 8x8
    """

    def __init__(self, in_channels=3, embed_dim=256, depths=(2, 2, 4),
                 num_heads=(3, 6, 12), base_dim=96, patch_size=4, window_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.window_size = window_size

        # Patch embedding: (B, 3, 128, 128) -> (B, 32, 32, base_dim)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([base_dim]),  # will be applied after permute
        )

        # Build stages
        dims = [base_dim * (2 ** i) for i in range(len(depths))]  # [96, 192, 384]
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        for i, (depth, nhead, dim) in enumerate(zip(depths, num_heads, dims)):
            # Transformer blocks for this stage
            stage = nn.ModuleList([
                SwinBlock(dim=dim, num_heads=nhead, window_size=window_size,
                          shift=(j % 2 == 1))
                for j in range(depth)
            ])
            self.stages.append(stage)

            # Patch merging (except last stage)
            if i < len(depths) - 1:
                self.merges.append(PatchMerging(dim))

        self.norm = nn.LayerNorm(dims[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.projector = nn.Sequential(
            nn.Linear(dims[-1], dims[-1]),
            nn.ReLU(),
            nn.Linear(dims[-1], embed_dim),
        )

        self._out_dim = dims[-1]  # 384

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed[0](x)       # (B, base_dim, H/p, W/p)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)        # (B, H, W, C)

        # Apply LN
        x = nn.functional.layer_norm(x, [C])

        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)
            if i < len(self.merges):
                x = self.merges[i](x)

        # (B, H', W', C_last) -> pool
        x = self.norm(x)
        x = x.reshape(x.size(0), -1, x.size(-1))  # (B, H'*W', C)
        h = self.pool(x.permute(0, 2, 1)).squeeze(-1)  # (B, C)
        z = self.projector(h)
        return h, z


class PatchMerging(nn.Module):
    """Merge 2x2 patches into 1, doubling channels."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""

    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.view(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        ).permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            # mask: (nW, ws*ws, ws*ws), attn: (B*nW, heads, ws*ws, ws*ws)
            nW = mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    """Swin Transformer block with optional shifted window."""

    def __init__(self, dim, num_heads, window_size=4, shift=False, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows
        x = self._window_partition(x, self.window_size)  # (num_windows*B, ws*ws, C)

        # Attention mask for shifted windows
        mask = None
        if self.shift_size > 0:
            mask = self._compute_mask(H, W, x.device)

        x = self.attn(x, mask=mask)

        # Merge windows
        x = self._window_reverse(x, self.window_size, H, W)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return x

    def _window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def _compute_mask(self, H, W, device):
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # Partition into windows: (nW, ws*ws, 1)
        nW = (H // self.window_size) * (W // self.window_size)
        mask_windows = img_mask.view(1, H // self.window_size, self.window_size,
                                      W // self.window_size, self.window_size, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(nW, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, ws*ws, ws*ws)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask


class ResNetEncoder(nn.Module):
    """Simple ResNet-18 backbone for comparison."""

    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        from torchvision.models import resnet18
        resnet = resnet18(weights=None)
        # Modify first conv for potentially different input
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    For a batch of N pairs, treats the other view of same image as positive
    and all other 2(N-1) views as negatives.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1, z2: (N, D) projected embeddings for two views
        Returns:
            scalar loss
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)

        # Cosine similarity matrix: (2N, 2N)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2 * N, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+N) and (i+N, i)
        pos_idx = torch.arange(2 * N, device=z.device)
        pos_idx[:N] += N
        pos_idx[N:] -= N

        # NT-Xent: cross entropy with positives as targets
        loss = F.cross_entropy(sim, pos_idx)
        return loss


class SegNetEncoderWrapper(nn.Module):
    """Wraps SegNet encoder to match the (h, z) interface for MoCo/SimCLR."""

    def __init__(self, segnet_model, embed_dim=256):
        super().__init__()
        # Copy only encoder parts from SegNet
        self.enc1 = segnet_model.enc1
        self.enc2 = segnet_model.enc2
        self.enc3 = segnet_model.enc3
        self.pool = segnet_model.pool
        self.fc_encode = segnet_model.fc_encode
        self.latent_dim = segnet_model.latent_dim

        self.projector = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, embed_dim),
        )

    @property
    def out_dim(self):
        return self.latent_dim

    def forward(self, x):
        x = self.enc1(x)
        x, _ = self.pool(x)
        x = self.enc2(x)
        x, _ = self.pool(x)
        x = self.enc3(x)
        x, _ = self.pool(x)
        x = x.view(x.size(0), -1)
        h = self.fc_encode(x)  # (B, 512)
        z = self.projector(h)  # (B, embed_dim)
        return h, z


class MoCo(nn.Module):
    """Momentum Contrast (MoCo v2) for jet image contrastive learning.

    Maintains a momentum-updated encoder and a queue of negative keys.
    """

    def __init__(self, encoder, embed_dim=256, queue_size=4096, momentum=0.999, temperature=0.1):
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.temperature = temperature
        self.momentum = momentum

        # Freeze key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Queue
        self.register_buffer("queue", F.normalize(torch.randn(embed_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - ptr
            keys = keys[:batch_size]
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_q, x_k):
        """
        Args:
            x_q: query images (N, 3, H, W)
            x_k: key images (N, 3, H, W) — different augmented view
        Returns:
            logits (N, 1+queue_size), labels (N,)
        """
        _, q = self.encoder_q(x_q)  # (N, D)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update()
            _, k = self.encoder_k(x_k)  # (N, D)
            k = F.normalize(k, dim=1)

        # Positive logits: (N, 1)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)
        # Negative logits: (N, queue_size)
        l_neg = torch.mm(q, self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)

        return logits, labels
