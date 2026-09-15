"""Phase 2.2: per-layer projector, mapping ViT dim (768) -> LLM hidden (1536).

One instance per insertion layer, all identical in size. Trainable.
"""
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim=768, out_dim=1536, hidden=None):
        super().__init__()
        hidden = hidden or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):  # [B, N, in_dim] -> [B, N, out_dim]
        return self.net(x)


class PerceiverResampler(nn.Module):
    """Optional shared resampler: N image tokens -> num_latents tokens.

    Default OFF. Shared across insertion layers (projectors stay per-layer).
    """
    def __init__(self, dim=768, num_latents=64, num_heads=8):
        super().__init__()
        import torch
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):  # [B, N, dim] -> [B, num_latents, dim]
        b = x.size(0)
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return self.ln(out)
