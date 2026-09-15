"""Phase 2.3: Flamingo-style gated cross-attention block.

text = query, image tokens = key/value. Zero-init tanh gates make the block a no-op
at step 0, so training starts as the untouched frozen LLM. LayerNorms and gates run
in fp32 for fp16 stability.
"""
import torch
import torch.nn as nn


class GatedCrossAttn(nn.Module):
    def __init__(self, embed_dim=1536, num_heads=16, ffn_mult=4):
        super().__init__()
        self.ln_q = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(1))  # attention gate
        self.ln_f = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_mult * embed_dim),
            nn.GELU(),
            nn.Linear(ffn_mult * embed_dim, embed_dim),
        )
        self.beta = nn.Parameter(torch.zeros(1))   # ffn gate

    def forward(self, h_text, img_tokens, img_key_padding_mask=None):
        """h_text: [B, T, D]  img_tokens: [B, N, D] (already projected to D).

        img_key_padding_mask: [B, N] with True where an image key should be ignored
        (used when a batch mixes images with different token counts; usually None).
        Returns [B, T, D].
        """
        # LayerNorms/gates in fp32 for fp16 stability; attention math follows autocast.
        q = self.ln_q(h_text.float()).to(h_text.dtype)
        a, _ = self.attn(q, img_tokens, img_tokens,
                         key_padding_mask=img_key_padding_mask, need_weights=False)
        gate_a = torch.tanh(self.alpha.float()).to(h_text.dtype)
        h = h_text + gate_a * a

        f_in = self.ln_f(h.float()).to(h.dtype)
        f = self.ffn(f_in)
        gate_b = torch.tanh(self.beta.float()).to(h.dtype)
        h = h + gate_b * f
        return h

    @torch.no_grad()
    def gate_magnitudes(self):
        """(tanh(alpha), tanh(beta)) as python floats — for logging / model card."""
        return (float(torch.tanh(self.alpha)), float(torch.tanh(self.beta)))
