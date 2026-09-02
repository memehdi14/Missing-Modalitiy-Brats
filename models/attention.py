"""
From-scratch scaled dot-product attention and multi-head attention.
Built with nn.Linear and torch.matmul ONLY — not nn.MultiheadAttention.

Key design choices (and why):
- Pre-projection (Q/K/V are separate linear layers) allows the model to learn
  different subspaces for querying, keying, and valuing.
- Attention mask: True = attend, False = block. Absent-modality tokens are blocked
  from both attending and being attended to, which is the core of masked fusion.
- nan_to_num on softmax output: if a token's entire row is masked (-inf), softmax
  produces NaN. We replace these with 0 so absent tokens produce zero output
  rather than corrupting gradients.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    """
    Core attention operation.

    Args:
        q, k, v : (B, n_heads, seq_len, head_dim)
        attn_mask: (B, 1, seq_len, seq_len) bool — True = keep, False = mask out.
                   None means attend everywhere.
        dropout_p: applied to attention weights during training only.

    Returns:
        out    : (B, n_heads, seq_len, head_dim)
        weights: (B, n_heads, seq_len, seq_len) — for inspection / debugging
    """
    head_dim = q.size(-1)
    scale = math.sqrt(head_dim)

    # (B, n_heads, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if attn_mask is not None:
        # masked_fill: where mask is False, set score to -inf so softmax → 0
        scores = scores.masked_fill(~attn_mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    # NaN arises when an entire row is -inf (absent-modality token attending nowhere).
    # Replace NaN with 0 — these tokens produce no output, which is correct.
    weights = torch.nan_to_num(weights, nan=0.0)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)

    out = torch.matmul(weights, v)
    return out, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention from scratch.

    Why separate Q/K/V projections rather than a single combined projection?
    It lets the model learn different representations for what it queries,
    what it keys on, and what it passes forward — standard transformer design.

    The attn_mask is constructed externally (in full_model.py) based on the
    presence_mask returned by the dataset, keeping this module general.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout

        # No bias on Q/K projections (common in ViT-style architectures)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform is stable for attention projections
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """
        Args:
            x        : (B, S, embed_dim)
            attn_mask: (B, 1, S, S) bool or None

        Returns:
            (B, S, embed_dim)
        """
        B, S, E = x.shape

        # Project and reshape into heads: (B, n_heads, S, head_dim)
        def project_and_split(proj, t):
            return proj(t).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = project_and_split(self.q_proj, x)
        k = project_and_split(self.k_proj, x)
        v = project_and_split(self.v_proj, x)

        dropout_p = self.dropout if self.training else 0.0
        attn_out, _ = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        # Merge heads: (B, n_heads, S, head_dim) → (B, S, embed_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, E)
        return self.out_proj(attn_out)
