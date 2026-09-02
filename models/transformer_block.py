"""
Transformer encoder block with Pre-LayerNorm.

Design choices (and why each matters for stability / publishability):

Pre-LayerNorm (vs Post-LN):
  Post-LN (original "Attention is All You Need") normalizes AFTER the residual add.
  This makes early training unstable — gradients can explode or vanish because the
  residual stream is unnormalized going into the next block.
  Pre-LN normalizes BEFORE the sub-layer (LN → sub-layer → + residual). This keeps
  the residual stream on a predictable scale and allows training without LR warmup
  tricks. Almost all modern transformers (GPT-2, LLaMA, ViT-22B) use Pre-LN or
  a variant. We do the same.

GELU (vs ReLU) in the FFN:
  GELU is a smoother activation that allows small negative values through, which helps
  gradient flow and typically improves convergence speed for transformers.

FFN dimension = 4 × embed_dim:
  This is the standard ratio from "Attention is All You Need" and has held up across
  architectures. It works well empirically for our scale.

Final LayerNorm after all blocks:
  Ensures the output of the encoder is normalized before being projected back to
  spatial features in the decoder, improving training stability.
"""

import torch
import torch.nn as nn
from models.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise FFN: Linear → GELU → Dropout → Linear → Dropout.
    Applied identically to each token position independently.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Pre-LN transformer encoder block.

    Forward pass:
        residual = x
        x = LN(x) → MHA(x, attn_mask) → + residual
        residual = x
        x = LN(x) → FFN(x) → + residual
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """
        x        : (B, S, embed_dim)
        attn_mask: (B, 1, S, S) bool or None
        """
        # Attention sub-layer
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        # FFN sub-layer
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N TransformerBlocks followed by a final LayerNorm.
    The final norm ensures the encoder output is on a predictable scale
    before being reshaped and passed to the CNN decoder.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """
        x        : (B, S, embed_dim)
        attn_mask: (B, 1, S, S) bool or None

        Returns  : (B, S, embed_dim) — same shape
        """
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.final_norm(x)
