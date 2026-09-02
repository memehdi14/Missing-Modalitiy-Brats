"""
Masked-Fusion Transformer for missing-modality 3D brain tumour segmentation.

This is the full model that wires together:
  MultiModalityTokenizer (stem.py) → TransformerEncoder (transformer_block.py)
  → aggregate + reshape → UNet3DDecoder (decoder.py)

Forward pass summary
────────────────────
1. Run all 4 modality stems in parallel to extract multi-scale features.
2. Tokenise the bottleneck features (6³ per modality → 216 tokens each).
3. Add 3D positional + modality-type embeddings.
4. Build the attention mask: absent-modality tokens are blocked from the
   attention computation entirely (they cannot attend or be attended to).
5. Run the stacked transformer encoder.
6. Aggregate the transformer output across present modalities (presence-gated
   average), then reshape back to a 3D feature map (B, embed_dim, 6, 6, 6).
7. Decode back to full resolution using U-Net skip connections.

Why this beats the zero-fill U-Net baseline
────────────────────────────────────────────
Zero-fill baseline: absent modality = channel of zeros. The model must learn
to ignore these zeros through weight learning alone. There is no explicit
mechanism to distinguish "truly absent" from "actually zero tissue intensity".
The model also cannot attend across modalities adaptively.

This model:
  • Absent tokens never enter the attention computation — no information leakage.
  • Modality-type embeddings let the model explicitly reason about which modality
    each token came from, enabling cross-modal attention to learn which pairs of
    modalities provide complementary information (e.g., T1ce highlights
    enhancing tumour, FLAIR highlights oedema).
  • Skip connections are presence-gated, so absent modality stems don't
    corrupt the decoder's high-resolution features.

Expected result: the biggest gains over the baseline should appear in the
single-modality and two-modality evaluation combos, where the baseline's
zero-fill approach is most harmful.
"""

import torch
import torch.nn as nn

from models.stem import MultiModalityTokenizer
from models.transformer_block import TransformerEncoder
from models.decoder import UNet3DDecoder, fuse_skip_features


class MaskedFusionTransformer(nn.Module):
    """
    Full model.

    Default hyperparameters are tuned for:
      - Patch size 96³ (bottleneck: 6³, 216 tokens/modality, 864 total)
      - A5000 laptop 16GB VRAM with AMP (batch_size=1, gradient accumulation)
      - embed_dim=192 divisible by 3 (for factored 3D pos emb) and by n_heads=6

    Args:
        n_modalities   : number of input modalities (4 for BraTS)
        stem_channels  : base channel count for each modality stem
        embed_dim      : transformer hidden dimension
        n_heads        : attention heads
        n_layers       : number of transformer encoder blocks
        dropout        : dropout rate (applied in attention and FFN)
        out_channels   : segmentation output channels (3: WT, TC, ET)
    """

    def __init__(
        self,
        n_modalities: int = 4,
        stem_channels: int = 24,
        embed_dim: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        dropout: float = 0.1,
        out_channels: int = 3,
    ):
        super().__init__()
        assert embed_dim % 3 == 0, "embed_dim must be divisible by 3 (factored 3D pos emb)"
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.n_modalities = n_modalities
        self.embed_dim = embed_dim

        # --- Tokenizer (4 stems + projection + pos emb + mod type emb) ---
        self.tokenizer = MultiModalityTokenizer(
            n_modalities=n_modalities,
            stem_channels=stem_channels,
            embed_dim=embed_dim,
        )

        # --- Transformer encoder ---
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            ffn_dim=embed_dim * 4,
            n_layers=n_layers,
            dropout=dropout,
        )

        # --- Decoder ---
        stem_channel_dims = self.tokenizer.stems[0].channel_dims  # [C, 2C, 4C, 4C, 4C]
        self.decoder = UNet3DDecoder(
            embed_dim=embed_dim,
            stem_channel_dims=stem_channel_dims,
            out_channels=out_channels,
        )

    def forward(
        self,
        image: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image        : (B, 4, P, P, P) — all 4 modality channels; absent are zero-filled
            presence_mask: (B, 4) float — 1.0 = present modality, 0.0 = absent

        Returns:
            logits       : (B, 3, P, P, P) — raw logits for WT, TC, ET
                           Apply sigmoid at evaluation time; loss function handles it internally.
        """
        # ── Step 1 & 2 & 3 & 4: stems + tokenise + pos/type emb + attn mask ──
        tokens, attn_mask, all_stem_feats, (Hf, Wf, Df), N_per_mod = (
            self.tokenizer(image, presence_mask)
        )
        # tokens  : (B, 4*N, embed_dim)
        # attn_mask: (B, 1, 4*N, 4*N) bool

        # ── Step 5: transformer encoder ──
        out_tokens = self.encoder(tokens, attn_mask=attn_mask)
        # out_tokens: (B, 4*N, embed_dim)

        # ── Step 6: aggregate transformer output across present modalities ──
        # Split back into per-modality chunks: each (B, N, embed_dim)
        mod_outs = out_tokens.reshape(
            out_tokens.shape[0], self.n_modalities, N_per_mod, self.embed_dim
        )  # (B, 4, N, embed_dim)

        # Presence-gated average: absent modality outputs contribute nothing
        pres = presence_mask[:, :, None, None].float()  # (B, 4, 1, 1)
        mod_outs_gated = mod_outs * pres                 # (B, 4, N, embed_dim)
        n_present = pres.sum(dim=1).clamp(min=1.0)       # (B, 1, 1)
        agg = mod_outs_gated.sum(dim=1) / n_present      # (B, N, embed_dim)

        # Reshape to spatial: (B, embed_dim, Hf, Wf, Df) = (B, 192, 6, 6, 6)
        bottleneck = agg.transpose(1, 2).reshape(
            agg.shape[0], self.embed_dim, Hf, Wf, Df
        )

        # ── Step 7: presence-gated skip connections from stems ──
        # Fuse across modalities at each of the 4 spatial scales (skip scale indices 0..3)
        fused_skips = [
            fuse_skip_features(all_stem_feats, presence_mask, scale_idx=s)
            for s in range(4)
        ]

        # ── Step 8: decode ──
        logits = self.decoder(bottleneck, fused_skips)
        return logits


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly: python models/full_model.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sanity check on {device}")

    model = MaskedFusionTransformer().to(device)

    B, P = 1, 96
    image = torch.randn(B, 4, P, P, P, device=device)

    # Simulate a case where T1 and FLAIR are present, T1ce and T2 are absent
    presence_mask = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=device)

    with torch.no_grad():
        logits = model(image, presence_mask)

    print(f"  Input:  {tuple(image.shape)}")
    print(f"  Output: {tuple(logits.shape)}")
    assert logits.shape == (B, 3, P, P, P), f"unexpected shape: {logits.shape}"
    print("  ✓ Forward pass OK")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")
