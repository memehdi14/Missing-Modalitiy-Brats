"""
Per-modality CNN stems and tokenization for the masked-fusion transformer.

Architecture overview:
  Each of the 4 MRI modalities gets its OWN CNN stem (separate weights, same architecture).
  This is critical: sharing weights across modalities would force T1 and FLAIR to produce
  features in the same representation space, but they encode fundamentally different
  tissue contrasts. Separate stems let the model learn modality-specific feature extractors.

  After the stem, each modality's bottleneck feature map (B, C, 6, 6, 6) is:
    1. Flattened into a spatial token sequence (B, 216, C)
    2. Projected to embed_dim (B, 216, D)
    3. Added 3D positional embeddings (shared across modalities — position in space
       is the same regardless of modality)
    4. Added modality-type embeddings (unique per modality — BERT-style token types)
       This is the key design choice that separates our model from zero-fill baselines:
       the transformer knows WHICH modality each token came from.

Spatial resolution (with patch_size=96, stem_channels=24):
  Stage 0:  (B, 24, 96, 96, 96)  ← skip connection 0
  Stage 1:  (B, 48, 48, 48, 48)  ← skip connection 1
  Stage 2:  (B, 96, 24, 24, 24)  ← skip connection 2
  Stage 3:  (B, 96, 12, 12, 12)  ← skip connection 3
  Bottleneck: (B, 96, 6, 6, 6)   ← tokenized → 216 tokens/modality
  Total tokens for 4 modalities: 864  (manageable for self-attention on 16GB VRAM)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks shared with decoder
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """Two conv3d → InstanceNorm → ReLU. Instance norm because batch size ≤ 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock3D(nn.Module):
    """Stride-2 MaxPool then ConvBlock3D. Halves spatial resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


# ---------------------------------------------------------------------------
# Per-modality stem
# ---------------------------------------------------------------------------

class ModalityStem(nn.Module):
    """
    3D CNN encoder for a SINGLE modality.
    Returns feature maps at 5 spatial scales for skip connections.

    channel_dims[i] gives the number of channels at scale i:
        [C, 2C, 4C, 4C, 4C]  with C = stem_channels
    """

    def __init__(self, stem_channels: int = 24):
        super().__init__()
        c = stem_channels

        self.stage0 = ConvBlock3D(1, c)         # 96³ → 96³  (24 ch)
        self.stage1 = DownBlock3D(c, c * 2)     # 96³ → 48³  (48 ch)
        self.stage2 = DownBlock3D(c * 2, c * 4) # 48³ → 24³  (96 ch)
        self.stage3 = DownBlock3D(c * 4, c * 4) # 24³ → 12³  (96 ch)
        self.bottleneck = DownBlock3D(c * 4, c * 4)  # 12³ → 6³ (96 ch)

        # Channel counts at each scale index (0..4)
        self.channel_dims = [c, c * 2, c * 4, c * 4, c * 4]

    def forward(self, x: torch.Tensor):
        """
        x: (B, 1, P, P, P)
        Returns list of 5 feature maps: [s0, s1, s2, s3, bottleneck]
        """
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        bn = self.bottleneck(s3)
        return [s0, s1, s2, s3, bn]


# ---------------------------------------------------------------------------
# 3D Positional Embedding
# ---------------------------------------------------------------------------

class LearnablePositionalEmbedding3D(nn.Module):
    """
    Factored 3D positional embedding.

    Why factored (x + y + z) rather than a full 3D lookup?
    A full 3D embedding table for a 6×6×6 grid would have 216 entries — that's
    actually fine. But factored embeddings generalise better if patch size changes
    and are standard practice (used in ViT-3D, mmFormer, etc.).

    embed_dim is split equally into three parts (x, y, z), each contributing
    embed_dim//3 dimensions. The three parts are concatenated.
    embed_dim must be divisible by 3 (192 // 3 = 64 ✓).
    """

    def __init__(self, embed_dim: int, max_spatial_size: int = 16):
        super().__init__()
        assert embed_dim % 3 == 0, f"embed_dim ({embed_dim}) must be divisible by 3"
        dim_each = embed_dim // 3
        self.emb_x = nn.Embedding(max_spatial_size, dim_each)
        self.emb_y = nn.Embedding(max_spatial_size, dim_each)
        self.emb_z = nn.Embedding(max_spatial_size, dim_each)

    def forward(self, Hf: int, Wf: int, Df: int, device: torch.device) -> torch.Tensor:
        """
        Returns positional embedding tensor of shape (Hf*Wf*Df, embed_dim).
        """
        ix = torch.arange(Hf, device=device)
        iy = torch.arange(Wf, device=device)
        iz = torch.arange(Df, device=device)

        gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")  # (Hf, Wf, Df) each
        gx, gy, gz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)

        pe = torch.cat([self.emb_x(gx), self.emb_y(gy), self.emb_z(gz)], dim=-1)
        return pe  # (N_tokens, embed_dim)


# ---------------------------------------------------------------------------
# Multi-modality tokenizer
# ---------------------------------------------------------------------------

class MultiModalityTokenizer(nn.Module):
    """
    Runs all 4 modality stems, produces a joint token sequence for the transformer,
    and builds the attention mask that prevents absent modalities from participating.

    Token layout in the sequence:
        [mod0_tok0 ... mod0_tokN | mod1_tok0 ... mod1_tokN | mod2 | mod3]
        Total length: 4 × N_spatial  (N_spatial = 6³ = 216 at patch_size=96)

    The attention mask is (B, 1, 4N, 4N) bool.
    Entry [b, 0, i, j] = True iff token i AND token j are from PRESENT modalities
    in sample b. This prevents:
        - absent tokens from attending to anything (their rows are all False)
        - present tokens from attending to absent tokens (those columns are False)
    """

    def __init__(
        self,
        n_modalities: int = 4,
        stem_channels: int = 24,
        embed_dim: int = 192,
        max_spatial_size: int = 16,
    ):
        super().__init__()
        assert embed_dim % 3 == 0
        self.n_modalities = n_modalities
        self.embed_dim = embed_dim

        # 4 separate CNN stems — one per modality
        self.stems = nn.ModuleList(
            [ModalityStem(stem_channels) for _ in range(n_modalities)]
        )

        # Project bottleneck features to embed_dim
        stem_bn_channels = self.stems[0].channel_dims[-1]  # 96 at stem_channels=24
        self.proj = nn.Linear(stem_bn_channels, embed_dim)

        # 3D positional embeddings — same for every modality (spatial position is shared)
        self.pos_emb = LearnablePositionalEmbedding3D(embed_dim, max_spatial_size)

        # Modality-type embeddings — unique per modality (like BERT token type IDs)
        # This is THE key publishable design choice: the transformer explicitly knows
        # which modality each token came from, unlike a zero-fill baseline where
        # the model must infer missingness from content alone.
        self.modality_type_emb = nn.Embedding(n_modalities, embed_dim)

    def forward(self, image: torch.Tensor, presence_mask: torch.Tensor):
        """
        Args:
            image        : (B, 4, H, W, D) — all 4 channels; absent ones are zero-filled
            presence_mask: (B, 4) float — 1.0 = present, 0.0 = absent

        Returns:
            tokens      : (B, 4*N, embed_dim) — full token sequence (absent tokens zeroed)
            attn_mask   : (B, 1, 4*N, 4*N) bool — True where attention is allowed
            skip_feats  : list of 4 lists; skip_feats[m][s] = (B, C_s, H_s, W_s, D_s)
            spatial_shape: (Hf, Wf, Df) — bottleneck spatial dims
            n_per_mod   : int — N_spatial tokens per modality
        """
        B = image.shape[0]
        device = image.device

        # --- Run each modality through its stem ---
        all_stem_feats = []
        for m in range(self.n_modalities):
            x_m = image[:, m : m + 1]           # (B, 1, H, W, D)
            feats_m = self.stems[m](x_m)         # list of 5 feature maps
            all_stem_feats.append(feats_m)

        # Spatial shape at bottleneck
        bn_feat = all_stem_feats[0][-1]          # (B, C, Hf, Wf, Df)
        B, C, Hf, Wf, Df = bn_feat.shape
        N = Hf * Wf * Df                          # tokens per modality (e.g. 216)

        # --- Shared 3D positional embedding ---
        pos = self.pos_emb(Hf, Wf, Df, device)  # (N, embed_dim)

        # --- Build per-modality tokens ---
        mod_tokens = []  # each (B, N, embed_dim)
        for m in range(self.n_modalities):
            feat = all_stem_feats[m][-1]          # (B, C, Hf, Wf, Df)
            flat = feat.flatten(2).transpose(1, 2)  # (B, N, C)
            toks = self.proj(flat)                  # (B, N, embed_dim)
            toks = toks + pos.unsqueeze(0)          # + 3D position
            toks = toks + self.modality_type_emb(
                torch.tensor(m, device=device)
            )                                       # + modality type
            mod_tokens.append(toks)

        # Concatenate: (B, 4*N, embed_dim)
        tokens = torch.cat(mod_tokens, dim=1)

        # --- Zero out tokens from absent modalities ---
        # presence_mask: (B, 4) → expand to (B, 4*N, 1)
        pres_per_tok = (
            presence_mask                        # (B, 4)
            .unsqueeze(-1)                       # (B, 4, 1)
            .expand(B, self.n_modalities, N)     # (B, 4, N)
            .reshape(B, self.n_modalities * N)   # (B, 4*N)
        )
        tokens = tokens * pres_per_tok.unsqueeze(-1)   # (B, 4*N, D)

        # --- Build attention mask ---
        # Token i can attend to token j iff BOTH i and j are from present modalities
        # pres_per_tok: (B, 4*N)  with values 0 or 1
        tok_pres_bool = pres_per_tok.bool()                       # (B, 4*N)
        attn_mask = (
            tok_pres_bool.unsqueeze(2) & tok_pres_bool.unsqueeze(1)
        )                                                          # (B, 4*N, 4*N)
        attn_mask = attn_mask.unsqueeze(1)                        # (B, 1, 4*N, 4*N)

        return tokens, attn_mask, all_stem_feats, (Hf, Wf, Df), N
