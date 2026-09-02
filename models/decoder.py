"""
U-Net style 3D decoder for the masked-fusion transformer.

Takes the transformer output (reshaped back to a 3D feature map) and progressively
upsamples it back to the input patch resolution using skip connections from the
per-modality CNN stems.

Key challenge: the skip connections come from 4 SEPARATE modality stems, but some
modalities may be absent. We solve this with presence-gated averaging:
  fused_skip = sum(present_modality_skips) / n_present_modalities

Why average and not concatenate all 4?
  Concatenating all 4 would double the channel count per scale (4 × C instead of C),
  roughly 4× the decoder parameters and memory. Averaging preserves channel count,
  keeps the decoder lightweight, and still lets the model use information from all
  present modalities. More sophisticated approaches (e.g., cross-attention between
  skip features) exist but add complexity without reliably improving Dice at our scale.

Why skip connections at all in the transformer model?
  The transformer bottleneck is at 6³ resolution — far too coarse to reconstruct
  fine tumour boundaries (especially ET, which can be small). Skip connections from
  the stem at 12³, 24³, 48³, 96³ inject the high-frequency spatial detail needed for
  precise boundary prediction.
"""

import torch
import torch.nn as nn
from models.stem import ConvBlock3D


# ---------------------------------------------------------------------------
# Skip connection fusion across modalities
# ---------------------------------------------------------------------------

def fuse_skip_features(
    feat_list: list,
    presence_mask: torch.Tensor,
    scale_idx: int,
) -> torch.Tensor:
    """
    Presence-gated average of per-modality skip features at a given scale.

    Args:
        feat_list    : list of 4 lists; feat_list[m][s] = (B, C, H, W, D)
        presence_mask: (B, 4) float — 1=present, 0=absent
        scale_idx    : which scale index (0..3) to fuse

    Returns:
        (B, C, H, W, D) — mean of present-modality features
    """
    n_modalities = len(feat_list)
    feats = [feat_list[m][scale_idx] for m in range(n_modalities)]  # list of (B,C,H,W,D)

    stacked = torch.stack(feats, dim=1)            # (B, 4, C, H, W, D)
    pres = presence_mask[:, :, None, None, None, None].float()  # (B, 4, 1, 1, 1, 1)

    masked = stacked * pres                         # zero out absent modalities
    n_present = pres.sum(dim=1).clamp(min=1.0)     # (B, 1, 1, 1, 1, 1) — avoid div/0
    fused = masked.sum(dim=1) / n_present           # (B, C, H, W, D)
    return fused


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------

class UpBlock3D(nn.Module):
    """
    Upsample (trilinear) → concat skip connection → ConvBlock3D.

    Using trilinear interpolation instead of ConvTranspose3d avoids the
    checkerboard artefacts that can degrade boundary sharpness in segmentation.
    The following ConvBlock3D learns to refine the upsampled + skip features.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, in_ch, H, W, D)
        skip: (B, skip_ch, 2H, 2W, 2D)
        """
        x = self.up(x)                  # (B, in_ch, 2H, 2W, 2D)
        x = torch.cat([x, skip], dim=1) # (B, in_ch+skip_ch, 2H, 2W, 2D)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class UNet3DDecoder(nn.Module):
    """
    4-stage U-Net decoder.

    Input  : (B, embed_dim, 6, 6, 6)   — reshaped transformer output
    Output : (B, out_channels, 96, 96, 96) — raw logits (no sigmoid)

    Skip connection channel sizes come from ModalityStem.channel_dims:
        scale 3: 96 ch  (12³)
        scale 2: 96 ch  (24³)
        scale 1: 48 ch  (48³)
        scale 0: 24 ch  (96³)
    These are fused across present modalities before being passed here.

    Decoder channel progression (embed_dim=192, stem_channels=24):
        6³  (192 ch)  → Up1 + skip[3] (96 ch)  → 96 ch (12³)
        12³  (96 ch)  → Up2 + skip[2] (96 ch)  → 48 ch (24³)
        24³  (48 ch)  → Up3 + skip[1] (48 ch)  → 24 ch (48³)
        48³  (24 ch)  → Up4 + skip[0] (24 ch)  → 16 ch (96³)
        96³  (16 ch)  → 1×1×1 conv              → 3 ch  (logits: WT, TC, ET)
    """

    def __init__(
        self,
        embed_dim: int = 192,
        stem_channel_dims: list = None,  # [24, 48, 96, 96, 96]
        out_channels: int = 3,
    ):
        super().__init__()
        if stem_channel_dims is None:
            stem_channel_dims = [24, 48, 96, 96, 96]

        s0, s1, s2, s3, _ = stem_channel_dims  # 24, 48, 96, 96, (96 = bottleneck)

        # Up1: 6³ → 12³
        self.up1 = UpBlock3D(embed_dim, s3, 96)
        # Up2: 12³ → 24³
        self.up2 = UpBlock3D(96, s2, 48)
        # Up3: 24³ → 48³
        self.up3 = UpBlock3D(48, s1, 24)
        # Up4: 48³ → 96³
        self.up4 = UpBlock3D(24, s0, 16)
        # Final segmentation head
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,          # (B, embed_dim, 6, 6, 6) — transformer bottleneck
        fused_skips: list,        # [fused_s0, fused_s1, fused_s2, fused_s3]
    ) -> torch.Tensor:
        """
        Returns raw logits (B, out_channels, P, P, P). Apply sigmoid in loss/eval.
        """
        x = self.up1(x, fused_skips[3])   # 6³  → 12³
        x = self.up2(x, fused_skips[2])   # 12³ → 24³
        x = self.up3(x, fused_skips[1])   # 24³ → 48³
        x = self.up4(x, fused_skips[0])   # 48³ → 96³
        return self.out_conv(x)
