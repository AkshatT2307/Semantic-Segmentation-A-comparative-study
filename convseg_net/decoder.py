"""
ConvSeg-Net Decoder — Boundary-Aware MLP Fusion.

Pipeline:
    1. MLP-project each encoder stage to a uniform channel dim C_e.
    2. Upsample all to H/4 resolution.
    3. Cross-Attention Boundary Gate: F4_hat queries [F1_hat ; F2_hat] → G.
    4. Concatenate [F1_hat, F2_hat, F3_hat, G] → 4·C_e.
    5. Fusion MLP → num_classes.

The Cross-Attention Boundary Gate (Step 3) is the key novel component:
    - Global semantic features (F4) selectively attend to local boundary
      features (F1, F2), producing boundary-refined semantic maps.
    - Sigmoid gating blends the cross-attention output with the original
      global features to prevent information loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjection(nn.Module):
    """Linear projection: C_in → C_e via 1×1 conv + LayerNorm."""

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) → (B, C_e, H, W)"""
        x = self.proj(x)
        B, C, H, W = x.shape
        # Apply LN on channel dim: (B, C, H, W) → (B, H*W, C) → LN → back
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class CrossAttentionBoundaryGate(nn.Module):
    """Cross-Attention Boundary Gate — novel decoder component.

    Global semantic features (F4_hat) query shallow boundary features
    ([F1_hat ; F2_hat]) via multi-head cross-attention, then a sigmoid
    gate blends the result:

        Q = F4_hat · W_Q
        K = [F1_hat ; F2_hat] · W_K
        V = [F1_hat ; F2_hat] · W_V
        G_raw = softmax(Q K^T / √d) · V
        σ = sigmoid(W_g · [F4_hat ‖ G_raw])
        G = σ ⊙ G_raw + (1 - σ) ⊙ F4_hat

    Args:
        embed_dim:      Channel dimension (C_e).
        num_heads:      Number of attention heads.
        attn_drop:      Dropout on attention weights.
        sr_ratio:       Optional spatial reduction on K/V (default 1 = none).
                        Set >1 to reduce memory for large resolution inputs.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q projection from global semantic features
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # K, V projections from local boundary features
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

        # Sigmoid gate: projects [F4_hat ‖ G_raw] → C_e → sigmoid
        self.gate_proj = nn.Linear(2 * embed_dim, embed_dim)

        # Optional spatial reduction on K/V
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_conv = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio,
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        f4_hat: torch.Tensor,
        f1_hat: torch.Tensor,
        f2_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        All inputs at same spatial resolution (H/4 × W/4).

        Args:
            f4_hat: (B, C_e, H, W) — global semantic features.
            f1_hat: (B, C_e, H, W) — shallow boundary features (stage 1).
            f2_hat: (B, C_e, H, W) — shallow boundary features (stage 2).
        Returns:
            G: (B, C_e, H, W) — boundary-refined semantic features.
        """
        B, C, H, W = f4_hat.shape
        N = H * W

        # Flatten to sequence: (B, N, C)
        q_in = f4_hat.flatten(2).permute(0, 2, 1)

        # Concatenate F1, F2 along spatial dim for K/V source
        # Both are (B, C, H, W), concat along spatial → (B, C, H, W) * 2
        boundary = torch.cat([f1_hat, f2_hat], dim=2)  # (B, C, 2H, W)

        # Optional spatial reduction
        if self.sr_ratio > 1:
            boundary = self.sr_conv(boundary)
            kv_in = boundary.flatten(2).permute(0, 2, 1)
            kv_in = self.sr_norm(kv_in)
        else:
            kv_in = boundary.flatten(2).permute(0, 2, 1)  # (B, 2*N, C)

        N_kv = kv_in.shape[1]

        # Q, K, V projections → multi-head reshape
        q = self.q_proj(q_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_in).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        g_raw = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        g_raw = self.out_proj(g_raw)

        # Sigmoid gating
        sigma = torch.sigmoid(self.gate_proj(torch.cat([q_in, g_raw], dim=-1)))  # (B, N, C)
        g = sigma * g_raw + (1.0 - sigma) * q_in  # (B, N, C)

        # Reshape back to spatial
        g = g.permute(0, 2, 1).reshape(B, C, H, W)
        return g


class ConvSegDecoder(nn.Module):
    """Boundary-Aware MLP Fusion decoder.

    Takes 4 encoder feature maps at different scales and produces segmentation
    logits at H/4 × W/4 resolution.

    The decoder is intentionally asymmetric:
        - F1_hat, F2_hat → used as boundary cues (K/V in the gate) AND
          concatenated for fusion.
        - F3_hat → concatenated directly (mid-level semantic).
        - F4_hat → enters the gate as Q, and the gated output G replaces
          raw F4_hat in the fusion concat (F4 is already mixed into G via the
          residual sigmoid gate, so concatenating raw F4 again is redundant).

    Final concat: [F1_hat, F2_hat, F3_hat, G] → 4·C_e → MLP → num_classes.

    Args:
        in_channels_list:   List of 4 encoder channel dims [C1, C2, C3, C4].
        embed_dim:          Unified channel dimension C_e for all stages.
        num_classes:        Number of segmentation classes.
        dropout:            Spatial dropout before final classification.
        gate_num_heads:     Number of attention heads in the boundary gate.
        gate_sr_ratio:      Spatial reduction ratio in boundary gate K/V.
    """

    def __init__(
        self,
        in_channels_list: list[int] = (64, 128, 320, 512),
        embed_dim: int = 256,
        num_classes: int = 19,
        dropout: float = 0.1,
        gate_num_heads: int = 8,
        gate_sr_ratio: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Step 1: MLP projections to unify channel dimension
        self.mlp_projs = nn.ModuleList([
            MLPProjection(c, embed_dim) for c in in_channels_list
        ])

        # Step 3: Cross-Attention Boundary Gate
        self.boundary_gate = CrossAttentionBoundaryGate(
            embed_dim=embed_dim,
            num_heads=gate_num_heads,
            sr_ratio=gate_sr_ratio,
        )

        # Step 4: Fusion MLP — 4·C_e → C_e → num_classes
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [F1, F2, F3, F4] from encoder, each (B, C_i, H_i, W_i).
        Returns:
            logits: (B, num_classes, H/4, W/4).
        """
        assert len(features) == 4

        # Step 1: Unify channel dimension
        projected = [proj(feat) for proj, feat in zip(self.mlp_projs, features)]

        # Step 2: Upsample all to H/4 × W/4 (= resolution of projected[0])
        target_size = projected[0].shape[2:]  # (H/4, W/4)
        upsampled = []
        for i, p in enumerate(projected):
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(p)

        f1_hat, f2_hat, f3_hat, f4_hat = upsampled

        # Step 3: Cross-Attention Boundary Gate
        g = self.boundary_gate(f4_hat, f1_hat, f2_hat)

        # Step 4: Concatenate [F1_hat, F2_hat, F3_hat, G] → 4·C_e
        fused = torch.cat([f1_hat, f2_hat, f3_hat, g], dim=1)  # (B, 4*C_e, H/4, W/4)

        # Step 5: Fusion MLP → classification
        out = self.fusion(fused)
        out = self.dropout(out)
        out = self.cls_head(out)
        return out  # (B, num_classes, H/4, W/4)
