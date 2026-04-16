# ConvSeg-Net: A Hybrid CNN-Transformer Architecture for Semantic Segmentation

## 1. Motivation & Design Philosophy

Modern semantic segmentation demands two complementary capabilities:

| Capability | Best Modeled By | Why |
|---|---|---|
| **Local boundary & texture extraction** | CNNs (ConvNeXt blocks) | Translation equivariance, strong inductive bias, $O(N)$ complexity |
| **Global semantic reasoning** | Transformers (Self-Attention) | Unbounded receptive field, dynamic content-based routing |

> [!IMPORTANT]
> **Core Insight**: CNNs excel at high-resolution early stages where spatial detail matters, while Transformers excel at low-resolution deep stages where global context matters. ConvSeg-Net exploits this by placing each where it is most effective *and* most computationally efficient.

---

## 2. High-Level Architecture

```mermaid
flowchart TB
    subgraph Input
        IMG["Input Image<br/>H x W x 3"]
    end

    subgraph ENCODER["ENCODER - Hierarchical Feature Pyramid"]
        direction TB
        S1["Stage 1 - CNN<br/>H/4 x W/4 x C1<br/>ConvNeXt Blocks"]
        S2["Stage 2 - CNN<br/>H/8 x W/8 x C2<br/>ConvNeXt Blocks"]
        S3["Stage 3 - Transformer<br/>H/16 x W/16 x C3<br/>Efficient Self-Attention"]
        S4["Stage 4 - Transformer<br/>H/32 x W/32 x C4<br/>Efficient Self-Attention"]
    end

    subgraph DECODER["DECODER - Boundary-Aware MLP Fusion"]
        direction TB
        MLP_PROJ["MLP Projection<br/>All stages -> C_embed"]
        CAG["Cross-Attention<br/>Boundary Gate"]
        FUSE["Concatenation + MLP Fusion"]
        HEAD["Segmentation Head<br/>H/4 x W/4 x N_cls"]
    end

    subgraph Output
        MASK["Predicted Mask<br/>H x W x N_cls"]
    end

    IMG --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4

    S1 -- "F1" --> MLP_PROJ
    S2 -- "F2" --> MLP_PROJ
    S3 -- "F3" --> MLP_PROJ
    S4 -- "F4" --> MLP_PROJ

    MLP_PROJ --> CAG
    CAG --> FUSE
    FUSE --> HEAD
    HEAD -->|"Bilinear Upsample x4"| MASK

    style S1 fill:#2d6a4f,color:#fff,stroke:#1b4332
    style S2 fill:#40916c,color:#fff,stroke:#2d6a4f
    style S3 fill:#7b2cbf,color:#fff,stroke:#5a189a
    style S4 fill:#9d4edd,color:#fff,stroke:#7b2cbf
    style CAG fill:#e85d04,color:#fff,stroke:#dc2f02
    style MASK fill:#0077b6,color:#fff,stroke:#023e8a
```

---

## 3. Encoder — Detailed Specification

### 3.1 Stage Configuration

| Stage | Resolution | Channels ($C_i$) | Block Type | Num Blocks | Downsampling |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | $\frac{H}{4} \times \frac{W}{4}$ | 64 | ConvNeXt | 3 | Patch Embed (4×4 conv, stride 4) |
| 2 | $\frac{H}{8} \times \frac{W}{8}$ | 128 | ConvNeXt | 3 | 2×2 conv, stride 2 |
| 3 | $\frac{H}{16} \times \frac{W}{16}$ | 320 | Transformer | 6 | 2×2 conv, stride 2 |
| 4 | $\frac{H}{32} \times \frac{W}{32}$ | 512 | Transformer | 3 | 2×2 conv, stride 2 |

### 3.2 ConvNeXt Block (Stages 1 & 2)

Each block follows the modernized ConvNeXt design:

```mermaid
flowchart LR
    subgraph ConvNeXt_Block["ConvNeXt Block"]
        direction TB
        A["Input: X"] --> B["7x7 Depthwise Conv<br/>(groups = C)"]
        B --> C["LayerNorm"]
        C --> D["1x1 Conv (C -> 4C)<br/>+ GELU"]
        D --> E["1x1 Conv (4C -> C)"]
        E --> F["Scale: gamma * output"]
    end
    A -- "Residual" --> G["+ Add"]
    F --> G
    G --> H["Output: Y"]

    style B fill:#2d6a4f,color:#fff
    style D fill:#40916c,color:#fff
    style E fill:#40916c,color:#fff
```

**Mathematically:**

$$Y = X + \gamma \cdot \text{Conv}_{1\times1}\Big(\text{GELU}\big(\text{Conv}_{1\times1}(\text{LN}(\text{DWConv}_{7\times7}(X)))\big)\Big)$$

where $\gamma$ is a learnable per-channel scale initialized to $10^{-6}$.

> [!TIP]
> The 7×7 depthwise convolution provides a large receptive field at minimal parameter cost (only $49 \times C$ parameters vs. $49 \times C^2$ for standard convolution). This is critical for capturing object boundaries at high resolution.

---

### 3.3 Efficient Self-Attention Block (Stages 3 & 4)

We use **Spatial-Reduction Attention (SRA)** from SegFormer to keep attention tractable:

```mermaid
flowchart TB
    subgraph Transformer_Block["Transformer Block"]
        direction TB
        X["Input: X<br/>(N tokens x C)"]
        
        subgraph ESA["Efficient Self-Attention"]
            Q_proj["Linear -> Q<br/>(N x C)"]
            KV_reduce["Spatial Reduction<br/>Conv RxR, stride R<br/>(N -> N/R^2)"]
            KV_proj["Linear -> K, V<br/>(N/R^2 x C)"]
            ATT["Attention:<br/>softmax(QK^T / sqrt(d)) * V"]
        end
        
        subgraph FFN["Mix-FFN"]
            FF1["Linear (C -> 4C)"]
            DW["3x3 Depthwise Conv"]
            GELU["GELU"]
            FF2["Linear (4C -> C)"]
        end
    end

    X --> Q_proj
    X --> KV_reduce
    KV_reduce --> KV_proj
    Q_proj --> ATT
    KV_proj --> ATT
    ATT -- "+X (residual)" --> FF1
    FF1 --> DW
    DW --> GELU
    GELU --> FF2
    FF2 -- "+residual" --> OUT["Output: Y"]

    style KV_reduce fill:#7b2cbf,color:#fff
    style ATT fill:#9d4edd,color:#fff
    style DW fill:#c77dff,color:#fff
```

**Spatial Reduction Attention (SRA):**

$$Q = XW_Q, \quad K = \text{Reshape}(X, \frac{N}{R^2}) \cdot W_K, \quad V = \text{Reshape}(X, \frac{N}{R^2}) \cdot W_V$$

$$\text{SRA}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{head}}}\right) V$$

| Stage | Reduction Ratio $R$ | Effective Attention Tokens | Complexity |
|:---:|:---:|:---:|:---:|
| 3 | 2 | $N/4$ | $O(N \cdot N/4) = O(N^2/4)$ |
| 4 | 1 | $N$ (full) | $O(N^2)$, but $N$ is only $\frac{HW}{1024}$ |

**Mix-FFN** (replaces positional encoding):

$$Y = \text{MLP}\big(\text{GELU}(\text{DWConv}_{3\times3}(\text{MLP}(X)))\big) + X$$

> [!NOTE]
> The 3×3 depthwise convolution inside the FFN implicitly encodes positional information (as shown in SegFormer), eliminating the need for explicit positional embeddings and enabling arbitrary input resolutions at inference time.

---

## 4. Decoder — Boundary-Aware MLP Fusion

This is the **novel contribution** over both FCN and SegFormer.

### 4.1 Full Decoder Pipeline

```mermaid
flowchart TB
    subgraph Stage_Outputs["Encoder Feature Maps"]
        F1["F1<br/>H/4 x W/4 x 64"]
        F2["F2<br/>H/8 x W/8 x 128"]
        F3["F3<br/>H/16 x W/16 x 320"]
        F4["F4<br/>H/32 x W/32 x 512"]
    end

    subgraph Step1["Step 1: Unify Channel Dimension"]
        P1["MLP: 64 -> Ce"]
        P2["MLP: 128 -> Ce"]
        P3["MLP: 320 -> Ce"]
        P4["MLP: 512 -> Ce"]
    end

    subgraph Step2["Step 2: Unify Spatial Resolution"]
        U1["Identity<br/>(already H/4)"]
        U2["Upsample x2<br/> -> H/4"]
        U3["Upsample x4<br/> -> H/4"]
        U4["Upsample x8<br/> -> H/4"]
    end

    subgraph Step3["Step 3: Cross-Attention Boundary Gate"]
        direction TB
        GATE["Cross-Attention Gate<br/>Query: F4_hat (global semantic)<br/>Key/Value: [F1_hat ; F2_hat] (local boundary)<br/><br/>Output: Boundary-refined<br/>global features"]
    end

    subgraph Step4["Step 4: Fuse & Predict"]
        CONCAT["Concat: [F1_hat ; F2_hat ; F3_hat ; G]<br/>along channel dim<br/> -> H/4 x W/4 x 4Ce"]
        FINAL_MLP["MLP: 4Ce -> N_cls"]
    end

    F1 --> P1 --> U1
    F2 --> P2 --> U2
    F3 --> P3 --> U3
    F4 --> P4 --> U4

    U1 -- "F1_hat" --> GATE
    U2 -- "F2_hat" --> GATE
    U4 -- "F4_hat" --> GATE
    U3 -- "F3_hat" --> CONCAT

    GATE -- "G (gated)" --> CONCAT
    U1 -- "F1_hat" --> CONCAT
    U2 -- "F2_hat" --> CONCAT

    CONCAT --> FINAL_MLP
    FINAL_MLP --> SEG["Segmentation Map<br/>H/4 x W/4 x N_cls"]

    style GATE fill:#e85d04,color:#fff,stroke:#dc2f02
    style P1 fill:#2d6a4f,color:#fff
    style P2 fill:#40916c,color:#fff
    style P3 fill:#7b2cbf,color:#fff
    style P4 fill:#9d4edd,color:#fff
    style FINAL_MLP fill:#0077b6,color:#fff
```

> [!NOTE]
> **Why the decoder is intentionally asymmetric:**
> - $\hat{F}_1$ and $\hat{F}_2$ are shallow CNN features, so they are used as local boundary cues for $K,V$ in the gate.
> - $\hat{F}_4$ is the deep semantic query and is already mixed into $G$ via residual gating, so concatenating raw $\hat{F}_4$ again would be redundant.
> - $\hat{F}_3$ is fused directly as a mid-level semantic feature; excluding it from the gate keeps gate compute focused on boundary refinement.
> - This design matches Section 4.2 equations and preserves the $4C_e$ fusion width in Step 4.

### 4.2 Cross-Attention Boundary Gate (Novel Component)

This is the key innovation: **global semantics selectively attend to local boundaries**.

```mermaid
flowchart TB
    subgraph Inputs
        F4_hat["F4_hat - Global Semantic Features<br/>(deep Transformer output)"]
        F12["[F1_hat ; F2_hat] - Local Boundary Features<br/>(shallow CNN output, concatenated)"]
    end

    subgraph CrossAttGate["Cross-Attention Boundary Gate"]
        direction TB
        Q_gate["W_Q * F4_hat -> Q<br/>(H/4*W/4 x d)"]
        K_gate["W_K * [F1_hat ; F2_hat] -> K<br/>(H/4*W/4 x d)"]
        V_gate["W_V * [F1_hat ; F2_hat] -> V<br/>(H/4*W/4 x d)"]
        
        ATTN["A = softmax(QK^T / sqrt(d))"]
        BLEND["G_raw = A * V"]
        SIGMOID["sigma = sigmoid(W_g * [F4_hat ; G_raw])"]
        GATED["G = sigma * G_raw + (1 - sigma) * F4_hat"]
    end

    F4_hat --> Q_gate
    F12 --> K_gate
    F12 --> V_gate
    Q_gate --> ATTN
    K_gate --> ATTN
    ATTN --> BLEND
    V_gate --> BLEND
    BLEND --> SIGMOID
    F4_hat --> SIGMOID
    SIGMOID --> GATED
    F4_hat --> GATED
    GATED --> OUT_G["G - Boundary-Refined<br/>Semantic Features"]

    style ATTN fill:#e85d04,color:#fff,stroke:#dc2f02
    style SIGMOID fill:#f48c06,color:#fff,stroke:#e85d04
    style GATED fill:#ffba08,color:#000,stroke:#f48c06
```

**Mathematical Formulation:**

$$Q = \hat{F}_4 W_Q, \quad K = [\hat{F}_1 \| \hat{F}_2] W_K, \quad V = [\hat{F}_1 \| \hat{F}_2] W_V$$

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right), \quad G_{\text{raw}} = A \cdot V$$

$$\sigma = \text{sigmoid}\big(W_g \cdot [\hat{F}_4 \| G_{\text{raw}}]\big)$$

$$G = \sigma \odot G_{\text{raw}} + (1 - \sigma) \odot \hat{F}_4$$

> [!IMPORTANT]
> **Why Cross-Attention instead of simple concatenation?**
> - **FCN** concatenates or adds skip connections blindly — all local features are equally weighted.
> - **SegFormer** discards explicit skip logic — the MLP decoder has no mechanism to selectively refine boundaries.
> - **ConvSeg-Net's Gate** lets the Transformer's global understanding *query* the CNN's boundary maps, selectively amplifying edges that correspond to real semantic boundaries (e.g., object contours) while suppressing irrelevant texture edges (e.g., grass texture).

---

## 5. Complete Forward Pass — Summary

```mermaid
flowchart LR
    subgraph E["Encoder"]
        direction TB
        I["Image<br/>H x W x 3"] --> S1_b["Conv Stem<br/>4x4, s4"]
        S1_b --> CNX1["ConvNeXt x3<br/>H/4, C1=64"]
        CNX1 --> DS1["downsample 2"]
        DS1 --> CNX2["ConvNeXt x3<br/>H/8, C2=128"]
        CNX2 --> DS2["downsample 2"]
        DS2 --> TR1["Transformer x6<br/>H/16, C3=320"]
        TR1 --> DS3["downsample 2"]
        DS3 --> TR2["Transformer x3<br/>H/32, C4=512"]
    end

    subgraph D["Decoder"]
        direction TB
        PROJ["MLP Project<br/>all -> Ce=256"]
        UP["Upsample all<br/>-> H/4"]
        XATTN["Cross-Attn<br/>Boundary Gate"]
        CAT["Concat + MLP<br/>-> N_cls"]
    end

    CNX1 -- "F1" --> PROJ
    CNX2 -- "F2" --> PROJ
    TR1 -- "F3" --> PROJ
    TR2 -- "F4" --> PROJ
    PROJ --> UP --> XATTN --> CAT --> M["Mask<br/>H x W x N_cls"]

    style CNX1 fill:#2d6a4f,color:#fff
    style CNX2 fill:#40916c,color:#fff
    style TR1 fill:#7b2cbf,color:#fff
    style TR2 fill:#9d4edd,color:#fff
    style XATTN fill:#e85d04,color:#fff
```

---

## 6. Computational Complexity Analysis

| Component | Complexity | Justification |
|---|---|---|
| Stage 1 (CNN, H/4) | $O(HW \cdot C_1 \cdot k^2)$ | Depthwise conv is $O(N)$ in spatial dims |
| Stage 2 (CNN, H/8) | $O(\frac{HW}{4} \cdot C_2 \cdot k^2)$ | 4× fewer spatial tokens |
| Stage 3 (Transformer, H/16) | $O\left(\frac{H^2W^2}{256} \cdot \frac{C_3}{R^2}\right)$ | SRA reduces K,V by $R^2=4$ |
| Stage 4 (Transformer, H/32) | $O\left(\frac{H^2W^2}{1024} \cdot C_4\right)$ | Full attention, but only $\frac{HW}{1024}$ tokens |
| Cross-Attention Gate | $O\left(\frac{HW}{16} \cdot C_e \cdot 2\right)$ | Q from F₄ (small), K/V from F₁∪F₂ (large but linear projection) |
| MLP Decoder | $O\left(\frac{HW}{16} \cdot 4C_e \cdot N_{cls}\right)$ | Single linear projection |

**Total**: Asymptotically comparable to SegFormer-B2 but with stronger local feature extraction from the CNN stem.

---

## 7. Model Variants

| Variant | $C_1$–$C_4$ | Blocks per Stage | Params (est.) | Target |
|:---:|:---:|:---:|:---:|:---:|
| **ConvSeg-T** (Tiny) | 32-64-160-256 | 2-2-4-2 | ~6M | Mobile / Edge |
| **ConvSeg-S** (Small) | 64-128-320-512 | 3-3-6-3 | ~25M | Standard benchmark |
| **ConvSeg-B** (Base) | 64-128-320-512 | 3-3-18-3 | ~45M | High accuracy |
| **ConvSeg-L** (Large) | 96-192-384-768 | 3-3-27-3 | ~85M | SOTA push |

---

## 8. Training Recipe (Recommended)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | $6 \times 10^{-5}$ (encoder), $6 \times 10^{-4}$ (decoder) |
| LR Schedule | Polynomial decay, power 1.0 |
| Weight Decay | 0.01 |
| Batch Size | 16 (8 GPUs × 2) |
| Augmentation | Random resize (0.5–2.0), crop 512×512, flip, PhotoDistort |
| Loss | Cross-Entropy + Lovász-Softmax (boundary-aware) |
| Pretraining | ImageNet-1K supervised (encoder only) |
| Fine-tune Epochs | 160K iterations |

---

## 9. Design Rationale vs. Prior Work

```mermaid
flowchart LR
    subgraph FCN_lim["FCN Limitations"]
        A1["(x) No global context"]
        A2["(v) Strong boundaries"]
        A3["(v) O(N) complexity"]
    end

    subgraph SegF_lim["SegFormer Limitations"]
        B1["(v) Global context"]
        B2["(x) Weak boundaries<br/>(patch artifacts)"]
        B3["(!) O(N^2/R^2) attention"]
    end

    subgraph ConvSeg["ConvSeg-Net"]
        C1["(v) Global context<br/>(Transformer stages 3-4)"]
        C2["(v) Strong boundaries<br/>(CNN stages 1-2<br/>+ Cross-Attn Gate)"]
        C3["(v) Efficient<br/>(CNN where expensive,<br/>Transformer where cheap)"]
    end

    FCN_lim --> ConvSeg
    SegF_lim --> ConvSeg

    style ConvSeg fill:#0077b6,color:#fff,stroke:#023e8a
    style FCN_lim fill:#6c757d,color:#fff
    style SegF_lim fill:#6c757d,color:#fff
```

---

## 10. Key Novelty Claims

1. **Compute-Optimal Placement**: CNNs at high-resolution stages (where attention is $O(N^2)$ expensive) and Transformers at low-resolution stages (where convolutions have limited receptive fields).

2. **Cross-Attention Boundary Gate**: A learnable, content-dependent fusion mechanism that replaces both FCN's blind skip connections and SegFormer's flat concatenation. Deep semantic features *query* shallow boundary features, producing boundary-refined semantic maps.

3. **Positional-Encoding-Free Design**: The Mix-FFN's depthwise convolution provides implicit positional information, enabling variable input resolution at test time without interpolating positional embeddings.

---

## Open Questions for Review

> [!WARNING]
> **Attention Cost in Decoder**: The Cross-Attention Gate operates at H/4 × W/4 resolution. For a 512×512 input, that's 16,384 query tokens attending to 16,384 key tokens. Should we apply spatial reduction here too (e.g., reduce K/V by 4×)?

> [!IMPORTANT]
> **Pretraining Strategy**: Should stages 1-2 be initialized from a pretrained ConvNeXt, and stages 3-4 from a pretrained SegFormer encoder? Or should the whole encoder be trained jointly from scratch on ImageNet?

> [!NOTE]
> **Loss Function**: The Lovász-Softmax loss is boundary-aware and directly optimizes IoU. Is this redundant with the Cross-Attention Boundary Gate, or complementary?
