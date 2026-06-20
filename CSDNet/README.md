# CSDNet: Content-Structure Dual-aware Network for Image Super-Resolution

PyTorch implementation of CSDNet.

## Overview

CSDNet introduces **dual-aware routing** that jointly considers content similarity and
structural compatibility when grouping pixels into attention clusters. A lightweight
**Frequency Structure Encoder (FSE)** extracts local structure descriptors via DCT
decomposition, which guide both the routing process and a structure-aware fusion module.

Key idea: pixels that belong to the same object should be grouped together not just
because they look similar (content), but because they share the same local structure.

## Architecture

```
LR → Head Conv → [DAB × 8] → PixelShuffle Upsampler → Last Conv → SR
                   │
                   ├─ IGPA: prototype refinement → global K/V
                   ├─ IGA:  grouped local + global cross-attention  
                   ├─ FSE:  DCT → low/high branches → Φ_cross fusion
                   ├─ Structure Fusion: content + α · structure
                   └─ LWI:  patch-based local window interaction
```

### Module Summary

| Module | Description |
|--------|-------------|
| **FSE** | 3×3 DCT → low/high-frequency MLP branches → cross-frequency interaction (Φ_cross) → structure descriptor |
| **IGPA** | Prototype refinement via iterative clustering → global K/V projection |
| **IGA** | Local-window self-attention + cross-attention to global prototypes |
| **LWI** | Overlapping patch-based local self-attention |
| **DAB** | One full transformer block: IGPA → IGA → Structure Fusion → LWI → ConvFFN |
| **Structure Fusion** | Learnable α mixing: content + sigmoid(α) · structure |

## Files

```
csdnet/
├── __init__.py
└── archs/
    ├── __init__.py
    ├── csdnet.py       # CSDNet, DAB, StructureFusionModule
    ├── fse.py          # FrequencyStructureEncoder
    ├── attention.py    # IGPA, IGA, LWI, Attention, ConvFFN, PreNorm
    └── utils.py        # center_iter, ema_inplace, etc.
```

## Quick Start

```python
import torch
from csdnet import CSDNet

model = CSDNet(upscale=4)          # ×4 super-resolution
model.eval()

lr = torch.randn(1, 3, 256, 256)   # (B, C, H, W), range [0, 1] or [0, 255]
with torch.no_grad():
    sr = model(lr)                  # (1, 3, 1024, 1024)
```

## Installation

```bash
pip install torch einops
```

## Efficiency

| Scale | Params | Multi-Adds |
|:---:|:---:|:---:|
| ×2 | ~477.5K | ~37.3G |
| ×3 | ~549.7K | ~42.4G |
| ×4 | ~534.5K | ~53.3G |

Multi-Adds measured with fvcore on 3×256×256 LR input.

## Results (×4)

| Dataset | PSNR | SSIM |
|---------|------|------|
| Set5 | 32.51 | 0.9000 |
| Set14 | 28.86 | 0.7879 |
| BSD100 | 27.77 | 0.7447 |
| Urban100 | 26.87 | 0.8078 |
| Manga109 | 31.26 | 0.9181 |

## Citation

```bibtex
@article{csdnet2025,
  title={CSDNet: Content-Structure Dual-aware Network for Image Super-Resolution},
  author={...},
  journal={...},
  year={2025}
}
```
