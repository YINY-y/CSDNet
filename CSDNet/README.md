# CSDNet: Content-Structure Dual-aware Network for Image Super-Resolution

PyTorch implementation of CSDNet.

## Overview

CSDNet introduces **dual-aware routing** that jointly considers content similarity and
structural compatibility when grouping pixels into attention clusters. A lightweight
**Frequency Structure Encoder (FSE)** extracts local structure descriptors via DCT
decomposition, which guide both the routing process and a structure-aware fusion module.

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
