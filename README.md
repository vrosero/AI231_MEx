# AI231 — Machine Learning Operations

Course submission repository.

## MEx 1 — 3-Layer CNN for MNIST (einops/einsum)

`mex1.ipynb` — a 3-layer CNN where every layer/operation is implemented with
`einops`/`einsum` (no `nn.Conv2d` / `nn.Linear`). Trained 5 epochs on the DGX
cluster (A100-SXM4-40GB, AMP fp16).

| Metric | Value |
|---|---|
| Test accuracy | 97.35% |
| Grid (16 samples) | 15/16 correct |
| Parameters | 421,642 |
| GPU | A100-SXM4-40GB (auto-selected freest) |
| Precision | AMP fp16 |

**Architecture**
```
input (B, 1, 28, 28)
  → EinConv2d 1→32, k=3, p=1   [unfold + einsum 'bhwd,do->bhwo']
  → ReLU → EinMaxPool 2×2       [einops reduce 'max']
  → EinConv2d 32→64, k=3, p=1   [unfold + einsum]
  → ReLU → EinMaxPool 2×2       [einops reduce 'max']
  → einops rearrange 'b c h w -> b (c h w)'   [flatten]
  → EinLinear 3136→128          [einsum '...i,io->...o']
  → ReLU → EinLinear 128→10     [einsum]
```

**Artifacts**
- `mex1.ipynb` — executed notebook (figures embedded inline)
- `grid_4x4.png` — 16 test samples, ground truth vs prediction
- `curves.png` — training loss + train/test accuracy

**Environment:** PyTorch 2.13.0+cu130, einops 0.8.2, torchvision 0.28.0
**Node:** ai-n002.hpc.coe.upd.edu.ph (DGX, 8× A100-SXM4-40GB)
