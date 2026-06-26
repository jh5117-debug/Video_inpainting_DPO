# Exp29 Metric Summary

No metric-producing smoke or adapter gate has run yet.

MiniMax has verified PAI/NAS weights and is eligible for isolated inference
smoke. EffectErase is blocked before metrics because official weights were not
found.

## 2026-06-26 MiniMax Trainable Forward

- no-grad loss: 0.0171425510
- grad loss: 0.0171425510
- grad norm: 0.7473063172
- gradient tensors: 461
- peak VRAM: 12561.50 MiB
- missing keys: 0
- unexpected keys: 0

## 2026-06-26 MiniMax Adapter Gates

- zero-gap DPO loss: 0.6931471825
- one-step grad norm preclip: 0.8897291490
- one-step parameter delta probe: 2.0694979922992497e-11
- step10 parameter delta probe: 1.1061271569642785e-10
- step10 peak VRAM: 44614.22 MiB
- heldout `davis_hockey` PSNR delta: -0.0008006331
- heldout `davis_koala` PSNR delta: +0.0024137239
