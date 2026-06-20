# Exp24 Official Source Audit

Status: `ASSET_AUDIT_STARTED`.

Initial public-source audit performed on 2026-06-21.

| Model | Official/public source candidate | Current status |
|---|---|---|
| DiffuEraser | `https://github.com/lixiaowen-xw/diffueraser`, HF `lixiaowen/diffuEraser` | public source found; pending Exp24 revalidation |
| VideoPainter | `https://github.com/TencentARC/VideoPainter` | public source found; pending deploy |
| CoCoCo | `https://github.com/zibojia/COCOCO` | public source found; SD inpaint dependency still needs deploy audit |
| VideoComposer | `https://github.com/ali-vilab/videocomposer`; diffusers issue references `damo-vilab/videocomposer` | ambiguous naming/source; must disambiguate before deploy |
| VACE | `https://github.com/ali-vilab/VACE` | public source found; Wan/VACE native flow matching backend required |
| MiniMax-Remover | `https://github.com/zibojia/MiniMax-Remover` | public source found; native train forward audit pending |
| FloED | `https://github.com/NevSNev/FloED` / `FloED-main` search results | official complete code/weights not yet confirmed |
| EffectErase | `https://github.com/FudanCVL/EffectErase`, HF `FudanCVL/EffectErase` | public code/weights source found; VOR data remains `WAITING_AUTH` |
| ProPainter | `https://github.com/sczhou/ProPainter` | public source found; DPO status `NOT_APPLICABLE_NON_DIFFUSION_DPO` |

No checkpoint was downloaded in this initial commit.

