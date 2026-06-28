# Exp40 MiniMax PSNR-Safe SFT Warmup Grid Worker

Status: `MINIMAX_SFT_PSNRSAFE_NEGATIVE_AT_30STEP`
Worker: `gpu1_cd`
Scope: `S0`
Steps: `30`
Train rows: `64`
Search rows: `24`

This worker uses raw output as the primary evaluation output. Diagnostic comp is not used.

Metric note: LPIPS/Ewarp are not produced by this existing MiniMax runner; no substitute values are invented.

Recipe summary:
- `SFTmC_S0_lr3em05`: `NUMERIC_GATE30_FAIL`, full `-1.816781`, mask `-1.634597`, boundary `-1.899575`, outside `-2.624405`, temporal `0.525111`
- `SFTmC_S0_lr0.0001`: `NUMERIC_GATE30_FAIL`, full `-6.736656`, mask `-5.878329`, boundary `-5.605923`, outside `-8.168581`, temporal `0.938385`
- `SFTmC_S0_lr0.0003`: `NUMERIC_GATE30_FAIL`, full `-14.573955`, mask `-10.318876`, boundary `-11.124060`, outside `-17.201737`, temporal `28.548663`
- `SFTmD_S0_lr3em05`: `NUMERIC_GATE30_FAIL`, full `-2.395389`, mask `-2.186769`, boundary `-2.440809`, outside `-3.140433`, temporal `0.414593`
- `SFTmD_S0_lr0.0001`: `NUMERIC_GATE30_FAIL`, full `-6.801210`, mask `-6.469689`, boundary `-6.300287`, outside `-7.698793`, temporal `1.156952`
- `SFTmD_S0_lr0.0003`: `NUMERIC_GATE30_FAIL`, full `-15.079110`, mask `-10.781672`, boundary `-11.678147`, outside `-17.713054`, temporal `24.738019`
