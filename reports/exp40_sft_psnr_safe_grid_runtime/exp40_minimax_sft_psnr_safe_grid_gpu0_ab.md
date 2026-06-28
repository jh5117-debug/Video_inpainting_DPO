# Exp40 MiniMax PSNR-Safe SFT Warmup Grid Worker

Status: `MINIMAX_SFT_PSNRSAFE_NEGATIVE_AT_30STEP`
Worker: `gpu0_ab`
Scope: `S0`
Steps: `30`
Train rows: `64`
Search rows: `24`

This worker uses raw output as the primary evaluation output. Diagnostic comp is not used.

Metric note: LPIPS/Ewarp are not produced by this existing MiniMax runner; no substitute values are invented.

Recipe summary:
- `SFTmA_S0_lr3em05`: `NUMERIC_GATE30_FAIL`, full `-2.303278`, mask `-1.711663`, boundary `-1.975436`, outside `-3.365938`, temporal `0.584767`
- `SFTmA_S0_lr0.0001`: `NUMERIC_GATE30_FAIL`, full `-7.172223`, mask `-6.615492`, boundary `-6.301120`, outside `-8.013239`, temporal `0.475373`
- `SFTmA_S0_lr0.0003`: `NUMERIC_GATE30_FAIL`, full `-13.827806`, mask `-10.202384`, boundary `-10.424003`, outside `-16.048283`, temporal `7.137381`
- `SFTmB_S0_lr3em05`: `NUMERIC_GATE30_FAIL`, full `-2.306074`, mask `-1.666656`, boundary `-1.952630`, outside `-3.355257`, temporal `0.595906`
- `SFTmB_S0_lr0.0001`: `NUMERIC_GATE30_FAIL`, full `-6.963161`, mask `-5.897037`, boundary `-5.718433`, outside `-8.264182`, temporal `0.436655`
- `SFTmB_S0_lr0.0003`: `NUMERIC_GATE30_FAIL`, full `-10.849289`, mask `-9.839992`, boundary `-9.675967`, outside `-12.674501`, temporal `7.373321`
