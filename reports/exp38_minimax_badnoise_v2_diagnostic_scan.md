# Exp38 MiniMax Bad-Noise v2 Diagnostic Scan

Status: `MINIMAX_BAD_NOISE_STATES_READY`

Bad-noise v2 was run on the filtered LocalDPO v2 train30 pool only. This is a diagnostic/mining step, not training.

Runtime:

- GPU: `0`
- dtype: `bfloat16`
- train rows: `30`
- candidate states per row: `64`
- total candidate states: `1920`
- K_noise: `8`
- timesteps: `[0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95]`
- manifest SHA256: `22dbd28c776dcccf2b8b4e49bb81f17ebf79cfbee58867699471e65958b30bac`

Hard-state diagnostic:

- hard_A vs random gradient-proxy ratio mean: `0.563042`
- hard_A vs random gradient-proxy ratio max: `0.856340`
- hard_A vs random loser-local ratio mean: `0.330301`
- hard_A vs random loser-local ratio max: `0.705323`

Interpretation:

The state miner completed and produced auditable hard_state_A/B/C entries for all 30 filtered train rows. However, the selected hard_state_A does not amplify the proxy signal relative to random states on average. That means the next 10-step rescue is allowed as a bounded diagnostic, but expectations should remain low and 30-step remains locked unless the 10-step quality gate is actually positive.

Outputs:

- manifest: `exp38_minimax_full_adapter_breakthrough/manifests/badnoise_v2_train30_states.jsonl`
- CSV: `reports/exp38_minimax_badnoise_v2_diagnostic_scan.csv`
- summary: `reports/exp38_minimax_badnoise_v2_summary.json`
