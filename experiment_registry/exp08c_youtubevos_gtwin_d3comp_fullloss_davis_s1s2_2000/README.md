# Exp8c YouTube-VOS GT-Win D3-Comp Full-Loss DAVIS S1/S2

Exp8c keeps the D3 selected-primary-comp loser and mask unchanged, but replaces
the DPO winner with original YouTube-VOS GT frames aligned by
`canonical_frame_indices`.

This is a target-domain diagnostic, not a final success claim. It tests whether
using original aligned GT as the winner reduces DPO target mismatch relative to
the cached D3 winner path.

Status at 2026-06-06 12:36 CST: H20 formal Stage1 is running with the validated
fp32/nosplit configuration and has reached `global_step=150` with
`dpo_diagnostics.csv` present. The checked log tail had no `Traceback`,
`ERROR`, `OutOfMemory`, or `SIGFPE`.

PAI status at 2026-06-06: ready to launch from git-tracked code, not from
terminal-only patches.

- manifest tool: `tools/prepare_exp8c_gtwin_manifest.py`
- PAI launcher: `scripts/launch_exp8c_youtubevos_gtwin_d3comp_fullloss_s1s2_2000_davis_pai.sh`
- PAI YouTube-VOS root: `/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train`
- PAI source D3 manifest: `data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl`
- PAI generated manifest: `data/generated_losers/exp08c_youtubevos_gtwin_d3comp_lose_fixed_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`

PAI precision policy:

```text
MIXED_PRECISION=bf16
POLICY_DTYPE=bf16
VAE_DTYPE=fp32
REF_DTYPE=bf16
TEXT_DTYPE=bf16
SPLIT_POS_NEG_FORWARD=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Do not create or patch Exp8c code directly in a PAI terminal. Code changes must
start in HAL, be committed and pushed, then pulled/synced to H20 and PAI.
