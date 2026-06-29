# Exp43 / Exp44 Pseudo-Success H20 Path Validation

Status: `H20_EXP43_EXP44_PSEUDOSUCCESS_PREFLIGHT_BLOCKED_MISSING_TARGETS`

## Check

- H20 host: `instance-afs92r3e`
- H20 time: `2026-06-29T17:45:55+08:00`
- Exp43 branch: `research/exp43-h20-minimax-stage2-sft-runner-20260629`
- Exp43 HEAD checked on H20: `c603cc75d9702fbb85d38c39035359e9cd93a0f4`
- Exp44 source branch fetched on H20:
  `origin/research/exp44-pai-minimax-targeted-same-source-mining-20260629`
- Requested first experiment: pseudo-success SFT 30-step
- Explicitly forbidden first experiment: GT-only SFT

## Result

The Exp44 pseudo-success manifests were fetched and checked on H20. The
condition and mask assets are present through the H20 mirror path, but the
pseudo-success target outputs are not present in either the original `/mnt/nas`
location or the H20 mirror.

| split | rows | condition original | condition mirror | mask original | mask mirror | target_frames original | target_frames mirror | target_mp4 original | target_mp4 mirror |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 24 | 0 | 24 | 0 | 24 | 0 | 0 | 0 | 0 |
| search | 8 | 0 | 8 | 0 | 8 | 0 | 0 | 0 | 0 |
| shadow | 8 | 0 | 8 | 0 | 8 | 0 | 0 | 0 | 0 |

## Decision

Do not launch the pseudo-success SFT 30-step preflight yet.

Launching now would either fail during data loading or silently force a GT-only
fallback, which is explicitly forbidden by the handoff instructions. No
training, optimizer step, GT-only SFT, DPO, longer run, or model update was
started.

## Required Next Step

Mirror the Exp44 targeted mining pseudo-success outputs to H20 before retrying:

- PAI/NAS source root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`
- H20 mirror destination:
  `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`

After that mirror is present, rerun the path validation. If all
`target_frames_dir` paths resolve, run only the pseudo-success SFT 30-step
preflight.
