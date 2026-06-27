# Exp30 DiffuEraser / ProPainter Candidate Stack Audit

Date: 2026-06-27

Status:

- `DIFFUSERASER_VERIFIED_STACK_FOUND_EXP30_WRAPPER_PORT_REQUIRED`
- `PROPAINTER_CANDIDATE_ASSETS_READY`
- `NEW_GENERATORS_SMOKE2_PENDING`

## Scope

This milestone audits whether DiffuEraser and ProPainter can be added back to
Exp30 Smoke16 v3 without silently changing generator identity or relying on a
broken asset path. No model inference, Smoke16 v3, Smoke32, Gate64, adapter
gate, or training was launched.

Protected lanes remained read-only. The PAI audit observed Exp31 still running
on GPU1 and cli4 locks reserving GPU1-GPU4; Exp30 did not send signals or
modify protected outputs.

## DiffuEraser

Exp25 contains verified quality evidence for a DiffuEraser OR stack:

- Source report:
  `reports/exp25_diffueraser_or_root_cause_matrix_v2.md`
- Matrix CSV:
  `reports/exp25_diffueraser_or_root_cause_matrix_v2.csv`
- Decision:
  `DIFFUSERASER_NATIVE_OR_STACK_USABLE`

The strongest verified stack is:

- stack id: `DE-B_sft_raw6_d8_propainter`
- core checkpoint:
  `/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000`
- PCM: none
- steps: 6
- guidance: 0.0
- prior: ProPainter
- mask dilation: 8
- hard comp: no
- raw loser: yes

Exp25 v2 matrix result:

| stack | ok | medium-hard | hard-plausible | trivial-bad | mean mask PSNR | mean outside PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DE-A_sft_canonical_raw6_d0_propainter` | 12/12 | 8 | 4 | 0 | 20.654364 | 29.116529 |
| `DE-B_sft_raw6_d8_propainter` | 12/12 | 9 | 3 | 0 | 21.977292 | 28.808167 |

However, the current Exp30 wrapper
`DPO_finetune/infer_diffueraser_candidate.py` still calls
`inference/run_OR.py` with the legacy PCM-parameterized path. That is not the
same generator identity as the Exp25 explicit no-PCM overlay wrapper
`exp25_vor_or_preference_data/scripts/infer_diffueraser_or_exp25.py`.

Therefore Exp30 cannot yet label DiffuEraser as an enabled candidate family.
The safe next action is to port the explicit no-PCM overlay wrapper into
`exp30_vor_or_multimodel_minimax/scripts/`, then run a two-sample Exp30 smoke
before allowing DiffuEraser in Smoke16 v3.

## ProPainter

The Exp30 ProPainter wrapper is present:

- `DPO_finetune/infer_propainter_candidate.py`

The official ProPainter runtime expects:

- `ProPainter.pth`
- `raft-things.pth`
- `recurrent_flow_completion.pth`

PAI has a valid complete ProPainter weight directory:

`/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter`

PAI observed files:

- `ProPainter.pth`
- `raft-things.pth`
- `recurrent_flow_completion.pth`

Important path warning:

- `/mnt/nas/hj/weights/propainter` contains only
  `raft-things.pth.corrupt_20260607_233725` and must not be used.
- `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/propainter/current` points to
  the valid third-party directory above.

HAL also has a complete local copy under
`/home/hj/Video_inpainting_DPO/weights/propainter`, with SHA256:

- `ProPainter.pth`:
  `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591`
- `raft-things.pth`:
  `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`
- `recurrent_flow_completion.pth`:
  `22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283`

ProPainter can enter a two-sample smoke after the Exp30 PAI worktree/runtime
snapshot is created or synchronized. It is not yet allowed into Smoke16 v3
without that smoke.

## Runtime Readiness

PAI code snapshot path expected by Exp30:

`/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp30_vor_or_minimax`

Current audit result:

- PAI Exp30 worktree: missing at audit time.
- Exp30 Smoke16 v2 materialized inputs: available on NAS.
- DiffuEraser core checkpoint: readable on PAI.
- ProPainter valid weight path: readable on PAI.

This means generator assets are mostly ready, but runtime has to be synced
before GPU smoke. The sync must be limited to the Exp30 worktree and must not
touch cli4, Exp31, or Exp33.

## Decision

DiffuEraser:

`DIFFUSERASER_VERIFIED_STACK_FOUND_EXP30_WRAPPER_PORT_REQUIRED`

ProPainter:

`PROPAINTER_CANDIDATE_ASSETS_READY`

Smoke16 v3:

`BLOCKED_PENDING_NEW_GENERATORS_SMOKE2`

Gate64 / Smoke32 / MiniMax adapter:

remain stopped.

