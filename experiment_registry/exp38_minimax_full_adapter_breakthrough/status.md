# Exp38 Status

Current status: `EXP38_READBACK_COMPLETED`

Exp38 starts from Exp37 and is scoped to MiniMax full adapter root-cause,
objective/data rescue, and DiffuEraser/VideoPainter paper-evidence packaging.

## 2026-06-28 Readback

- Branch: `research/exp38-minimax-full-adapter-breakthrough-20260628`.
- Start HEAD: `558c2f263469f4ee6ee46e2a1b26a8082515dded`.
- Base: `origin/research/exp37-minimax-localdpo-badnoise-rescue-20260627`.
- Exp30/35/36/37 PRDs, registries, reports, metrics, visual reviews, and
  relevant MiniMax scripts were read.
- MiniMax is not a checkpoint/load failure and not an ignored-weight failure.
- MiniMax remains plumbing-positive but quality-negative: Exp30 `0/32`, Exp35
  `0/48`, Exp36 `0/24`, and Exp37 only `1/16` visible improvement per recipe.
- PAI GPU0/GPU1 are physically free, while GPU2-GPU7 are occupied by other
  compute jobs. Legacy cli4 GPU lock files still exist.
- PAI Exp38 log root is writable; PAI experiment root under
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo` is not writable by
  `hj` during readback.
- No GPU task, training, RC-FPO, long run, or protected-lane action was
  launched.

Report:

- `reports/exp38_minimax_full_readback.md`

## 2026-06-28 Failure Taxonomy

Current status: `MINIMAX_FAILURE_TAXONOMY_BUILT`

- Training launched: false.
- GPU task launched: false.
- Code/loading failure is mostly ruled out.
- Inference ignoring adapter weights is ruled out.
- Objective signal too weak is the strongest current explanation.
- Bad-noise/timestep alignment, data difficulty, trainable-scope/update scale,
  and train-vs-heldout behavior remain open.
- Next allowed milestone: train-overfit diagnosis.

Reports:

- `reports/exp38_minimax_failure_taxonomy.md`
- `reports/exp38_minimax_failure_taxonomy.csv`
- `reports/exp38_minimax_decision_tree.json`

## 2026-06-28 Train-Overfit Diagnosis

Current status: `MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK_WITH_LOCAL_DRIFT`

- GPU0/GPU1 were explicitly reserved by the user and audited empty before use.
- No GPU0/GPU1 process was killed because no compute PID existed.
- GPU2/GPU3/GPU4 running jobs were untouched.
- Exp37 R1 LocalDPO-badnoise checkpoint-10 was evaluated on GPU0.
- Exp36 S1 winner-SFT checkpoint-10 was evaluated on GPU1.
- Exp37 R1 moves outputs but introduces mixed outside/global drift; not
  quality-positive.
- Exp36 S1 is near no-change on both train and heldout.
- Next milestone: LocalDPO v2 and bad-noise v2; 30-step remains locked.

Reports:

- `reports/exp38_minimax_train_overfit_diagnosis.md`
- `reports/exp38_minimax_train_overfit_metrics.csv`
- `reports/exp38_minimax_train_overfit_visual_review.csv`
- `reports/exp38_minimax_train_overfit_summary.json`

## 2026-06-28 LocalDPO v2 Filtered Pool

Current status: `MINIMAX_LOCALDPO_V2_FILTERED_POOL_READY`

- PAI run completed under `localdpo_v2_20260628`; the outer SSH session ended
  with `143` after outputs were already written.
- Codex reviewed 48 selected review sheets.
- Raw train32/heldout16 selected pool contains 5 visually trivial rows.
- Filtered pool excludes trivial rows and keeps train30 + heldout13.
- VOR-Eval was not used.
- No new training was launched in this milestone.

Manifests:

- `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_train30_filtered.jsonl`
- `exp38_minimax_full_adapter_breakthrough/manifests/localdpo_v2_heldout13_filtered.jsonl`

Next milestone: bad-noise v2 hard-state mining on the filtered pool using
GPU0/GPU1.
