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
