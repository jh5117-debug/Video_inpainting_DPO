# Exp38 MiniMax Train-vs-Heldout Diagnosis

Status: **MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK_WITH_LOCAL_DRIFT**

GPU use: PAI GPU0 ran `exp37_r1_localdpo_badnoise`; GPU1 ran `exp36_s1_winner_sft`. GPU0/1 were empty before launch, no processes were killed, and no signal was sent. GPU2/3/4 existing jobs were not touched.

Runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp38_minimax_full_adapter_breakthrough/train_overfit_20260628`

## Results

### Exp37 R1 LocalDPO-badnoise checkpoint-10

Train32 mean deltas: full PSNR -0.5863, mask PSNR +0.1521, boundary PSNR +0.0691, outside PSNR -0.8950.

Heldout16 mean deltas: full PSNR +0.2008, mask PSNR +0.1619, boundary PSNR -0.0498, outside PSNR +0.0282.

Visual review: Step10 is not identical to Step0. It produces visible local/global changes, but many train samples show full/outside degradation or tonal drift. Heldout changes are mixed and not quality-positive.

### Exp36 S1 Winner-SFT checkpoint-10

Train32 mean deltas: full PSNR +0.0162, mask PSNR -0.0064, boundary PSNR -0.0017, outside PSNR +0.0249.

Heldout16 mean deltas: full PSNR -0.0102, mask PSNR -0.0083, boundary PSNR -0.0109, outside PSNR -0.0145.

Visual review: compact start/mid/end pages and representative full strips show near-identical Step0/Step10. This is not a meaningful train overfit.

## Diagnosis

Exp38 confirms that MiniMax can load adapted checkpoints and, under R1, can visibly move outputs. The failure is not a GPU execution issue and not an inference-load issue. The current issue is objective/data/update localization: R1 can move pixels but damages outside/global appearance; S1 is too weak to produce meaningful changes.

Next action: build LocalDPO v2 corruption with strict outside preservation and bad-noise v2 hard states before any SFT/DPO rescue. Do not unlock 30-step or long training.

## Artifacts

- Metrics CSV: `reports/exp38_minimax_train_overfit_metrics.csv`
- Visual review CSV: `reports/exp38_minimax_train_overfit_visual_review.csv`
- Summary JSON: `reports/exp38_minimax_train_overfit_summary.json`
- Runtime reports: `/home/hj/H20_Video_inpainting_DPO_exp38_minimax_full/reports/exp38_train_overfit_runtime`
- Compact review pages local cache: `reports/exp38_train_overfit_runtime/compact_review_pages/` (not intended as large Git artifact)
