# Exp44 Status

Current status: `MINIMAX_SAME_SOURCE_PAIR_GATE_PASSED`

## 2026-06-29 Readback

- Branch: `research/exp44-pai-minimax-targeted-same-source-mining-20260629`
- Base: `origin/research/exp42-pai-minimax-successful-removal-badnoise-20260629`
- Start HEAD: `efc07b2e9f5e84fe488b433b812a9f0bc72debaf`
- Exp42 overlap: `7` same-source success/failure scene groups.
- Plan locked: `56` total target groups across A/B/C/D.
- PAI GPU0/GPU1 were occupied by unrelated root-owned
  `qxq_sample_valtest_v0.py` jobs during readback; no cleanup was attempted.
- No GPU mining, training, optimizer step, H20 action, VOR-Eval use, or hard comp launched.

Reports:

- `reports/exp44_minimax_targeted_readback.md`
- `reports/exp44_source_group_plan.csv`
- `reports/exp44_source_group_plan.json`

## 2026-06-29 Targeted Source Manifest

- Status: `EXP44_TARGETED_SOURCE_MANIFEST_READY`
- Source rows: `40`
- Candidate budget: `452`
- Missing source rows: `0`
- Manifest: `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_targeted_source_manifest.jsonl`
- Manifest SHA256: `5147839e1e2d60e0ecc9c77a438a934918605b5fa550fa58d1e3291df7be168b`
- Runner: `exp44_pai_minimax_targeted_same_source_mining/scripts/mine_targeted_candidates.py`

No GPU inference or training has been launched by this preparation step.

## 2026-06-29 Targeted Second-Pass Mining

- Status: `MINIMAX_TARGETED_MINING_COMPLETED`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`
- GPU0 worker PID/PGID: `2263394` / `2263394`
- GPU1 worker PID/PGID: `2266642` / `2266642`
- Candidates completed: `452` / `452`
- Technical failures: `0`
- Auto success: `138`
- Auto medium-hard failure: `231`
- Auto boundary-bad: `31`
- Auto fogging/over-erasure: `25`
- Auto too-close: `27`
- Auto same-source pair capacity: `26`
- Auto overlap groups: `13`
- Training run: `false`
- Optimizer step: `false`
- VOR-Eval used: `false`
- Hard comp used: `false`
- Strict visual relabel required before pair construction: `true`

Reports:

- `reports/exp44_targeted_mining.md`
- `reports/exp44_targeted_mining_metrics.csv`
- `reports/exp44_targeted_mining_group_yield.csv`
- `reports/exp44_targeted_mining_summary.json`

## 2026-06-29 Strict Visual Relabeling

- Status: `MINIMAX_TARGETED_RELABEL_COMPLETED`
- Review pages opened: `47`
- Candidates relabeled: `452`
- Selected candidate page rows reviewed: `369`
- SUCCESS_CLEAN: `33`
- SUCCESS_USABLE: `92`
- FAILURE_MEDIUM_HARD: `137`
- Rejected / borderline / non-usable: `190`
- Same-source groups with usable success and medium-hard failure: `10`
- One-to-one same-source pair precheck: `18`
- Capped same-source combination precheck: `40`
- Training run: `false`
- Optimizer step: `false`
- VOR-Eval used: `false`
- Hard comp used: `false`

Reports:

- `reports/exp44_targeted_visual_relabel.md`
- `reports/exp44_targeted_visual_relabel.csv`
- `reports/exp44_targeted_visual_relabel_group_yield.csv`
- `reports/exp44_targeted_visual_relabel_summary.json`

Formal same-source pair construction remains pending Milestone D.

## 2026-06-29 Same-Source Pair Construction

- Status: `MINIMAX_SAME_SOURCE_PAIR_GATE_PASSED`
- Usable same-source pairs: `40`
- Minimum gate: `24`
- Target: `48`
- Train/search/shadow: `24` / `8` / `8`
- Split scene-group overlap: `0`
- Source groups with pairs: `10`
- Max pairs per group: `4`
- DPO winner field: GT background `V_bg`
- Pseudo-success field: retained for Stage2-style distillation metadata
- DPO loser field: same-source MiniMax `FAILURE_MEDIUM_HARD`
- Training run: `false`
- Optimizer step: `false`
- VOR-Eval used: `false`
- Hard comp used: `false`

Reports:

- `reports/exp44_same_source_pair_construction.md`
- `reports/exp44_same_source_pair_construction.csv`
- `reports/exp44_same_source_pair_group_yield.csv`
- `reports/exp44_same_source_pair_summary.json`

The pair gate unlocks bad-noise v4 state construction only, not SFT/DPO
training or MiniMax quality-positive claims.
