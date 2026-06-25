# Exp27 True-Model Objective Parity Readback

- milestone: true DiffuEraser policy/reference SDPO + Linear-DPO gate
- branch: `research/exp27-paper-grounded-preference-study`
- starting HEAD: `005e60342324c54520d87a2f72a90c933a4cd7bd`
- PAI hostname: `dsw-753014-85f54df947-bkp7h`
- permission state: `PAI_POSTMAINTENANCE_PERMISSIONS_RECOVERED`

Files read:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/49_exp27_paper_grounded_preference_study.md`
- `experiment_registry/exp27_paper_grounded_preference_study/status.md`
- `reports/exp27_nontrivial_parity_and_localdpo_smoke_20260624.md`
- `exp27_paper_grounded_preference_study/scripts/run_exp27_real_batch_parity.py`
- `exp27_paper_grounded_preference_study/scripts/scan_sdpo_real_distribution.py`
- `exp27_paper_grounded_preference_study/code/official_parity.py`
- `training/dpo/train_stage1.py`
- `training/dpo/dataset/generated_loser_manifest_dataset.py`

Source-of-truth assets:

- SFT-48000: `/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000`
- Exp11 outer b0.75 Stage1: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s1_2000_davis_pai/last_weights`
- BR GT-win manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/exp08c_youtubevos_gtwin_d3comp_lose_fixed_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`
- manifest identity check: first row is `generation_source=diffueraser_only`, `generation_model=diffueraser`, `canonical_num_frames=16`, `canonical_height=320`, `canonical_width=512`

Already completed:

- LocalDPO mask primitive passed.
- LocalDPO latent fusion/outside reinjection passed.
- LocalDPO 6-video corruption smoke passed.
- LocalDPO original loss 1/10-step plumbing passed.
- SDPO residual proxy scan passed, but it is not true policy/reference forward.
- Linear-DPO proxy parity passed, but it is not true policy/reference forward.

Pending in this milestone:

- Real DiffuEraser S0 parity state: policy = frozen SFT-48000 initialization, reference = identical SFT-48000.
- Real DiffuEraser S1 representative state: policy = Exp11 outer b0.75 Stage1, reference = frozen SFT-48000.
- SDPO official-vs-adapter lambda/loss/output-gradient parity.
- Linear-Frozen/EMA true-model probe based on real model MSE records.

Banned repeats and non-goals:

- Do not use the existing random tensor `run_exp27_real_batch_parity.py` as `TRUE_MODEL_PARITY`.
- Do not start RC-FPO.
- Do not start O0-O5 long objective studies.
- Do not run 50-step LocalDPO in this milestone.

Planned outputs:

- `reports/exp27_sdpo_true_model_forward_parity.md`
- `reports/exp27_sdpo_true_model_distribution_scan.csv`
- `reports/exp27_sdpo_true_model_summary.json`
- `reports/exp27_linear_true_model_parity.md`
- `reports/exp27_linear_true_model_parity.csv`
- PAI output under `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp27_paper_grounded_preference_study/`
