# Experiment Artifact Registry

Updated: 2026-06-04

This registry repairs the experiment naming and artifact-tracking gap found during
the 2026-06-04 audit. It separates **Old Exp5**, **New Exp5**, and **New Exp6**.
There is no standalone ordinary "Exp6" in the weekly story; the run previously
called Exp6 should be presented as **New Exp6 / no-comp diagnostic**.

Audit inputs:

- H20 scan repo: `/home/nvme01/H20_Video_inpainting_DPO`
- H20 scan outputs: `reports/experiment_artifact_audit/*.txt`
- Local qualitative inventory: `/home/hj/dpo-2-1-exp`
- PAI artifacts: pending manual search; current known location hint from user is
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs`.

Status values:

- `complete`: independent folder, checkpoints or eval outputs, dpo-diag, and report evidence found.
- `partial`: some artifacts found, but one required artifact class is missing.
- `missing`: expected experiment artifact not found in current H20/HAL scan.
- `deleted`: qualitative artifact was reported or observed as removed/incomplete.
- `needs manual PAI search`: likely lives on PAI and must be completed from user-returned scan files.

| Experiment | Expected name | Found dirs | dpo-diag found | key reports | qualitative videos | checkpoint dirs | status | action |
|---|---|---|---|---|---|---|---|---|
| Exp4 | `official_videodpo_diffueraser_data_fullmask_loser` / `exp4-data` | H20 training folder not found. Local visual-only folder exists: `/home/hj/dpo-2-1-exp/exp4-data` with 4 PNG files and 0 MP4 files. | Not found in H20 scan. | Referenced in PPT/story as full-mask generated-loser negative result; no full report located. | Only PNG evidence locally; no video folder with MP4. | None found. | partial / artifact incomplete | Keep as negative data-quality smoke. Do not present as complete training. Search PAI for full generated-loser data and any dpo-diag. |
| Old Exp5 | `exp5_d2_comp_k4_stage1/stage2_full` and `exp5_d2_comp_k4_beta10_s1s2_4000` | H20 current scan found scripts/PRD references but not complete run folder. Local visual folder: `/home/hj/dpo-2-1-exp/exp5`. | Not found in H20 scan; PRD records diagnostic fields and collapse interpretation. | `PRD/00_current_status.md`, `PRD/01_experiment_matrix.md`, `PRD/07_exp5_exp6_winner_anchored_rerun.md`, `PRD/dpo_diagnostics_and_metrics_plan.md`. | `/home/hj/dpo-2-1-exp/exp5` has MP4 side-by-side samples. | Not found in current H20 tree. | partial / needs manual PAI search | Treat as collapsed diagnostic only. Search PAI for original run folder, VBench output, and dpo-diagnostics CSV. |
| New Exp5 | `exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` | H20 current scan found launcher and PRD references only. Local visual folder: `/home/hj/dpo-2-1-exp/new-exp5`. | Not found in H20 scan. | `PRD/07_exp5_exp6_winner_anchored_rerun.md` records objective and qualitative interpretation. | `/home/hj/dpo-2-1-exp/new-exp5` has MP4 samples. | Not found in current H20 tree. | partial / needs manual PAI search | Keep separate from Old Exp5. Search PAI for full Stage1/Stage2 run, dpo-diagnostics, qual30, and VBench outputs. |
| New Exp6 | `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` | H20 found multiple run folders under `experiments/dpo/stage1/20260531_*` and `20260601_004753_*`, plus stage2 `20260601_004753_*`; local visual folder: `/home/hj/dpo-2-1-exp/new-exp6`. | Found: `experiments/dpo/stage1/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage1/dpo_diagnostics.csv`; found stage2 dpo diag too. | `logs/analysis/new_exp6_prompt_length`, `PRD/07_exp5_exp6_winner_anchored_rerun.md`. | `/home/hj/dpo-2-1-exp/new-exp6` has MP4 samples. | Found checkpoint-3000/checkpoint-4000 for final H20 stage1 run; stage2 diagnostics found. | partial / complete on H20 for core training | Present as **New Exp6**, not ordinary Exp6. Summarize dpo-diag in PPT/PRD; PAI search can still fill missing eval reports if any. |
| Exp7 | `exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` | H20 current scan found scripts/PRD references; local visual folders: `/home/hj/dpo-2-1-exp/exp7-gate1500`, `/home/hj/dpo-2-1-exp/Exp7_DPO_Stage1_last`, `/home/hj/dpo-2-1-exp/Exp7_DPO_S1_DPO_S2_last`, `/home/hj/dpo-2-1-exp/Hybrid_DPO_S1_last__Official_DiffuEraser_base_Stage2`. | Full H20 dpo-diag not found in current H20 path scan; partial-mask eval metrics/report were previously generated on NAS/PAI side. | `PRD/dpo_diagnostics_and_metrics_plan.md`, `PRD/07_exp5_exp6_winner_anchored_rerun.md`, prior `reports/exp7_dpoS1_sftS2_hybrid_eval_report.md` if present on PAI/H20. | Local Exp7 and hybrid MP4 samples exist. | Not found in current H20 scan. | partial / needs manual PAI search | Keep Exp7 as task-alignment gate. Search PAI/H20 NAS for `exp7_gate1500_*`, `partialmask_eval`, and dpo diagnostics. |
| Exp8 | `exp8_youtubevos_d3_comp_regionloss_wingap_lose025_stage1_gate1500` | H20 scan found launcher only: `scripts/launch_exp8_youtubevos_d3_comp_regionloss_wingap_stage1_gate_pai.sh`. | Not found. | `PRD/12_exp8_regionloss_and_exp9_nolose_plan.md` records plan. | None found. | None found. | missing / planned or blocked | Do not claim result. If PAI launched it, user must return PAI scan outputs. |
| Exp9 | `exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500` and H20 nocomp/no-lose gates | H20 found `experiments/dpo/stage1/20260603_131758_exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20_stage1` and `20260604_080411_exp9_youtubevos_d3_comp_wingap_nolose_stage1_gate1000_h20_stage1`; local visual folders `exp9_d3_comp_gate_pai_ckpt500`, `exp9_d3_nocomp_gate`. | Found for H20 nocomp and no-lose: `dpo_diagnostics.csv`. PAI clean comp dpo-diag still needs PAI scan. | H20 target eval found: `logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243`; no-lose report found: `reports/exp9_nolose_gate_h20_report.md`. | Local comp/nocomp MP4 samples exist. | Found checkpoint-500/1000/1500 for H20 nocomp; found checkpoint-500/1000 for H20 no-lose. | partial / H20 complete, PAI comp pending search | Present conclusion as early-window signal. Use PAI manual search to register clean comp ckpt500/1000/1500 and eval artifacts. |

## Immediate Registry Gaps

1. Old Exp5 and New Exp5 complete run folders are not visible in the current H20
   scan, even though qualitative MP4 folders exist locally.
2. Exp4 has only a small local PNG evidence folder; no complete training/eval
   artifact was located.
3. Exp7 partial-mask eval and hybrid reports need PAI/NAS artifact confirmation.
4. Exp8 must not be presented as completed unless PAI returns a run folder and
   dpo-diag.
5. PAI manual artifact search is required before finalizing a complete registry.

