# Exp46 Exp45 Pseudo-Success Readback

## Environment

- Host: `instance-afs92r3e`
- Branch: `research/exp46-h20-minimax-exp45-pseudosuccess-sft-20260629`
- Start HEAD: `feef1b73317bea903e0e247d077d84c740665fa4`
- Exp45 PAI branch: `origin/research/exp45-pai-minimax-pair-scaleup-20260629`
- Exp45 PAI HEAD: `d0c8430a5ba35f37415ed52d53040829ef1123d6`
- Worktree: `/home/nvme01/H20_Video_inpainting_DPO_exp46_minimax_pseudosuccess_sft`
- Status before readback edits: `clean`

## Read Sources

H20 local sources read:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/57_exp43_h20_minimax_stage2_sft_runner.md`
- `experiment_registry/exp43_h20_minimax_stage2_sft_runner/*`
- `reports/exp43_h20_bf16_safe_preflight.md`
- `reports/exp43_h20_stage2_sft_ladder.md`
- `reports/exp43_exp44_pseudosuccess_path_validation.*`

PAI Exp45 branch sources read through git object store:

- `PRD/59_exp45_pai_minimax_pair_scaleup.md`
- `experiment_registry/exp45_pai_minimax_pair_scaleup/*`
- `reports/exp45_pair_scaleup_readback.md`
- `reports/exp45_targeted_scaleup_mining.*`
- `reports/exp45_visual_relabel.*`
- `reports/exp45_stage2_formal_handoff.*`
- `reports/exp45_h20_required_filelist.txt`
- `reports/exp45_h20_required_sha256.txt`
- `reports/exp45_h20_handoff_package.*`
- `reports/exp45_minimax_paper_positioning.md`

## Answers

1. Exp45 finished PAI-only pair scale-up and formal Stage2 handoff packaging. It reported `MINIMAX_STAGE2_FORMAL_DATA_READY`, with no PAI training and no optimizer step.
2. Final Exp45 formal split is pseudo-success train/search/shadow = 64/24/24; GT and preference views are also packaged but are not first-run training targets for Exp46.
3. H20 must mirror the files listed in `reports/exp45_h20_required_filelist.txt` with checksums from `reports/exp45_h20_required_sha256.txt`. The filelist has 326 entries and the checksum file has 327 lines.
4. Prior H20 validation found condition and mask paths present but pseudo-success target paths missing before the Exp45 mirror. Current Exp46 must mirror Exp45 targeted mining outputs before manifest validation.
5. GT-only SFT must not run first because Exp43 GT-style 30-step SFT was strongly quality-negative on shadow: full PSNR -6.5506, mask -4.2232, boundary -5.3735, outside -8.4532, Ewarp +0.5934.
6. Exp46 should reuse the existing Exp43 isolated Stage2 MiniMax SFT ladder runner, adding only Exp46 wrappers/configs under `exp46_h20_minimax_pseudosuccess_sft/`.
7. Precision recommendation is BF16-safe only if preflight passes: MiniMax/LoRA bf16, VAE/loss/targets/residuals/region reductions fp32, GradScaler disabled for bf16. If BF16 fails, use fp32 for this experiment only.
8. H20 GPU snapshot is recorded below. GPU0 currently has nonzero memory, GPUs1-7 are effectively free in the snapshot; final training allocation must be decided after the formal preflight and process audit.
9. The 30-step gate is PROMISING only if shadow full PSNR improves by at least +0.02 or mask/boundary improve with full not worse than -0.02, boundary/outside not worse than -0.02, LPIPS not worse >0.001, Ewarp not worse >0.05, visual better/tie >=18/24, worse <=4/24, no systemic fogging/over-erasure/boundary/outside damage, and not dominated by one scene.
10. Forbidden this round: GT-only SFT, DPO, 300/500/1000/2000-step, VOR-Eval, hard comp, shared trainer edits, `inference/metrics.py` edits, MiniMax official source edits, PAI writes/GPU/process changes, MiniMax third-backbone positive claim, universal adapter claim, final SOTA claim.

## Exp45 Data Summary

- New candidates mined: 72
- Strict visual relabel: SUCCESS_CLEAN=8, SUCCESS_USABLE including clean=28, FAILURE_MEDIUM_HARD=22
- Final same-source pair rows: 112
- Scene overlap: 0
- VOR-Eval: excluded
- Hard comp: excluded

## H20 GPU Snapshot

```text
index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, 28 MiB, 97871 MiB, 0 %
1, 1 MiB, 97871 MiB, 0 %
2, 1 MiB, 97871 MiB, 0 %
3, 1 MiB, 97871 MiB, 0 %
4, 1 MiB, 97871 MiB, 0 %
5, 1 MiB, 97871 MiB, 0 %
6, 1 MiB, 97871 MiB, 0 %
7, 1 MiB, 97871 MiB, 0 %
```

```text
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command 
# Idx           #    C/G      %      %      %      %      %      %    name 
    0     883875     G      -      -      -      -      -      -    Xorg           
    1          -     -      -      -      -      -      -      -    -              
    2          -     -      -      -      -      -      -      -    -              
    3          -     -      -      -      -      -      -      -    -              
    4          -     -      -      -      -      -      -      -    -              
    5          -     -      -      -      -      -      -      -    -              
    6          -     -      -      -      -      -      -      -    -              
    7          -     -      -      -      -      -      -      -    -              
```

## Git Log At Start

```text
feef1b7 Block Exp43 pseudo-success preflight on missing targets
c603cc7 Add Exp43 SFT-A generated ladder reports
4549acb Record Exp43 MiniMax SFT-A 30-step blocker
eda52a8 Implement Exp43 isolated Stage2 SFT ladder runner
b51723f Prepare Exp43 H20 MiniMax data splits
c399822 Implement and test Exp43 BF16-safe SFT runner preflight
401d7a4 Add Exp43 H20 Stage2 SFT runner readback
03ce2eb Audit H20 GPU release for Exp43

```
