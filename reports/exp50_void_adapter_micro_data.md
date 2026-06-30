# Exp50 VOID Adapter Micro Data

Status: `VOID_ADAPTER_MICRO_DATA_READY`

## Scope

- Source: VOR-Train Gate8 only.
- VOR-Eval: excluded.
- Training: not run.
- Optimizer step: not run.
- Hard comp: not used.
- Purpose: prepare the minimum train4/heldout4 data for the later G1 trainable-forward, zero-gap, and one-step gates.

## Split

- Train: 4 rows: ['BLENDER_CON001_00636', 'BLENDER_CON001_00843', 'REAL_ENV219_00001_003_05', 'REAL_ENV259_00102_002_04']
- Heldout: 4 rows: ['BLENDER_CON001_00742', 'BLENDER_CON001_00744', 'REAL_ENV102_00001_002_02', 'REAL_ENV200_00001_006_02']
- Scene overlap: no.
- Source-type balance: train 2 BLENDER / 2 REAL; heldout 2 BLENDER / 2 REAL.

## Loser Policy

- Same-model VOID raw losers: 2 medium-hard rows (`BLENDER_CON001_00636`, `REAL_ENV219_00001_003_05`).
- Controlled local-corruption losers: 6 rows where F2 output was usable or too-close. The corruption is a labeled local blur/tone/noise perturbation inside the object/affected quadmask region only; it is not used as a promoted output and is not a hard-comp claim.

## Outputs

- `manifests/exp50_void_adapter_train4.jsonl`
- `manifests/exp50_void_adapter_heldout4.jsonl`
- `reports/exp50_void_adapter_micro_data.csv`
- `reports/exp50_void_adapter_micro_data_summary.json`

## Next Gate

G1 may run only trainable forward, zero-gap, and one-step. Direct 10-step remains forbidden until one-step passes.

## Alignment Fix

Official VOID raw outputs contain 85 frames while Gate8 condition/winner videos contain 24 frames. For the two same-model medium-hard loser rows, G0 writes aligned loser derivatives using the first 24 raw VOID frames resized to the winner resolution. The manifest preserves the original `raw_void_loser_path` and marks `loser_alignment=first24_resized_to_winner_resolution`.
