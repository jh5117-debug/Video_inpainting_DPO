# Exp26 External 49F Clean-Source Inventory

- status: `EXP26_EXTERNAL_49F_INVENTORY_COMPLETE`
- candidate_count: `2024`
- valid_49f_count: `54`
- valid_davis_49f_count: `54`
- selected_count: `32`
- candidate_inventory_sha256: `03157fa156bcf9ee2d636812947fcff96f11e77d078c85df3d9d6cebee3ea9e1`
- selected_manifest_sha256: `be118a7ce7d462bda6c339053d0c112994c8da7fab6cf00a4ee5dae87b628e5a`

## Searched Roots

- `/mnt/workspace/hj/nas_hj/data/external/DPO_Finetune_data`
- `/mnt/nas/hj`
- `/mnt/workspace/hj/nas_hj`
- `/mnt/workspace/hj`
- `/home/hj`

## Source Decision

The only source family that passed the strict 49-frame clean-source gate
in this pass was the local DAVIS-derived frame set under
`DPO_Finetune_data/*/gt_frames`. The adjacent `comparison.mp4` files
were explicitly not used as clean sources because they are visualization
movies rather than raw clean input.

DAVIS is external to the locked VOR-BG train/search/shadow splits and is
suitable for internal held-out validation. Paper usage must cite and comply
with DAVIS terms.

## Selected Rows

| rank | sample_id | frames | resolution | subject | environment | motion |
| ---: | --- | ---: | --- | --- | --- | --- |
| 0 | `davis_bear` | 82 | 512x512 | animal | outdoor_mixed | low |
| 1 | `davis_bmx-bumps` | 90 | 512x512 | mixed_or_object | outdoor_mixed | high |
| 2 | `davis_boat` | 75 | 512x512 | mixed_or_object | water | low |
| 3 | `davis_boxing-fisheye` | 87 | 512x512 | human | outdoor_mixed | medium |
| 4 | `davis_breakdance-flare` | 71 | 512x512 | human | outdoor_mixed | high |
| 5 | `davis_bus` | 80 | 512x512 | vehicle | urban | medium |
| 6 | `davis_car-turn` | 80 | 512x512 | vehicle | urban | medium |
| 7 | `davis_cat-girl` | 89 | 512x512 | animal | outdoor_mixed | medium |
| 8 | `davis_classic-car` | 63 | 512x512 | vehicle | urban | low |
| 9 | `davis_color-run` | 84 | 512x512 | mixed_or_object | outdoor_mixed | medium |
| 10 | `davis_crossing` | 52 | 512x512 | mixed_or_object | urban | low |
| 11 | `davis_dance-jump` | 60 | 512x512 | human | outdoor_mixed | medium |
| 12 | `davis_disc-jockey` | 76 | 512x512 | mixed_or_object | outdoor_mixed | medium |
| 13 | `davis_dog-gooses` | 86 | 512x512 | animal | outdoor_mixed | medium |
| 14 | `davis_drift-turn` | 64 | 512x512 | mixed_or_object | outdoor_mixed | high |
| 15 | `davis_drone` | 91 | 512x512 | mixed_or_object | outdoor_mixed | medium |
| 16 | `davis_elephant` | 80 | 512x512 | animal | outdoor_mixed | medium |
| 17 | `davis_flamingo` | 80 | 512x512 | animal | outdoor_mixed | medium |
| 18 | `davis_hike` | 80 | 512x512 | human | foliage_grass | medium |
| 19 | `davis_hockey` | 75 | 512x512 | human | outdoor_mixed | high |
| 20 | `davis_horsejump-low` | 60 | 512x512 | mixed_or_object | outdoor_mixed | medium |
| 21 | `davis_kid-football` | 68 | 512x512 | human | outdoor_mixed | medium |
| 22 | `davis_kite-walk` | 80 | 512x512 | mixed_or_object | outdoor_mixed | medium |
| 23 | `davis_koala` | 100 | 512x512 | animal | foliage_grass | low |
| 24 | `davis_lady-running` | 65 | 512x512 | human | outdoor_mixed | medium |
| 25 | `davis_mallard-water` | 80 | 512x512 | animal | water | medium |
| 26 | `davis_miami-surf` | 70 | 512x512 | mixed_or_object | water | high |
| 27 | `davis_motocross-bumps` | 60 | 512x512 | mixed_or_object | outdoor_mixed | high |
| 28 | `davis_paragliding` | 70 | 512x512 | mixed_or_object | outdoor_mixed | medium |
| 29 | `davis_rhino` | 90 | 512x512 | animal | outdoor_mixed | low |
| 30 | `davis_scooter-board` | 91 | 512x512 | vehicle | urban | high |
| 31 | `davis_surf` | 55 | 512x512 | mixed_or_object | water | high |

## Gate

- rows >= 16: PASS
- train/search/shadow overlap: `0` by source family and manifest IDs
- VOR-Eval overlap: `0` by source family
- no model outputs inspected before selection
- no masks, seeds, or prompts generated yet; preregistration remains the next milestone
