# Exp20 BF07/P4 Visual Review

- visual_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/exp20_bf07_p4_visuals`
- contact_sheets: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/exp20_bf07_p4_visuals/anonymous_contact_sheets`
- local review copies: `/home/hj/tmp_exp20_bf07_p4_visuals`
- method mapping: A=Exp11-S1, B=EQ_P0, C=Exp11-S2, D=EQ_P4, E=EQ_BF07, F=EQ_AD04, G=SFT
- status: `CODEX_REVIEWED`

## Summary

BF07 does not show a stable visible improvement over P4 on shadow-dev. Across all 20 contact sheets:

| Judgement | Count |
|---|---:|
| BF07 visibly better than P4 | 0 |
| P4 visibly better than BF07 | 4 |
| Tie / no reliable visual difference | 16 |
| BF07 new obvious artifact | 0 |
| P4 new obvious artifact | 0 |

The common pattern is that P4 and BF07 are visually very close. BF07's lower Ewarp in several rows does not correspond to a visible reduction in flicker, better moving boundaries, or clearer mask texture in the contact sheets. In thin-structure or high-motion cases, BF07 is sometimes slightly softer than P4.

P4 also does not show a stable visual advantage over Exp11-S1 on shadow-dev. This matches the quantitative result: P4 is below shadow Exp11-S1 in three-seed mean PSNR and does not pass the preregistered promotion gate.

## Notable Cases

- `0d62fa582a`: tiger/fence; BF07 slightly softens fence and stripe detail compared with P4.
- `75a55907dc`: snow/ski; BF07 is slightly more washed/soft at the object edge.
- `d3b25a44b3`: deer/fence; BF07 softens thin fence structure.
- `e228ce16fd`: fast dog/person foreground; BF07 is slightly softer on foreground and board edges.
- `237c6ebaf4`, `8473ae2c60`, `bb2245ab94`, `f94cd39525`: representative ties; no robust BF07 improvement despite metric-level Ewarp differences in some cases.

## Decision Impact

Visual review does not rescue BF07 or P4 after the shadow-dev statistical gate failed. The final visual conclusion is:

`NO_VISUAL_PROMOTION_SIGNAL`

Therefore Exp20 should stop after multiseed shadow confirmation and should not enter the 500-step gate.

| video | BF07 vs P4 | P4 vs Exp11 | notes |
|---|---|---|---|
| 00fef116ee | tie | tie_no_clear_gain | Cyclist road scene; P4 and BF07 are nearly identical. |
| 0b6db1c6fd | tie | tie_no_clear_gain | Person/animal outdoor case; no stable BF07 advantage. |
| 0d62fa582a | p4_better | tie_no_clear_gain | Tiger/fence fine structure; BF07 is slightly softer. |
| 237c6ebaf4 | tie | tie_no_clear_gain | Horse/rider scene; no visible edge or seam improvement. |
| 264ca20298 | tie | tie_no_clear_gain | Green texture scene; no robust BF07 improvement. |
| 5e130392e1 | tie | tie_no_clear_gain | Indoor perspective case; no clear difference. |
| 75a55907dc | p4_better | tie_no_clear_gain | Snow/ski; BF07 is slightly more washed/soft. |
| 7eb9424a53 | tie | tie_no_clear_gain | Sky/action-cam case; tied. |
| 8467aa6c5c | tie | tie_no_clear_gain | Snowboard/skier scene; tied. |
| 8473ae2c60 | tie | tie_no_clear_gain | Motorbike; BF07 Ewarp gain is not visually obvious. |
| aeb9de8f66 | tie | tie_no_clear_gain | Cycling/crowd scene; negligible differences. |
| b5c525cb08 | tie | tie_no_clear_gain | High-contrast cloth/flag case; tied. |
| bb2245ab94 | tie | tie_no_clear_gain | Small animal/hand; inconsistent local sharpness. |
| d3b25a44b3 | p4_better | tie_no_clear_gain | Deer/fence; BF07 slightly softens fence. |
| d76e963754 | tie | tie_no_clear_gain | Indoor red/black structure; tied. |
| dbd729449a | tie | tie_no_clear_gain | Snow walking case; tied. |
| e228ce16fd | p4_better | tie_no_clear_gain | Fast small dog/person; BF07 slightly softer. |
| ea8a5b5a78 | tie | tie_no_clear_gain | Indoor sports/wide shot; tied. |
| f94cd39525 | tie | tie_no_clear_gain | Stroller/dog shadow case; tied. |
| fc0db37221 | tie | tie_no_clear_gain | Dog/road case; tied. |
