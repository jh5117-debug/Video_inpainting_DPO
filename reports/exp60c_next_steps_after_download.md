# Exp60C Next Steps After Download

Status: `VIDEOPAINTER_VPDATA_SUBSET_READY_ON_PAI`

The repaired raw VPData subset is now complete on PAI/NAS and verified. The next exact milestone should be a separate PAI D3 mask generation gate using the PAI manifests only.

## Next Exact Milestone

1. Read `manifests/exp60c_vpdata_train1000_sources_pai.jsonl` and `manifests/exp60c_vpdata_test100_sources_pai.jsonl`.
2. Generate D3/partial-mask-style masks for train1000 only, with test100 held out.
3. Verify mask decode/count/area distribution and train/test separation.
4. Do not generate VideoPainter losers in the mask milestone unless separately authorized.
5. Do not train or run DPO until mask generation passes and a loser-generation gate is preregistered.

## Restrictions Still Active

- Do not use test100 for training, pair selection, threshold setting, or checkpoint selection.
- Do not download full VPData.
- Do not run VideoPainter loser generation, inference, DPO, or training in the next mask-only gate.
- Do not claim VPData validation, DPO positive, universal adapter, or final SOTA.
