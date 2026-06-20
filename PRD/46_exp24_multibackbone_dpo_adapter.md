# Exp24: Multi-Backbone VideoDPO / Diff-DPO Adapter

Status: `ASSET_AUDIT_STARTED`

Branch: `research/exp24-multibackbone-dpo-adapter`

HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp24_adapter`

PAI worktree target: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp24_adapter`

## Goal

Deploy and validate multiple public video inpainting / object removal backbones on PAI, then implement native DPO plumbing and adapters without forcing every model into DiffuEraser's epsilon-prediction objective.

## Target Models

1. DiffuEraser
2. VideoPainter
3. CoCoCo
4. VideoComposer / VideoComp
5. VACE
6. MiniMax-Remover
7. FloED
8. EffectErase
9. ProPainter

## Current Source Audit

Initial public sources were found for DiffuEraser, VideoPainter, CoCoCo, VACE, MiniMax-Remover, EffectErase, and ProPainter. VideoComposer needs project disambiguation. FloED needs confirmation that complete official code and weights are publicly released.

EffectErase VOR data status: `WAITING_AUTH`.

No formal smoke result has been recorded yet.

## 2026-06-21 Non-GPU Asset Check

While Exp23 owns GPU4-7, Exp24 remained in non-GPU asset/backend preparation.

Commands run:

```bash
bash exp24_multibackbone_dpo_adapter/scripts/asset_audit_status.sh
python -m py_compile exp24_multibackbone_dpo_adapter/backends/*.py
python -m unittest discover -s exp24_multibackbone_dpo_adapter/tests -p 'test_*.py'
```

Result:

- backend code compiles;
- backend status unit tests pass;
- EffectErase VOR remains `WAITING_AUTH`;
- no GPU inference or DPO smoke was started;
- GPU4-7 were not used by Exp24.

## Backend Contract

Implemented initial abstract interface:

`exp24_multibackbone_dpo_adapter/backends/base.py`

The interface requires native:

- policy/reference loading;
- native prediction target;
- shared noise/timestep;
- native error map;
- DPO loss;
- adapter save/load;
- base and adapter inference;
- checkpoint identity.

## Current Matrix

See:

- `reports/exp24_model_asset_matrix.csv`
- `exp24_multibackbone_dpo_adapter/reports/exp24_source_audit.md`

## Next Gate

One model at a time:

1. clone official repo;
2. pin commit/revision;
3. download public weights only;
4. checksum;
5. rsync to PAI;
6. create isolated environment;
7. run real inference smoke.

No model is marked ready until real inference succeeds.
