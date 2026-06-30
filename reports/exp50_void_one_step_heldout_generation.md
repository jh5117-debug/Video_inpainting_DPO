# Exp50 VOID One-Step Heldout Generation

Time: 2026-07-01T00:59:58+08:00

Status: `VOID_ONE_STEP_HELDOUT_GENERATION_READY`

## Protocol

- Used existing F2 Step0 official pass1 outputs.
- Created Step1 checkpoint by replacing only `proj_out.weight` and `proj_out.bias` from the one-step adapter into a temporary pass1 safetensors checkpoint.
- Ran official `inference/cogvideox_fun/predict_v2v.py` on heldout4, split over GPU0/GPU1.
- No VOR-Eval, no hard comp, no training, no optimizer step.

## Runs

- GPU0: seqs=['BLENDER_CON001_00742', 'BLENDER_CON001_00744'] returncode=0 log=`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2/gpu0_runtime_log.txt`
- GPU1: seqs=['REAL_ENV102_00001_002_02', 'REAL_ENV200_00001_006_02'] returncode=0 log=`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2/gpu1_runtime_log.txt`

## Outputs

- BLENDER_CON001_00742: generated frames=24 resolution=672x384 evidence=`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2/evidence/BLENDER_CON001_00742`
- BLENDER_CON001_00744: generated frames=24 resolution=672x384 evidence=`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2/evidence/BLENDER_CON001_00744`
- REAL_ENV102_00001_002_02: generated frames=24 resolution=672x384 evidence=`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2/evidence/REAL_ENV102_00001_002_02`
- REAL_ENV200_00001_006_02: generated frames=24 resolution=672x384 evidence=`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2/evidence/REAL_ENV200_00001_006_02`

## Safety

No root process was killed. Existing root GPU processes were left untouched; generation used available free memory on GPU0/GPU1.
