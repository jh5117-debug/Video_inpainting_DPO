# Exp50 VOID 10-Step Heldout Generation

Time: 2026-07-01T08:11:22+08:00

Status: `VOID_ADAPTER_10STEP_GENERATION_READY`

## Protocol

- Used existing F2 Step0 official pass1 outputs.
- Created Step10 checkpoint by replacing only `proj_out.weight` and `proj_out.bias` from the adapter into a temporary pass1 safetensors checkpoint.
- Ran official `inference/cogvideox_fun/predict_v2v.py` on heldout4, split over requested free GPUs.
- No VOR-Eval, no hard comp, no training, no optimizer step in this generation step.

## Runs

- GPU2: seqs=['BLENDER_CON001_00742', 'REAL_ENV102_00001_002_02'] returncode=0 log=`/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/adapter_10step_h20_cache/heldout_evidence/gpu2_runtime_log.txt`
- GPU3: seqs=['BLENDER_CON001_00744', 'REAL_ENV200_00001_006_02'] returncode=0 log=`/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/adapter_10step_h20_cache/heldout_evidence/gpu3_runtime_log.txt`

## Outputs

- BLENDER_CON001_00742: generated frames=24 resolution=672x384 evidence=`/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/adapter_10step_h20_cache/heldout_evidence/evidence/BLENDER_CON001_00742`
- BLENDER_CON001_00744: generated frames=24 resolution=672x384 evidence=`/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/adapter_10step_h20_cache/heldout_evidence/evidence/BLENDER_CON001_00744`
- REAL_ENV102_00001_002_02: generated frames=24 resolution=672x384 evidence=`/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/adapter_10step_h20_cache/heldout_evidence/evidence/REAL_ENV102_00001_002_02`
- REAL_ENV200_00001_006_02: generated frames=24 resolution=672x384 evidence=`/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/adapter_10step_h20_cache/heldout_evidence/evidence/REAL_ENV200_00001_006_02`

## Safety

No root process was killed. Existing root GPU processes were left untouched; generation used requested free GPU memory.
