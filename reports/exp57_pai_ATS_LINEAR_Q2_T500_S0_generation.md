# Exp50 VOID Exp57 PAI ATS_LINEAR_Q2_T500_S0 Heldout Generation

Time: 2026-07-02T07:23:31+08:00

Status: `EXP57_PAI_HELDOUT_GENERATION_READY`

## Protocol

- Used existing F2 Step0 official pass1 outputs.
- Created Step1 checkpoint by replacing only `proj_out.weight` and `proj_out.bias` from the adapter into a temporary pass1 safetensors checkpoint.
- Ran official `inference/cogvideox_fun/predict_v2v.py` on heldout4, split over requested free GPUs.
- No VOR-Eval, no hard comp, no training, no optimizer step in this generation step.

## Runs

- GPU1: seqs=['BLENDER_CON001_00742', 'BLENDER_CON001_00744', 'REAL_ENV102_00001_002_02', 'REAL_ENV200_00001_006_02'] returncode=0 log=`/home/hj/exp57_void_adaptive_transition_pai_outputs/adaptive_video/ATS_LINEAR_Q2_T500_S0/gpu1_runtime_log.txt`

## Outputs

- BLENDER_CON001_00742: generated frames=24 resolution=672x384 evidence=`/home/hj/exp57_void_adaptive_transition_pai_outputs/adaptive_video/ATS_LINEAR_Q2_T500_S0/evidence/BLENDER_CON001_00742`
- BLENDER_CON001_00744: generated frames=24 resolution=672x384 evidence=`/home/hj/exp57_void_adaptive_transition_pai_outputs/adaptive_video/ATS_LINEAR_Q2_T500_S0/evidence/BLENDER_CON001_00744`
- REAL_ENV102_00001_002_02: generated frames=24 resolution=672x384 evidence=`/home/hj/exp57_void_adaptive_transition_pai_outputs/adaptive_video/ATS_LINEAR_Q2_T500_S0/evidence/REAL_ENV102_00001_002_02`
- REAL_ENV200_00001_006_02: generated frames=24 resolution=672x384 evidence=`/home/hj/exp57_void_adaptive_transition_pai_outputs/adaptive_video/ATS_LINEAR_Q2_T500_S0/evidence/REAL_ENV200_00001_006_02`

## Safety

No root process was killed. Existing root GPU processes were left untouched; generation used requested free GPU memory.
