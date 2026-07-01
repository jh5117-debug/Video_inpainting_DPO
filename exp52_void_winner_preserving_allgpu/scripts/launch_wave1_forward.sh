#!/usr/bin/env bash
set -u
cd /home/nvme01/H20_Video_inpainting_DPO_exp52_void_allgpu_rescue

PY=/home/nvme01/conda_envs/void_exp50_official_v2/bin/python
REPO=/home/nvme01/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model
BASE=/home/nvme01/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP
VOID=/home/nvme01/H20_Video_inpainting_DPO/weights/void/netflix_void-model
CACHE=/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/cache/tensor_cache
OUT=/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu
LOGDIR=/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp52_void_winner_preserving_allgpu/wave1_forward
PIDFILE=/home/nvme01/H20_Video_inpainting_DPO/runtime/exp52_void_winner_preserving_allgpu/wave1_forward_pids.tsv

mkdir -p "$LOGDIR" "$(dirname "$PIDFILE")"
: > "$PIDFILE"
printf "cell\tgpu\tpid\tlog\n" >> "$PIDFILE"

cells=(R1_Q0_T500_S0 R1_Q2_T500_S0 R2_Q0_T500_S0 R2_Q2_T500_S0 R3_Q0_T500_S0 R3_Q2_T500_S0 R4_Q0_T500_S0 R4_Q2_T500_S0)
recipes=(R1_WinnerPreserve_LocalDPO R1_WinnerPreserve_LocalDPO R2_WinnerPreserve_LoserClip R2_WinnerPreserve_LoserClip R3_SDPO_Safe R3_SDPO_Safe R4_LinearDPO_vPrediction R4_LinearDPO_vPrediction)
variants=(q0_current q2_strict_affected q0_current q2_strict_affected q0_current q2_strict_affected q0_current q2_strict_affected)
gpus=(0 1 2 3 4 5 6 7)

for i in "${!cells[@]}"; do
  cell=${cells[$i]}
  recipe=${recipes[$i]}
  variant=${variants[$i]}
  gpu=${gpus[$i]}
  delay=$((i * 60))
  log="$LOGDIR/${cell}.log"
  (
    sleep "$delay"
    echo "START $(date -Ins) cell=$cell gpu=$gpu recipe=$recipe variant=$variant"
    PYTHONPATH=. "$PY" exp52_void_winner_preserving_allgpu/scripts/rescue_cell_forward.py \
      --repo "$REPO" \
      --base-model "$BASE" \
      --void-weights "$VOID" \
      --cache-root "$CACHE" \
      --output-root "$OUT" \
      --reports-dir reports \
      --cell "$cell" \
      --recipe "$recipe" \
      --variant "$variant" \
      --timestep 500 \
      --gpu "$gpu"
    rc=$?
    echo "END $(date -Ins) cell=$cell rc=$rc"
    exit "$rc"
  ) > "$log" 2>&1 &
  pid=$!
  printf "%s\t%s\t%s\t%s\n" "$cell" "$gpu" "$pid" "$log" >> "$PIDFILE"
  echo "launched $cell gpu=$gpu pid=$pid log=$log"
done
