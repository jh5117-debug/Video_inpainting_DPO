#!/usr/bin/env bash
set -euo pipefail

cd /home/nvme01/VideoDPO
mkdir -p /home/nvme01/VideoDPO_runs

PY="${PY:-/home/nvme01/conda_envs/videodpo/bin/python}"
RUN_TAG="${RUN_TAG:-vc2_dpo_videoinpainting_h20_gpu6-7_$(date +%Y%m%d_%H%M%S)}"
STDOUT="/home/nvme01/VideoDPO_runs/${RUN_TAG}.stdout.log"
PID_FILE="/home/nvme01/VideoDPO_runs/${RUN_TAG}.pid"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"

H20_DPO_ROOT="${H20_DPO_ROOT:-/home/nvme01/H20_Video_inpainting_DPO/DPO_Finetune_Data_Multimodel_v1}"
VIDEOINPAINTING_REPO="${VIDEOINPAINTING_REPO:-/home/nvme01/H20_Video_inpainting_DPO}"
MAX_STEPS="${MAX_STEPS:-10000}"
DPO_DIAG_EVERY="${DPO_DIAG_EVERY:-300}"
VAL_EVERY="${VAL_EVERY:-2000}"
VAL_MAX_VIDEOS="${VAL_MAX_VIDEOS:-1}"
VAL_DDIM_STEPS="${VAL_DDIM_STEPS:-25}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ ! -x "${PY}" ]]; then
  echo "[launch] python not executable: ${PY}" >&2
  exit 1
fi
if [[ ! -d "${H20_DPO_ROOT}" ]]; then
  echo "[launch] H20_DPO_ROOT not found: ${H20_DPO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${VIDEOINPAINTING_REPO}" ]]; then
  echo "[launch] VIDEOINPAINTING_REPO not found: ${VIDEOINPAINTING_REPO}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export PYTHONPATH=/home/nvme01/VideoDPO
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export VIDEODPO_LOG_DPO_DIAG=1
export VIDEODPO_DPO_DIAG_EVERY="${DPO_DIAG_EVERY}"
export VIDEODPO_CONSOLE_LOG_LEVEL="${VIDEODPO_CONSOLE_LOG_LEVEL:-INFO}"
export VIDEODPO_FILE_LOG_LEVEL="${VIDEODPO_FILE_LOG_LEVEL:-INFO}"
export VIDEOINPAINTING_REPO

CMD=(
  "${PY}" -m torch.distributed.run
  --nnodes=1
  --nproc_per_node=2
  --master_port "${MASTER_PORT}"
  scripts/train.py
  -t
  --devices "0,1"
  --base configs/vc2_dpo/config.yaml
  --name "${RUN_TAG}"
  --logdir /home/nvme01/VideoDPO_runs
  lightning.logger.target=pytorch_lightning.loggers.CSVLogger
  lightning.logger.params.name=csv
  lightning.logger.params.save_dir=/home/nvme01/VideoDPO_runs
  lightning.trainer.num_nodes=1
  lightning.trainer.devices=2
  lightning.trainer.accelerator=gpu
  lightning.trainer.max_steps="${MAX_STEPS}"
  lightning.trainer.max_epochs=1000
  lightning.trainer.accumulate_grad_batches=1
  lightning.trainer.log_every_n_steps=1
  lightning.trainer.precision=32
  lightning.callbacks.image_logger.params.batch_frequency=-1
  lightning.callbacks.metrics_over_trainsteps_checkpoint.params.every_n_train_steps="${VAL_EVERY}"
  lightning.callbacks.video_inpainting_metric_logger.target=utils.video_inpainting_validation.VideoInpaintingMetricLogger
  lightning.callbacks.video_inpainting_metric_logger.params.every_n_train_steps="${VAL_EVERY}"
  lightning.callbacks.video_inpainting_metric_logger.params.max_videos="${VAL_MAX_VIDEOS}"
  lightning.callbacks.video_inpainting_metric_logger.params.ddim_steps="${VAL_DDIM_STEPS}"
  lightning.callbacks.video_inpainting_metric_logger.params.unconditional_guidance_scale=12.0
  lightning.callbacks.video_inpainting_metric_logger.params.video_inpainting_repo="${VIDEOINPAINTING_REPO}"
  model.params.beta_dpo=5000.0
  model.params.log_every_t=100
  data.params.batch_size=1
  data.params.num_workers="${NUM_WORKERS}"
  data.params.train.target=data.video_inpainting_dpo_data.VideoInpaintingDPODataset
  data.params.train.params.data_root="${H20_DPO_ROOT}"
  "data.params.train.params.resolution=[320,512]"
  data.params.train.params.video_length=16
  data.params.train.params.frame_stride=1
  "data.params.train.params.caption=clean background"
  data.params.train.params.davis_oversample=10
)

echo "[launch] RUN_TAG=${RUN_TAG}"
echo "[launch] STDOUT=${STDOUT}"
echo "[launch] PID_FILE=${PID_FILE}"
echo "[launch] MASTER_PORT=${MASTER_PORT}"
echo "[launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[launch] H20_DPO_ROOT=${H20_DPO_ROOT}"
echo "[launch] VIDEOINPAINTING_REPO=${VIDEOINPAINTING_REPO}"
echo "[launch] MAX_STEPS=${MAX_STEPS} diag_every=${DPO_DIAG_EVERY} val_every=${VAL_EVERY}"
echo "[launch] PY=${PY}"
"${PY}" -V

setsid nohup "${CMD[@]}" > "${STDOUT}" 2>&1 < /dev/null &
echo $! > "${PID_FILE}"
disown

echo "STDOUT=${STDOUT}"
echo "PID_FILE=${PID_FILE}"
