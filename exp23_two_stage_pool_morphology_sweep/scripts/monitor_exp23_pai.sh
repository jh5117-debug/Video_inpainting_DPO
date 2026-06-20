#!/usr/bin/env bash
set -euo pipefail

echo "Exp23 monitor"
date
hostname
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps -ef | grep -E "exp23|Phy|train_exp23" | grep -v grep || true

