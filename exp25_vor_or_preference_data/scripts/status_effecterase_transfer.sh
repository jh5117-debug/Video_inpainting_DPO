#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python exp25_vor_or_preference_data/scripts/monitor_effecterase_transfer.py --once
cat reports/effecterase_download_status.md

