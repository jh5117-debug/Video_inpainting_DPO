# PAI Manual Artifact Search Commands

```bash
# PAI manual experiment registry audit. Copy/paste on PAI.
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
set -euo pipefail

REPORT=reports/pai_experiment_registry_audit.md
CSV=reports/pai_experiment_registry_paths.csv
mkdir -p reports
printf 'source,path,type,matched_keyword,note\n' > "$CSV"

KEY_RE='exp4|exp5|old_exp5|new_exp5|exp6|new_exp6|exp7|exp8|exp9|d2_comp|d2_nocomp|d3_comp|d3_nocomp|wingap|winner_gap|nolose|regionloss|youtubevos|davis'
ROOTS=(
  /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs
  /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/reports
  /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/PRD
  /mnt/nas/hj/H20_Video_inpainting_DPO/experiments
  /mnt/nas/hj/H20_Video_inpainting_DPO/logs
  /mnt/nas/hj/H20_Video_inpainting_DPO/reports
)

add_path() {
  local src="$1" path="$2" typ="$3" note="${4:-}"
  local key
  key=$(echo "$path" | grep -Eio "$KEY_RE" | head -1 || true)
  [ -n "$key" ] || key="matched_context"
  python - "$src" "$path" "$typ" "$key" "$note" >> "$CSV" <<'PYCSV'
import csv, sys
csv.writer(sys.stdout).writerow(sys.argv[1:])
PYCSV
}

for root in "${ROOTS[@]}"; do
  [ -e "$root" ] || continue
  while IFS= read -r p; do add_path "$root" "$p" "file_or_dir" "fixed-glob audit"; done < <(
    find "$root" -maxdepth 5 \( -type f -o -type d \) 2>/dev/null \
      | grep -Ei "$KEY_RE|dpo[-_]?diagnostics|diagnostics|metrics|vbench|checkpoint|stage1.log|stage2.log" \
      | sort
  )
done

{
  echo "# PAI Experiment Registry Audit"
  echo
  date
  echo
  echo "## Fixed Roots"
  printf -- '- `%s`\n' "${ROOTS[@]}"
  echo
  echo "## Summary"
  wc -l "$CSV"
  echo
  echo "## SFT-48000 Weight Check"
  for p in \
    /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000 \
    /mnt/nas/hj/weights/diffuEraser/converted_weights_step48000 \
    /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000
  do
    if [ -d "$p" ]; then
      echo "FOUND $p"
      du -sh "$p" || true
      find "$p" -maxdepth 2 -type f | sed -n '1,10p'
    else
      echo "MISSING $p"
    fi
  done
  echo
  echo "## DPO diagnostics candidates"
  grep -Ei 'dpo[-_]?diagnostics|diagnostics.*csv' "$CSV" | sed -n '1,120p' || true
  echo
  echo "## Exp7 / prior / mask hints"
  grep -Ei 'exp7|partialmask|propainter|prior|mask' "$CSV" | sed -n '1,160p' || true
  echo
  echo "## Exp5/NewExp5/NewExp6 hints"
  grep -Ei 'exp5|new_exp5|exp6|new_exp6|wingap' "$CSV" | sed -n '1,160p' || true
  echo
  echo "## Exp8/Exp9 target-domain hints"
  grep -Ei 'exp8|exp9|youtubevos|davis|regionloss|nolose' "$CSV" | sed -n '1,180p' || true
} | tee "$REPORT"

echo "REPORT=$REPORT"
echo "CSV=$CSV"

```
