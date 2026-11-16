#!/usr/bin/env bash
# Run from the workspace root; launched inside tmux after explicit approval.
set -uo pipefail
topic=topics/2026-11-16-sra-selfflow-mnist
out="$topic/runs/smoke"
mkdir -p "$out"
/home/pc/lab/envs/main/bin/python "$topic/code/run.py" \
  --exp smoke --seeds 0 --data-root /mnt/d/Datasets \
  --out "$out" --classifier "$out/classifier.pt" 2>&1 | tee "$out/smoke.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$out/exit.code"
exit "$status"
