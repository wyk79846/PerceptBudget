#!/usr/bin/env bash
# Continue the same frozen frequency comparison; retain the two completed tasks.
set -euo pipefail
pb_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$pb_root"
exec bash scripts/run_m4_ssh.sh \
  --acceptance-group m3 --low-full --full-frequency-pair \
  --task-indices 2 4 5 6 7 8 9 10 11 \
  --frequency-reference-run /home/wyk/workspace/任务产出/perceptbudget/M4/20261102T083327.472973Z-frequency-pairs-recovered-f322ac57 \
  --critical-wait --critical-wait-phases --first-refresh-step 30 --phase-refresh "$@"
