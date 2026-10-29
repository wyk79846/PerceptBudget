#!/usr/bin/env bash
# Run inside tmux: preserve the sudo-authorized terminal across SSH disconnects.
set -uo pipefail

pb_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
pb_python=/home/wyk/venvs/lutvla/bin/python
cd -- "$pb_root" || exit 1

for pb_arg in "$@"; do
    if [[ "$pb_arg" == --dry-run ]]; then
        exec "$pb_python" -u scripts/run_m4_acceptance.py "$@"
    fi
done

if [[ -z "${TMUX:-}" || ! -t 0 ]]; then
    echo 'Start tmux first: tmux new -s pb-m4-v2' >&2
    exit 1
fi
if [[ "$EUID" == 0 ]]; then
    echo 'Run as the normal project user; only clock helpers use sudo.' >&2
    exit 1
fi

sudo -v || exit 1
pb_log="/tmp/perceptbudget-m4-ssh-$(date +%Y%m%d-%H%M%S).log"
echo "ssh_log=$pb_log"
(
    while sleep 60; do
        sudo -n -v || exit 1
    done
) &
pb_auth_pid=$!
trap 'kill "$pb_auth_pid" 2>/dev/null || true' EXIT

"$pb_python" -u scripts/run_m4_acceptance.py "$@" 2>&1 | tee "$pb_log"
pb_status=${PIPESTATUS[0]}
echo "M4_SSH_EXIT_CODE=$pb_status" | tee -a "$pb_log"
exit "$pb_status"
