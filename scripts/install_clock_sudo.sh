#!/usr/bin/env bash
# One explicit administrator installation; no password stored, no GPU writes.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
pb_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
pb_prefix=/usr/local/lib/perceptbudget-clocks
pb_wrapper=/usr/local/sbin/perceptbudget-clocks
pb_policy=/etc/sudoers.d/perceptbudget-clocks

pb_rules() {
cat <<'RULES'
# PerceptBudget: fixed bounded clocks only; no arbitrary Python or shell.
wyk ALL=(root) NOPASSWD: /usr/local/sbin/perceptbudget-clocks inspect, /usr/local/sbin/perceptbudget-clocks restore, /usr/local/sbin/perceptbudget-clocks runtime --seconds 120, /usr/local/sbin/perceptbudget-clocks runtime3 --seconds 120, /usr/bin/true ""
RULES
}

if [[ ${1:-} == --dry-run ]]; then
    pb_rules
    exit 0
fi
if [[ $EUID != 0 ]]; then
    echo 'Run once: sudo bash scripts/install_clock_sudo.sh' >&2
    exit 1
fi
if [[ -e /run/perceptbudget-clocks/original.json ]]; then
    echo 'A clock lease/snapshot is active; finish or restore it before installation.' >&2
    exit 1
fi
if [[ ${1:-} == --remove ]]; then
    rm -f -- "$pb_policy"
    /usr/sbin/visudo -c
    echo 'Clock NOPASSWD authorization removed; root-owned helper retained for audit.'
    exit 0
fi
if [[ $# != 0 ]]; then
    echo 'Only --dry-run or --remove is accepted.' >&2
    exit 1
fi

for pb_path in /usr/local /usr/local/lib /usr/local/sbin /etc/sudoers.d; do
    [[ -d "$pb_path" && ! -L "$pb_path" ]] || exit 1
    if [[ $(stat -c %u "$pb_path") != 0 || -n $(find "$pb_path" -maxdepth 0 -perm /022 -print) ]]; then
        echo "Parent directory must be root-owned and not group/other writable: $pb_path" >&2
        exit 1
    fi
done
for pb_path in "$pb_prefix" "$pb_prefix/scripts" "$pb_prefix/perceptbudget"; do
    if [[ -e "$pb_path" || -L "$pb_path" ]]; then
        if [[ -L "$pb_path" || ! -d "$pb_path" || $(stat -c %u "$pb_path") != 0 || -n $(find "$pb_path" -maxdepth 0 -perm /022 -print) ]]; then
            echo "Existing helper directory is not protected: $pb_path" >&2
            exit 1
        fi
    fi
done
for pb_path in "$pb_prefix/scripts/control_device_clocks.py" "$pb_prefix/perceptbudget/device_clocks.py" "$pb_prefix/perceptbudget/__init__.py"; do
    [[ ! -L "$pb_path" ]] || exit 1
done
install -d -o root -g root -m 0755 "$pb_prefix/scripts" "$pb_prefix/perceptbudget"
install -o root -g root -m 0644 "$pb_root/scripts/control_device_clocks.py" "$pb_prefix/scripts/control_device_clocks.py"
install -o root -g root -m 0644 "$pb_root/perceptbudget/device_clocks.py" "$pb_prefix/perceptbudget/device_clocks.py"
# Root imports only the copied standard-library clock module, not the working repo.
printf '%s\n' '"""Root-owned clock-only package; no model or user configuration imports."""' > "$pb_prefix/perceptbudget/__init__.py"
chown root:root "$pb_prefix/perceptbudget/__init__.py"
chmod 0644 "$pb_prefix/perceptbudget/__init__.py"
pb_tmp=$(mktemp /usr/local/sbin/.perceptbudget-clocks.XXXXXX)
pb_rule_tmp=$(mktemp /etc/sudoers.d/.perceptbudget-clocks.XXXXXX)
trap 'rm -f -- "$pb_tmp" "$pb_rule_tmp"' EXIT
cat > "$pb_tmp" <<'WRAPPER'
#!/bin/sh
exec /usr/bin/python3 -I /usr/local/lib/perceptbudget-clocks/scripts/control_device_clocks.py "$@"
WRAPPER
chown root:root "$pb_tmp"
chmod 0755 "$pb_tmp"
mv -T -- "$pb_tmp" "$pb_wrapper"
sha256sum "$pb_prefix/scripts/control_device_clocks.py" "$pb_prefix/perceptbudget/device_clocks.py" "$pb_wrapper" > "$pb_prefix/installed.sha256"
chown root:root "$pb_prefix/installed.sha256"
chmod 0644 "$pb_prefix/installed.sha256"
pb_rules > "$pb_rule_tmp"
chown root:root "$pb_rule_tmp"
chmod 0440 "$pb_rule_tmp"
/usr/sbin/visudo -cf "$pb_rule_tmp"
mv -T -- "$pb_rule_tmp" "$pb_policy"
/usr/sbin/visudo -c
echo 'INSTALLED: exact120-second two/three-state clock protocols, inspect, restore, and /usr/bin/true.'
echo 'No model started, no frequencies changed. Other sudo commands still require authorization.'
