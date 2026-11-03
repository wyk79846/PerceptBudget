"""Use the separately installed, root-owned bounded clock helper when available."""
from pathlib import Path
import subprocess

INSTALLED=Path('/usr/local/sbin/perceptbudget-clocks')
ROOT=Path(__file__).resolve().parents[1]


def helper_command(*args):
    if INSTALLED.is_file():
        st=INSTALLED.stat()
        if st.st_uid!=0 or st.st_mode&0o022 or INSTALLED.is_symlink():
            raise RuntimeError('Installed clock helper is not root-owned and protected')
        return ['sudo','-n',str(INSTALLED),*args]
    return ['sudo','-n','/usr/bin/python3',str(ROOT/'scripts/control_device_clocks.py'),*args]


def authorize():
    """A benign exact-command grant avoids requesting a general sudo timestamp."""
    result=subprocess.run(['sudo','-n','/usr/bin/true'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if result.returncode:
        subprocess.run(['sudo','-v'],check=True)
