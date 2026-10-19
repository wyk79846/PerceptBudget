import subprocess
import sys
from pathlib import Path


def test_actual_mode_rejects_paused_and_ungated_combinations_before_gpu():
    root=Path(__file__).resolve().parents[1]
    for flags in (['--actual-delay','--controlled-delay-five'],
                  ['--actual-delay','--timing-only'],
                  ['--actual-delay'],['--experimental-varlen']):
        result=subprocess.run([sys.executable,'scripts/eval_rtc_openpi.py',*flags],
            cwd=root,capture_output=True,text=True)
        assert result.returncode == 2
        assert 'error:' in result.stderr
