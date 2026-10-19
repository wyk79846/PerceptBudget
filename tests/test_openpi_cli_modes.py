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


def test_horizon_50_is_not_silently_applied_to_fixed_h20_reference():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/eval_rtc_openpi.py','--horizon','50','--verify-only'],
        cwd=root,capture_output=True,text=True)
    assert result.returncode == 2
    assert 'requires explicit --actual-delay' in result.stderr


def test_horizon_25_is_recognized_but_requires_explicit_actual_mode():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/eval_rtc_openpi.py','--horizon','25','--verify-only'],
        cwd=root,capture_output=True,text=True)
    assert result.returncode == 2
    assert 'requires explicit --actual-delay' in result.stderr


def test_packed_conditioning_requires_explicit_source_graph_bank():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/eval_rtc_openpi.py','--source-packed-conditioning'],
        cwd=root,capture_output=True,text=True)
    assert result.returncode==2
    assert 'requires source-step-prefixes or profile' in result.stderr
