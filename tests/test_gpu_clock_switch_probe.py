from pathlib import Path
import pytest
from scripts.probe_gpu_clock_switch import HIGH,LOW,switch,summarize


class FakeControl:
    def __init__(self,path):
        self.gpu=path
        (path/'available_frequencies').write_text(f'{LOW} {HIGH}')
        self.target=HIGH
        self.other=2133000000
    def snapshot(self):
        return {'class/devfreq/17000000.gpu/min_freq':self.target,'emc':self.other}
    def bounds(self,low,high,minimum,maximum):
        assert low==self.gpu/'min_freq' and high==self.gpu/'max_freq'
        assert minimum==maximum
        self.target=minimum
    def read(self,path):
        assert path==self.gpu/'cur_freq'
        return self.target


def test_gpu_only_switch(tmp_path):
    control=FakeControl(tmp_path)
    row=switch(control,LOW)
    assert row['actual_hz']==LOW and row['non_gpu_settings_unchanged']
    assert row['command_to_readback_ms']>=row['settings_ms']>=0
    with pytest.raises(ValueError):switch(control,123)


def test_detects_non_gpu_change(tmp_path):
    control=FakeControl(tmp_path)
    original=control.bounds
    def changed(*args):
        original(*args)
        control.other=3199000000
    control.bounds=changed
    with pytest.raises(RuntimeError,match='Non-GPU'):
        switch(control,LOW)


def test_summary_requires_both_directions_and_complete_indices():
    rows=[dict(status='TRANSITION',direction=d,index=i,target_hz=LOW,actual_hz=LOW,
          non_gpu_settings_unchanged=True,command_to_readback_ms=float(i),settings_ms=1.)
          for d in ('up','down') for i in range(10)]
    assert summarize(rows)['up']['median_ms']==4.5
    with pytest.raises(ValueError):summarize(rows[:-1])
