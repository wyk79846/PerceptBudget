from unittest.mock import Mock

from perceptbudget import rtc_memory


def test_graph_progress_does_not_collect_while_headroom_is_available(monkeypatch):
    monkeypatch.setattr(rtc_memory,'memory_record',lambda *args:dict(available_mib=1800))
    cleanup=Mock()
    value=rtc_memory.resident_memory_record(object(),'prefix_650',cleanup=cleanup,
        floor_mib=600,collect=False)
    cleanup.assert_not_called()
    assert value['available_mib']==1800 and not value['cleanup_performed']


def test_low_memory_collects_and_reports_the_recovered_headroom(monkeypatch):
    samples=iter([500,950])
    monkeypatch.setattr(rtc_memory,'memory_record',lambda *args:dict(available_mib=next(samples)))
    cleanup=Mock()
    value=rtc_memory.resident_memory_record(object(),'prefix_650',cleanup=cleanup,
        floor_mib=600,collect=False)
    cleanup.assert_called_once()
    assert value['available_mib']==950 and value['cleanup_performed']


def test_episode_boundary_still_collects_before_measuring(monkeypatch):
    events=[]
    def sample(*args):
        events.append('measure')
        return dict(available_mib=1800)
    monkeypatch.setattr(rtc_memory,'memory_record',sample)
    value=rtc_memory.resident_memory_record(object(),'environment_closed',
        cleanup=lambda *args:events.append('collect'),floor_mib=600)
    assert events==['collect','measure'] and value['cleanup_performed']
