"""Native render buffers must be gone before the next reset allocates them."""
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('synchronize', [False, True])
def test_explicit_sim_free_is_ordered_and_render_destruction_is_idempotent(monkeypatch, synchronize):
    from perceptbudget.device_quality import install_render_buffer_cleanup
    events = []
    class Context:
        gl_ctx = type('GL', (), {'make_current': lambda self: events.append('current')})()
        con = type('Buffers', (), {'free': lambda self: events.append('buffers')})()
    class Sim:
        def free(self):
            events.append('sim')
    binding = ModuleType('robosuite.utils.binding_utils')
    binding.MjRenderContext, binding.MjSim = Context, Sim
    utils = ModuleType('robosuite.utils')
    utils.binding_utils = binding
    suite = ModuleType('robosuite')
    suite.utils = utils
    for name, module in [('robosuite', suite), ('robosuite.utils', utils),
                         ('robosuite.utils.binding_utils', binding)]:
        monkeypatch.setitem(sys.modules, name, module)
    gl = ModuleType('OpenGL.GL')
    gl.glFinish = lambda: events.append('finish')
    monkeypatch.setitem(sys.modules, 'OpenGL.GL', gl)
    import ctypes
    trim = Mock(side_effect=lambda unused: events.append('trim'))
    monkeypatch.setattr(ctypes, 'CDLL', lambda unused: type('Lib', (), {'malloc_trim': trim})())
    audit = install_render_buffer_cleanup(synchronize_release=synchronize)
    context = Context()
    sim = Sim()
    sim._render_context_offscreen = context
    sim.free()
    context.__del__()  # Old sensor closures may keep the same context alive.
    expected = ['current', 'buffers', 'finish', 'sim', 'trim'] if synchronize else ['current', 'buffers', 'sim']
    assert events == expected
    assert audit['freed'] == 1 and audit['errors'] == []
    assert audit['synchronized_frees'] == audit['host_trim_calls'] == int(synchronize)


def test_ram_stop_keeps_worker_evidence_and_terminates_only_child(tmp_path, monkeypatch):
    import json
    from scripts import guarded_run
    from scripts.profile_clock_screen import monitored_worker
    values = iter([2**40, 1])
    monkeypatch.setattr(guarded_run, 'available_ram_bytes', lambda: next(values))
    with pytest.raises(RuntimeError, match='RAM protection stopped worker'):
        monitored_worker([sys.executable, '-c', 'import time; time.sleep(30)'], None,
                         request_watchdog=tmp_path/'request_watchdog.json')
    record = json.loads((tmp_path/'ram_protection.json').read_text())
    assert record['available_bytes'] == 1 and record['floor_bytes'] == guarded_run.RAM_FLOOR_BYTES
    assert record['pid'] > 0 and record['worker_status']['VmRSS']
    assert not __import__('pathlib').Path(f"/proc/{record['pid']}").exists()
