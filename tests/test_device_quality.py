"""Plan and selection checks for a resident, paired M2 quality screen."""
from perceptbudget.device_quality import calibration_plan, select_candidate, SUITES


def rows(view, tid=0, outcome='success'):
    return [dict(suite=s, task_id=tid, view=view, outcome=outcome,
                 audit_passed=True, underflow_steps=0) for s in SUITES]


def test_calibration_budget_and_order():
    assert len(calibration_plan('full')) == 3
    assert len(calibration_plan('single')) == 6
    assert all(tid == 0 for _, tid, _ in calibration_plan('single'))


def test_holdout_cannot_choose_the_camera():
    full = rows('full')+rows('full', 3)
    fast = rows('agent')+rows('wrist')+rows('agent', 3, 'timeout')
    assert select_candidate(full, fast) == 'agent'


def test_new_failure_or_underflow_disqualifies_candidate():
    full = rows('full')
    agent = rows('agent')
    agent[0]['outcome'] = 'timeout'
    assert select_candidate(full, agent+rows('wrist')) == 'wrist'
    wrist = rows('wrist')
    wrist[0]['underflow_steps'] = 1
    assert select_candidate(full, agent+wrist) is None


def test_reference_must_pass_calibration():
    assert select_candidate(rows('full', outcome='timeout'), rows('agent')) is None


def test_single_view_shape_bank_is_exact_dual_view_bank_shift():
    from perceptbudget.rtc_source_step_graph import h50_state_prefix_lengths
    class Tokenizer:
        def encode(self, text, add_bos):
            return [0]*(133 if 'State: 0 ' in text else 149)
    full = h50_state_prefix_lengths(657, 'task', Tokenizer())
    single = h50_state_prefix_lengths(401, 'task', Tokenizer(), num_views=1)
    assert single == [x-256 for x in full]


def test_buffer_cleanup_does_not_destroy_egl_and_is_idempotent():
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from perceptbudget.device_quality import install_render_buffer_cleanup
    class Context:
        pass
    binding = SimpleNamespace(MjRenderContext=Context,MjSim=type('Sim',(),{'free':lambda self:None}))
    utils = SimpleNamespace(binding_utils=binding)
    with patch.dict(sys.modules, {'robosuite': SimpleNamespace(utils=utils),
        'robosuite.utils': utils,
        'robosuite.utils.binding_utils': binding}):
        audit = install_render_buffer_cleanup()
    context = Context()
    context.gl_ctx = SimpleNamespace(make_current=Mock(), free=Mock())
    context.con = SimpleNamespace(free=Mock())
    context.__del__()
    context.__del__()
    context.con.free.assert_called_once()
    context.gl_ctx.free.assert_not_called()
    assert audit['freed'] == 1 and audit['errors'] == []


def test_sim_free_releases_render_buffers_before_replacing_environment():
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch
    from perceptbudget.device_quality import install_render_buffer_cleanup
    events=[]
    class Context:
        pass
    class Sim:
        def free(self):
            events.append('sim_free')
    binding=SimpleNamespace(MjRenderContext=Context,MjSim=Sim)
    utils=SimpleNamespace(binding_utils=binding)
    with patch.dict(sys.modules,{'robosuite':SimpleNamespace(utils=utils),
        'robosuite.utils':utils,'robosuite.utils.binding_utils':binding}):
        record=install_render_buffer_cleanup()
    context=Context()
    context.gl_ctx=SimpleNamespace(make_current=lambda:events.append('current'))
    context.con=SimpleNamespace(free=lambda:events.append('buffer_free'))
    sim=Sim();sim._render_context_offscreen=context
    sim.free();context.__del__()
    assert events==['current','buffer_free','sim_free']
    assert record['freed']==record['explicit_sim_frees']==1


def test_shared_egl_factory_reuses_only_same_device_and_main_thread():
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    import pytest
    from perceptbudget.device_quality import install_render_buffer_cleanup
    class Context:
        pass
    gl = object()
    factory = Mock(return_value=gl)
    binding = SimpleNamespace(MjRenderContext=Context, GLContext=factory,MjSim=type('Sim',(),{'free':lambda self:None}))
    utils = SimpleNamespace(binding_utils=binding)
    with patch.dict(sys.modules, {'robosuite': SimpleNamespace(utils=utils),
        'robosuite.utils': utils, 'robosuite.utils.binding_utils': binding}):
        report = install_render_buffer_cleanup(shared_egl=True)
    assert binding.GLContext(max_width=640, max_height=480, device_id=0) is gl
    assert binding.GLContext(max_width=640, max_height=480, device_id=0) is gl
    assert report['egl_contexts_created'] == 1
    factory.assert_called_once()
    with pytest.raises(RuntimeError):
        binding.GLContext(max_width=640, max_height=480, device_id=1)


def test_completed_calibration_resume_does_not_load_a_model(tmp_path):
    import json
    from unittest.mock import patch
    from perceptbudget.device_quality import resident_quality
    from scripts import eval_q
    saved = dict(status='invalid', kind='full', episodes=[
        dict(suite=s, task_id=0, view='full', run=str(tmp_path/s)) for s in SUITES],
        model_loads=3, suffix_loads=3, error='guard125')
    (tmp_path/'full_quality.json').write_text(json.dumps(saved))
    with patch('scripts.audit_rtc_system.audit', return_value={'passed': True}), \
         patch.object(eval_q, '_check_cuda_and_native') as cuda:
        result = resident_quality({}, tmp_path, 'full', resume=True, calibration_only=True)
    cuda.assert_not_called()
    assert result['model_loads'] == 3
    assert result['status'] == 'completed' and result['calibration_only']
    assert len(result['episodes']) == 3
