from types import SimpleNamespace
from perceptbudget.rtc_visual_alternation import visual_geometry,FreshViewRTCAdapter
from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter
from perceptbudget.selective_pipeline import pipeline_type


def test_real_geometry_uses416_not_full_masking():
    assert visual_geometry('selective416')==dict(num_views=2,visual_tokens=416,vision_pool_factor=2)
    class Base:
        def _allocate_buffers(self):return self.encoder_seq_len,self.total_kv
    p=pipeline_type(Base)();p.num_views=2;p.vision_pool_factor=2;p.max_prompt_len=200;p.chunk_size=50
    assert p._allocate_buffers()==(616,666)
    assert p.vision_seq_enc==416


def test_stateful_prefix_families_dispatch_despite_range_overlap(monkeypatch):
    calls=[]
    class Bank:
        fallback='root'
        def __init__(self,lengths):self.entries={n:(n,n) for n in lengths}
        def select(self,n):return self.entries.get(n,('root',None))
    def prepare(self,lengths,progress=None):
        calls.append(list(lengths));self.step_bank=Bank(lengths)
        return dict(prepare_seconds=0.)
    monkeypatch.setattr(OriginalOpenPiRTCAdapter,'prepare_step_bank',prepare)
    a=object.__new__(FreshViewRTCAdapter)
    full=list(range(650,667));fast=[n-96 for n in full]
    a.config=dict(m2_visual_fast_mode='selective416',source_prefix_coverage_rule='h50_state8_tokenizer_bounds_v1',m2_visual_prefix_coverage=dict(full=full,fast=fast))
    a.torch=SimpleNamespace(cuda=SimpleNamespace(graph_pool_handle=object))
    a.prepare_step_bank(full+fast)
    assert calls==[full,fast]
    assert a.step_bank.select(560)==(560,560)
    assert a.step_bank.select(660)==(660,660)


def test416_uses_own_latency_calibration_and_recovery_prior():
    from perceptbudget.closed_loop import DeadlineRuntime
    costs=dict(service={'Fast':{h:dict(samples_seconds=[.3]) for h in ('hot','short','long')}})
    cfg=dict(m3_timing=dict(service_quantile=.95,safety_margin_seconds=.025),m3_events={},
        m3_service_samples=dict(hot=[.5],short=[.6],long=[.65]),m3_recovery_samples=[.7],
        m3_fast_enabled=True,m3_fast_cost_model=costs,m3_fast_selective416=True,
        m3_fast_service_calibration_seconds=[.4,.42,.41],rtc_initial_control_period_seconds=.05,
        H=50,m3_scheduler_mode='deadline_high')
    runtime=DeadlineRuntime(cfg)
    assert abs(runtime.fast_times.estimate('hot')-.42)<1e-9
    assert abs(runtime.fast_times.estimate('recovery')-.62)<1e-9
