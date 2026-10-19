from types import SimpleNamespace
import pytest


def test_step_bank_miss_returns_slow_reference_without_capture():
    from perceptbudget.rtc_source_step_graph import SourceStepBank
    bank=SourceStepBank.__new__(SourceStepBank)
    fallback,expert,graph=object(),object(),object()
    bank.fallback=fallback
    bank.entries={566:(expert,graph)}
    assert bank.select(566)==(expert,graph)
    assert bank.select(569)==(fallback,None)


def test_step_bank_rejects_unbounded_or_invalid_preparation():
    from perceptbudget.rtc_source_step_graph import SourceStepBank
    for lengths in ([],[0],[593],range(10)):
        with pytest.raises(ValueError,match='eight exact prefix'):
            SourceStepBank(None,lengths,None)


def test_step_graph_cannot_be_mixed_with_padded_or_varlen_attention():
    import torch
    from perceptbudget.rtc_source_step_graph import SourceStepGraph
    with pytest.raises(ValueError,match='exact original attention'):
        SourceStepGraph(SimpleNamespace(context_capacity=592),torch.zeros(25,32))
    with pytest.raises(ValueError,match='exact original attention'):
        SourceStepGraph(SimpleNamespace(context_capacity=None,varlen_attention_enabled=True),torch.zeros(25,32))


def test_step_graph_rejects_replaced_context():
    import torch
    from perceptbudget.rtc_source_step_graph import SourceStepGraph
    graph=SourceStepGraph.__new__(SourceStepGraph)
    graph.expert=SimpleNamespace(horizon=25,keys=torch.zeros(1),values=torch.zeros(1),prefix_mask=torch.ones(1))
    graph.binding=tuple(t.data_ptr() for t in (graph.expert.keys,graph.expert.values,graph.expert.prefix_mask))
    graph.expert.keys=graph.expert.keys.clone()
    with pytest.raises(RuntimeError,match='context replaced'):
        graph.replay(torch.zeros(25,32),torch.zeros(20,32),5,5)


def test_historical_profile_only_supplies_shapes_and_rejects_different_policy(tmp_path):
    import json
    from perceptbudget.rtc_source_step_graph import lengths_from_reference_profile
    run=tmp_path/'run'
    (run/'rtc').mkdir(parents=True)
    config=dict(H=25,N=5,state_enabled=True,state_prompt_capacity=80,
        controlled_delay_steps=5,rtc_suite='libero_spatial',rtc_author_order=True,
        experimental_original_varlen=False,original_cuda_graph=False)
    (run/'run.json').write_text(json.dumps(dict(config_snapshot=config)))
    (run/'summary.json').write_text(json.dumps(dict(task_id=2,suite='libero_spatial')))
    (run/'rtc/q5_requests.json').write_text(json.dumps([dict(rtc_guidance=dict(valid_prefix_length=n)) for n in (568,566,567)]))
    profile=tmp_path/'summary.json'
    profile.write_text(json.dumps(dict(records=[dict(task_id=2,source_run=str(run))])))
    lengths,record=lengths_from_reference_profile(profile,task_id=2,suite='libero_spatial',horizon=25)
    assert lengths==[566,567,568]
    assert record['requests_sha256']
    with pytest.raises(ValueError,match='configuration mismatch'):
        lengths_from_reference_profile(profile,task_id=2,suite='libero_spatial',horizon=50)
