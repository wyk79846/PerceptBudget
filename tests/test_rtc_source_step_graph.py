from types import SimpleNamespace
import pytest


def test_h50_state_bank_covers_full_numeric_token_variation():
    from perceptbudget.rtc_source_step_graph import h50_state_prefix_lengths
    class Tokenizer:
        def encode(self,text,add_bos):
            return [0]*(133 if 'State: 0 ' in text else 149)
    tokenizer=Tokenizer()
    lengths=h50_state_prefix_lengths(657,'task',tokenizer)
    assert lengths==list(range(645,662)) and 653 in lengths
    for bad in (True,511,713,657.0):
        with pytest.raises(ValueError):h50_state_prefix_lengths(bad,'task',tokenizer)
    with pytest.raises(ValueError):h50_state_prefix_lengths(640,'task',tokenizer)


def test_h50_number_token_width_bound_matches_installed_tokenizer():
    from pathlib import Path
    path=Path('/home/wyk/.cache/flash_rt/paligemma_tokenizer.model')
    if not path.exists():pytest.skip('local tokenizer unavailable')
    import sentencepiece as sp
    tokenizer=sp.SentencePieceProcessor(model_file=str(path))
    widths=[len(tokenizer.encode(' '+str(i))) for i in range(-1,256)]
    assert min(widths)==2 and max(widths)==4
    from perceptbudget.rtc_source_step_graph import h50_state_prefix_lengths
    lengths=h50_state_prefix_lengths(657,'pick up the ketchup and place it in the basket',tokenizer)
    assert lengths==list(range(645,662))



@pytest.mark.parametrize('beta',[1.,2.,5.,10.])
def test_step_beta_matches_eager_reference_at_zero_time(beta):
    import math
    import torch
    from perceptbudget.rtc_source_step_graph import SourceStepGraph
    from perceptbudget.openpi_reference import sample_original_rtc
    expert=SimpleNamespace(horizon=20,num_steps=1,velocity=lambda x,k: .1*x)
    noise=torch.ones(20,32)
    previous=torch.zeros(15,32)
    graph=SourceStepGraph.__new__(SourceStepGraph)
    graph.expert=expert
    graph.x=noise.clone().requires_grad_(True)
    graph.target=torch.zeros_like(noise)
    i=torch.arange(20,dtype=torch.float32)
    weights=((5-1-i)/(15-5+1)+1).clamp(0,1)
    graph.mask=torch.where(i>=15,0,weights*torch.expm1(weights)/(math.e-1))[:,None]
    graph.tau=torch.tensor(0.)
    graph.beta_value=beta
    graph.beta=torch.tensor(beta)
    got=graph._step()[0]
    ref,_=sample_original_rtc(expert,noise,previous,5,5,beta=beta,author_order=True)
    torch.testing.assert_close(got,ref,rtol=0,atol=0)


@pytest.mark.parametrize('beta',[0.,-1.,float('nan'),float('inf')])
def test_step_beta_rejects_invalid_values_before_capture(beta):
    from perceptbudget.rtc_source_step_graph import SourceStepGraph
    with pytest.raises(ValueError,match='beta'):
        SourceStepGraph(None,None,beta=beta)


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
    for lengths in ([],[0],[713],range(10)):
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
