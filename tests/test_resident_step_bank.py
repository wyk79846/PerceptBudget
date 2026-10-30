"""Resident banks retain exact overlapping shapes without accumulating tasks."""
import pytest
from types import SimpleNamespace
from perceptbudget.rtc_source_step_graph import SourceStepBank


def test_resident_keeps_overlap_and_builds_only_missing(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    bank=SourceStepBank.__new__(SourceStepBank)
    bank.resident=True
    released=[]
    retained=(object(),object())
    bank.entries={645:(object(),SimpleNamespace(release=lambda:released.append(645))),646:retained}
    bank.audits=[dict(valid=n,beta=5.) for n in bank.entries]
    built=[]
    def prepare(lengths,noise,beta,progress):
        built.extend(lengths)
        bank.entries.update({n:object() for n in lengths})
    bank._prepare=prepare
    bank.prepare_resident([646,647],None,beta=5.)
    assert built==[647]
    assert released==[645]
    assert bank.entries[646] is retained
    assert set(bank.entries)=={646,647}
    assert bank.audits==[dict(valid=646,beta=5.,reused=True)]
    with pytest.raises(ValueError):bank.prepare_resident(range(18),None,beta=5.)
    with pytest.raises(ValueError):bank.prepare_resident([646],None,beta=4.)


def test_source_graph_release_resets_native_graph_before_dropping_buffers():
    from perceptbudget.rtc_source_step_graph import SourceStepGraph
    graph=SourceStepGraph.__new__(SourceStepGraph)
    graph.outputs=object();events=[]
    graph.graph=SimpleNamespace(reset=lambda:events.append(graph.outputs is not None))
    graph.full_graph=SimpleNamespace(reset=lambda:events.append(graph.outputs is not None))
    graph.context=graph.expert=graph.condition_cache=object()
    graph.release()
    assert events==[True,True]
    assert graph.graph is graph.full_graph is graph.outputs is graph.context is graph.expert is None
    graph.release()  # Safe when cleanup is called again after an exception.
