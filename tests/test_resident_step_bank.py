"""Resident banks retain exact overlapping shapes without accumulating tasks."""
import pytest
import torch
from perceptbudget.rtc_source_step_graph import SourceStepBank


def test_resident_keeps_overlap_and_builds_only_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    bank=SourceStepBank.__new__(SourceStepBank)
    bank.resident=True
    retained=object()
    bank.entries={645:object(),646:retained}
    bank.audits=[dict(valid=n,beta=5.) for n in bank.entries]
    built=[]
    def prepare(lengths,noise,beta,progress):
        built.extend(lengths)
        bank.entries.update({n:object() for n in lengths})
    bank._prepare=prepare
    bank.prepare_resident([646,647],None,beta=5.)
    assert built==[647]
    assert bank.entries[646] is retained
    assert set(bank.entries)=={646,647}
    assert bank.audits==[dict(valid=646,beta=5.,reused=True)]
    with pytest.raises(ValueError):bank.prepare_resident(range(18),None,beta=5.)
    with pytest.raises(ValueError):bank.prepare_resident([646],None,beta=4.)
