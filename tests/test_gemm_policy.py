from types import SimpleNamespace

import pytest

from perceptbudget.gemm_policy import FirstHeuristicGemm, apply_policy


def test_first_candidate_positional_and_keyword():
    calls = []
    native = SimpleNamespace(autotune_bf16_nn=lambda *a, **kw: calls.append((a, kw)), marker=3)
    proxy = FirstHeuristicGemm(native)
    proxy.autotune_bf16_nn(1, 2, 3, 4, 5, 6, 16)
    proxy.autotune_bf16_nn(1, 2, 3, 4, 5, 6, num_algos=16)
    assert calls == [((1, 2, 3, 4, 5, 6, 1), {}), ((1, 2, 3, 4, 5, 6), {'num_algos': 1})]
    assert proxy.marker == 3


def test_policy_guards():
    pipe = SimpleNamespace(pipeline=None, gemm=object())
    original = pipe.gemm
    apply_policy(pipe, 'autotune')
    assert pipe.gemm is original
    apply_policy(pipe, 'heuristic-first')
    assert pipe.gemm.native is original
    with pytest.raises(ValueError):
        apply_policy(pipe, 'unknown')
    pipe.pipeline = object()
    with pytest.raises(RuntimeError):
        apply_policy(pipe, 'heuristic-first')
