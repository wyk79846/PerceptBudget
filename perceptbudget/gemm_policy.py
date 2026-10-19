"""Optional reproducible first-heuristic selection; no FlashRT source changes.

This fixes candidate selection, not cross-device/version numerical equivalence.
It is a new numerical baseline, not a replay of previously autotuned winners.
"""


class FirstHeuristicGemm:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def autotune_bf16_nn(self, *args, **kwargs):
        # Native signature has seven arguments including optional num_algos.
        if len(args) > 6:
            args = (*args[:6], 1)
            kwargs.pop('num_algos', None)
        else:
            kwargs['num_algos'] = 1
        return self.native.autotune_bf16_nn(*args, **kwargs)


def apply_policy(pipe, policy):
    if policy == 'autotune':
        return
    if policy != 'heuristic-first':
        raise ValueError('Unknown native GEMM policy: ' + str(policy))
    if pipe.pipeline is not None:
        raise RuntimeError('GEMM policy must be installed before pipeline creation')
    pipe.gemm = FirstHeuristicGemm(pipe.gemm)
