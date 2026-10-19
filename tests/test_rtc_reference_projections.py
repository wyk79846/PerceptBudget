def test_reference_projection_route_preserves_fp32_expert_input():
    import torch
    from perceptbudget.rtc_expert import DifferentiableExpert
    expert = DifferentiableExpert.__new__(DifferentiableExpert)
    expert.device = torch.device('cpu')
    expert.dtype = torch.bfloat16
    expert.chunk_size, expert.num_steps = 20, 5
    expert._context = {}
    observed = []
    def step(x, index):
        observed.append(x.dtype)
        return x*2
    expert._step_delta = step
    noise = torch.full((20, 32), .123456, requires_grad=True)
    expert.reference_projections = None
    expert.step_delta(noise, 0)
    expert.reference_projections = {'diagnostic': True}
    output = expert.step_delta(noise, 0)
    assert observed == [torch.bfloat16, torch.float32]
    assert torch.equal(output, noise*2)
    gradient, = torch.autograd.grad(output.sum(), noise)
    assert torch.equal(gradient, torch.full_like(noise, 2))
