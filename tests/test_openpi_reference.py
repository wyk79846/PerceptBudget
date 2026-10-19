def test_prefix_coordinate_conversion_and_valid_crop():
    import torch
    from perceptbudget.openpi_reference import original_prefix
    split = torch.arange(18*7*256, dtype=torch.float32).reshape(18, 7, 1, 256)
    adjacent = split.reshape(18, 7, 1, 2, 128).transpose(-1, -2).flatten(-2)
    value = split+1
    key, out_value = original_prefix(adjacent, value, 5)
    assert key.shape == (18, 1, 1, 5, 256)
    assert torch.equal(key[:, 0].transpose(1, 2), split[:, :5])
    assert torch.equal(out_value[:, 0].transpose(1, 2), value[:, :5])


def test_author_sampling_matches_independent_toy_reference():
    import torch
    from perceptbudget.openpi_reference import sample_original_rtc
    from perceptbudget.rtc_guidance import paper_reference
    class Toy:
        horizon, num_steps = 20, 5
        @staticmethod
        def velocity(x, step):
            return .3*x+.1*x.roll(1, 0)+.01*step
    torch.manual_seed(7)
    noise, previous = torch.randn(20, 32), torch.randn(15, 32)
    output, norms = sample_original_rtc(Toy(), noise, previous, 8, 5)
    reference, ref_norms = paper_reference(lambda x,k: -Toy.velocity(x,k)/5,
                                         noise, previous, delay=8, executed=5, num_steps=5)
    torch.testing.assert_close(output, reference, atol=5e-6, rtol=5e-6)
    torch.testing.assert_close(torch.tensor(norms), torch.tensor(ref_norms), atol=5e-5, rtol=5e-6)


def test_original_bootstrap_is_unguided_original_expert():
    import torch
    from perceptbudget.openpi_reference import sample_original_rtc
    class Constant:
        horizon, num_steps = 20, 5
        @staticmethod
        def velocity(x, step):
            return torch.ones_like(x)
    output, norms = sample_original_rtc(Constant(), torch.zeros(20, 32), None, 0, 0)
    torch.testing.assert_close(output, -torch.ones(20, 32))
    assert norms == []
