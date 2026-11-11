"""Opt-in SAFE monitor: cumulative author score, one Full response per episode."""
import time


class SafeFailureMonitor:
    def __init__(self, expert, torch, checkpoint):
        from perceptbudget.failure_features import ExpertFeatureTap
        from perceptbudget.safe_paper import make_model
        self.torch = torch
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.record = saved['record']
        if saved.get('normalization') != 'none':
            raise ValueError('SAFE runtime requires the saved unnormalized feature contract')
        device = expert.suffix.action_out_proj.weight.device
        # Constructing the probe must not consume the policy's random numbers.
        with torch.random.fork_rng(devices=[]):
            self.model = make_model(self.record['width'], 'cpu')
        self.model.load_state_dict(saved['state_dict'])
        self.model.to(device).eval().requires_grad_(False)
        self.band = saved['cp_band'].detach().cpu().flatten().tolist()
        self.alpha = float(saved['alpha'])
        self.checkpoint = str(checkpoint)
        self.tap = ExpertFeatureTap(expert, torch, layout='corners')
        self.total = torch.zeros((), device=device, dtype=torch.float32)
        self.reset()

    def reset(self):
        self.total.zero_()
        self.count = 0
        self.alarmed = False

    def begin_request(self):
        self.tap.reset()

    def observe(self, expected_steps):
        started = time.monotonic()
        torch = self.torch
        with torch.no_grad():
            endpoints = (self.tap.first, self.tap.last)
            h = (0,) if self.record['horizon'] == 'first' else (1,) if self.record['horizon'] == 'last' else (0, 1)
            k = (0,) if self.record['denoise'] == 'first' else (1,) if self.record['denoise'] == 'last' else (0, 1)
            feature = torch.cat([endpoints[step][pos] for step in k for pos in h])
            score = self.model.projector(feature.unsqueeze(0)).flatten()[0]
            self.total.add_(score)
            probability = float(score.item())
            cumulative = float(self.total.item())
        actual = int(self.tap.calls.item())
        if actual != expected_steps:
            raise RuntimeError(f'SAFE observed {actual} expert forwards, expected {expected_steps}')
        threshold = self.band[self.count] if self.count < len(self.band) else None
        exceeded = threshold is not None and cumulative >= threshold
        new_alarm = exceeded and not self.alarmed
        self.alarmed = self.alarmed or exceeded
        self.count += 1
        return dict(policy='safe_cumulative_one_full_v1', request_index=self.count - 1,
            probability=probability, cumulative_score=cumulative, threshold=threshold,
            in_calibrated_window=threshold is not None, exceeded=exceeded,
            new_alarm=new_alarm, alpha=self.alpha, feature_horizon=self.record['horizon'],
            feature_denoise=self.record['denoise'], expert_forwards=actual,
            monitor_seconds=time.monotonic() - started)

    def close(self):
        self.tap.close()


def cap_refresh(decision, pending, *, now, step, current_profile):
    """Bring the next request forward; keep any earlier existing constraint."""
    d = dict(decision)
    ticks = int(current_profile != 'baseline')
    latest = now + (pending['request_step'] - step - ticks) * d['estimated_control_period_seconds']
    cap = dict(scope='safe_failure_probe', request_step=pending['request_step'],
        recovery_ticks=ticks, latest_start=latest, alarm=pending)
    if latest < d.get('first_refresh_cap', {}).get('latest_start', float('inf')):
        d['first_refresh_cap'] = cap
    if latest < d['latest_start']:
        d.update(latest_start=latest, start_now=latest <= now,
            wait_seconds=max(0., latest - now), start_constraint='first_refresh')
    d['safe_refresh'] = dict(pending)
    return d
