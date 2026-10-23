"""Fresh full/wrist perception alternation; no stale KV, one frozen expert."""
import hashlib
import time
from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter


def visual_geometry(mode):
    if mode not in ('full','wrist','pool2','pool2_agent'):
        raise ValueError('Unknown visual context mode')
    return dict(num_views=1 if mode=='wrist' else 2,
                visual_tokens=320 if mode=='pool2_agent' else 128 if mode=='pool2' else 256 if mode=='wrist' else 512,
                vision_pool_factor=2 if mode.startswith('pool2') else 1)


def scheduled_visual_mode(prediction_id, fast_mode, enabled, full_per_fast=1):
    if type(full_per_fast) is not int or full_per_fast not in (1,2):
        raise ValueError('Use one or two Full requests per Fast')
    return fast_mode if enabled and prediction_id>0 and prediction_id%(full_per_fast+1)==full_per_fast else 'full'


def episode_geometry(fast_mode, alternating):
    modes=('full',fast_mode) if alternating else ('full',)
    geometry={mode:visual_geometry(mode) for mode in modes}
    return dict(base_geometry=visual_geometry('full'),geometry_by_mode=geometry,
        visual_tokens_by_mode={mode:g['visual_tokens'] for mode,g in geometry.items()},
        visual_tokens=None if alternating else 512,
        vision_pool_factor=None if alternating else 1,
        geometry_scope='Mixed geometry; use per-request context_refresh' if alternating else 'Full only')


class FreshViewRTCAdapter(OriginalOpenPiRTCAdapter):
    def context_inputs(self, snapshot, mode):
        from perceptbudget.device_profile import select_images
        inputs = select_images(snapshot, 'wrist' if mode=='wrist' else 'full')
        if mode!='full' and self.config.get('m2_fast_no_state',False):
            inputs['state'] = None
        return inputs

    def fast_mode(self):
        return getattr(self,'config',{}).get('m2_visual_fast_mode','wrist')

    def __init__(self, model, torch, config):
        super().__init__(model, torch, config)
        self._full_model = model
        self._wrist_model = None
        self._guided_requests = 0

    def prepare_step_bank(self, lengths, *, progress=None):
        """Two bounded geometry banks, not one enlarged global limit."""
        lengths = sorted(set(lengths))
        fast_tokens = visual_geometry(self.fast_mode())['visual_tokens']
        wrist = [n for n in lengths if type(n) is int and fast_tokens <= n <= fast_tokens+200]
        full = [n for n in lengths if type(n) is int and 512 <= n <= 712]
        if len(wrist)+len(full) != len(lengths) or not wrist or not full or max(len(wrist),len(full)) > 17:
            raise ValueError('Require both exact geometry families, at most17 lengths each')
        if self.config.get('source_prefix_coverage_rule') != 'h50_state8_tokenizer_bounds_v1':
            raise ValueError('Dual context bank requires verified H50 state tokenizer bounds')
        reports, banks = [], []
        for family in (full, wrist):
            reports.append(super().prepare_step_bank(family, progress=progress))
            banks.append(self.step_bank)
        self.step_bank = DualGeometryBank(*banks)
        return dict(passed=True, backend='dual_geometry_source_order_step_graph',
            lengths=lengths, families=reports,
            prepare_seconds=sum(r['prepare_seconds'] for r in reports),
            limitation='Two independently bounded banks; no runtime graph capture')

    def attach_wrist(self, model):
        geometry=visual_geometry(self.fast_mode())
        pl=model._pipe.pipeline
        if (pl.num_views!=geometry['num_views'] or pl.vision_seq_enc!=geometry['visual_tokens']
                or getattr(pl,'vision_pool_factor',1)!=geometry['vision_pool_factor']):
            raise ValueError('Fast context geometry mismatch')
        self._wrist_model = model

    def context_mode(self, prompt, step, *, guided):
        return scheduled_visual_mode(self._guided_requests+1 if guided else 0,self.fast_mode(),
            self.config.get('m2_visual_fast_enabled'),self.config.get('m2_full_per_fast',1))

    def _activate(self, mode):
        if mode not in ('full',self.fast_mode()):
            raise ValueError('Mode does not belong to this visual adapter')
        model = self._full_model if mode == 'full' else self._wrist_model
        if model is None:
            raise RuntimeError('Wrist context not prepared')
        self.model, self.pipe, self.pipeline = model, model._pipe, model._pipe.pipeline
        geometry=visual_geometry(mode)
        if (self.pipeline.num_views!=geometry['num_views'] or self.pipeline.vision_seq_enc!=geometry['visual_tokens']
                or getattr(self.pipeline,'vision_pool_factor',1)!=geometry['vision_pool_factor']):
            raise RuntimeError('Active visual pipeline geometry changed')

    def prepare_wrist_context(self, snapshot, prompt):
        from perceptbudget.device_profile import select_images
        self._activate(self.fast_mode())
        try:
            return super()._prepare_context(self.context_inputs(snapshot,self.fast_mode()), prompt, {}, guided=False)
        finally:
            self._activate('full')

    def _prepare_context(self, snapshot, prompt, request, *, guided):
        begin = time.monotonic()
        valid = super()._prepare_context(snapshot, prompt, request, guided=guided)
        if self.step_bank is not None and self.step_bank.select(valid)[1] is None:
            raise RuntimeError(f'Unprepared visual prefix {valid}; stopped before expert, no slow fallback')
        if valid!=self.pipeline.vision_seq_enc+self.pipe.current_prompt_len:
            raise RuntimeError('Visual prefix offset differs from actual tokenized input')
        request['visual_context_seconds'] = time.monotonic()-begin
        return valid

    def _predict(self, snapshot, prompt, previous_raw, delay, executed, request):
        from perceptbudget.device_profile import select_images
        guided = previous_raw is not None
        if not guided:
            self._guided_requests = 0
        mode = self.context_mode(prompt, request.get('observation_step', 0), guided=guided)
        if request.get('planned_context_mode', mode) != mode:
            raise RuntimeError('Planned visual mode differs from executed mode')
        self._activate(mode)
        inputs = self.context_inputs(snapshot, mode)
        actions = super()._predict(inputs, prompt, previous_raw, delay, executed, request)
        request['context_refresh'] = dict(mode=mode, current_images_encoded=True,
            current_state_encoded=inputs.get('state') is not None,
            camera_indices=[1] if mode == 'wrist' else [0, 1],
            num_views=self.pipeline.num_views, visual_tokens=self.pipeline.vision_seq_enc,
            vision_pool_factor=getattr(self.pipeline,'vision_pool_factor',1),
            pooled_view='agent' if mode=='pool2_agent' else 'both' if mode=='pool2' else None,
            source_observation_step=request.get('observation_step', 0),
            source_prediction_id=request.get('prediction_id'),
            image_input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in inputs['images']],
            context_seconds=request.pop('visual_context_seconds'), limitation='Fresh reduced perception, not a validated quality-equivalent mode')
        if guided:
            self._guided_requests += 1
        return actions


class DualGeometryBank:
    """Sequential dispatch; each child retains its own bounded graph storage."""
    def __init__(self, full, wrist):
        if set(full.entries) & set(wrist.entries):
            raise ValueError('Overlapping geometry prefixes')
        self.full, self.wrist = full, wrist
        self.fallback = full.fallback

    def select(self, valid):
        return self.wrist.select(valid) if valid in self.wrist.entries else self.full.select(valid)
