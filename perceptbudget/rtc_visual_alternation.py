"""Fresh full/wrist perception alternation; no stale KV, one frozen expert."""
import hashlib
import time
from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter


class FreshViewRTCAdapter(OriginalOpenPiRTCAdapter):
    def __init__(self, model, torch, config):
        super().__init__(model, torch, config)
        self._full_model = model
        self._wrist_model = None
        self._guided_requests = 0

    def attach_wrist(self, model):
        if model._pipe.pipeline.num_views != 1:
            raise ValueError('Require actual single-view context')
        self._wrist_model = model

    def context_mode(self, prompt, step, *, guided):
        return ('wrist' if guided and self.config.get('m2_visual_fast_enabled')
                and self._guided_requests % 2 == 0 else 'full')

    def _activate(self, mode):
        model = self._full_model if mode == 'full' else self._wrist_model
        if model is None:
            raise RuntimeError('Wrist context not prepared')
        self.model, self.pipe, self.pipeline = model, model._pipe, model._pipe.pipeline

    def prepare_wrist_context(self, snapshot, prompt):
        from perceptbudget.device_profile import select_images
        self._activate('wrist')
        try:
            return super()._prepare_context(select_images(snapshot, 'wrist'), prompt, {}, guided=False)
        finally:
            self._activate('full')

    def _prepare_context(self, snapshot, prompt, request, *, guided):
        begin = time.monotonic()
        valid = super()._prepare_context(snapshot, prompt, request, guided=guided)
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
        inputs = select_images(snapshot, 'wrist' if mode == 'wrist' else 'full')
        actions = super()._predict(inputs, prompt, previous_raw, delay, executed, request)
        request['context_refresh'] = dict(mode=mode, current_images_encoded=True,
            current_state_encoded=inputs.get('state') is not None,
            camera_indices=[1] if mode == 'wrist' else [0, 1],
            num_views=self.pipeline.num_views, visual_tokens=self.pipeline.vision_seq_enc,
            source_observation_step=request.get('observation_step', 0),
            source_prediction_id=request.get('prediction_id'),
            image_input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in inputs['images']],
            context_seconds=request.pop('visual_context_seconds'), limitation='Fresh reduced perception, not a validated quality-equivalent mode')
        if guided:
            self._guided_requests += 1
        return actions
