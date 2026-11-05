"""Fresh full/wrist perception alternation; no stale KV, one frozen expert."""
import hashlib
import time
from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter


def visual_geometry(mode):
    if mode not in ('full','wrist','pool2','pool2_agent','selective416','selective320','selective464'):
        raise ValueError('Unknown visual context mode')
    return dict(num_views=1 if mode=='wrist' else 2,
                visual_tokens=464 if mode=='selective464' else 416 if mode=='selective416' else 320 if mode in ('pool2_agent','selective320') else 128 if mode=='pool2' else 256 if mode=='wrist' else 512,
                vision_pool_factor=2 if mode.startswith('pool2') or mode in ('selective416','selective320','selective464') else 1)


def configure_explicit_inputs(config):
    """Worker-level binding; retain state for416, preserve legacy320 behavior."""
    stateful_fast=(config.get('m3_ordinary_fast_context') and
        (config.get('m3_ordinary_fast_visual_mode')=='wrist' or
         (config.get('m3_fast_alternation') and not config.get('m3_fast_no_state'))))
    config['m2_fast_no_state']=not (config.get('m3_fast_selective416',False) or stateful_fast)
    if config.get('m3_fast_selective416'):
        config.pop('m4_fast_prompt_capacity',None)


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
        self.position_policy=None
        if config.get('m3_ordinary_fast_positions'):
            from perceptbudget.selective_pipeline import align_expert_positions
            stateful=not config.get('m2_fast_no_state',False)
            self._position_fast_prefixes=set()
            self.position_policy=align_expert_positions(self.expert,
                visual_geometry(self.fast_mode())['visual_tokens'],
                config['state_prompt_capacity'] if stateful else config['m4_fast_prompt_capacity'],
                fast_prefixes=self._position_fast_prefixes if stateful else None)
        self._full_model = model
        self._wrist_model = None
        self._guided_requests = 0

    def prepare_step_bank(self, lengths, *, progress=None):
        """Two bounded geometry banks, not one enlarged global limit."""
        lengths = sorted(set(lengths))
        fast_tokens = visual_geometry(self.fast_mode())['visual_tokens']
        if self.fast_mode() in ('selective416','selective464'):
            coverage=self.config['m2_visual_prefix_coverage']
            full=coverage['full'];wrist=coverage['fast']
            if set(full)&set(wrist) or set(full+wrist)!=set(lengths):
                raise ValueError('Selective prefix families overlap or differ from prepared lengths')
        else:
            wrist = [n for n in lengths if type(n) is int and fast_tokens <= n <= fast_tokens+200]
            full = [n for n in lengths if type(n) is int and 512 <= n <= 712]
        if len(wrist)+len(full) != len(lengths) or not wrist or not full or max(len(wrist),len(full)) > 17:
            raise ValueError('Require both exact geometry families, at most17 lengths each')
        if self.config.get('source_prefix_coverage_rule') != 'h50_state8_tokenizer_bounds_v1':
            raise ValueError('Dual context bank requires verified H50 state tokenizer bounds')
        if self.position_policy is not None and not self.config.get('m2_fast_no_state',False):
            self._position_fast_prefixes.clear();self._position_fast_prefixes.update(wrist)
        # Both geometries use the same frozen expert, one request at a time.
        # Share scratch storage just as exact lengths already do within a bank.
        if getattr(self,'_shared_step_bank_pool',None) is None:
            self._shared_step_bank_pool=self.torch.cuda.graph_pool_handle()
        reports, banks = [], []
        cached=getattr(self,'_dual_resident_banks',(None,None))
        self._dual_bank_preparing=True
        try:
            for family,bank in zip((full,wrist),cached):
                self._resident_step_bank=bank
                reports.append(super().prepare_step_bank(family, progress=progress))
                banks.append(self.step_bank)
        finally:
            self._dual_bank_preparing=False
            self._resident_step_bank=None
        self._dual_resident_banks=tuple(banks)
        self.step_bank = DualGeometryBank(*banks)
        return dict(passed=True, backend='dual_geometry_source_order_step_graph',
            lengths=lengths, families=reports,
            prepare_seconds=sum(r['prepare_seconds'] for r in reports),
            limitation='Two independently bounded banks; no runtime graph capture')

    def release_graph(self):
        self.torch.cuda.synchronize()
        for bank in getattr(self,'_dual_resident_banks',()):
            bank.release()
        self._dual_resident_banks=()
        self._shared_step_bank_pool=None
        return super().release_graph()

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
            pooled_view='agent' if mode in ('pool2_agent','selective416') else 'both' if mode in ('pool2','selective320','selective464') else None,
            source_observation_step=request.get('observation_step', 0),
            source_prediction_id=request.get('prediction_id'),
            image_input_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in inputs['images']],
            context_seconds=request.pop('visual_context_seconds'), limitation='Fresh reduced perception, not a validated quality-equivalent mode')
        if mode=='selective416':
            request['context_refresh'].update(compression='redundancy_selected_32_of64_blocks',external_tokens=160,wrist_tokens=256)
        if mode=='selective320':
            request['context_refresh'].update(compression='redundancy_selected_32_of64_blocks_per_camera',external_tokens=160,wrist_tokens=160)
        if mode=='selective464':
            request['context_refresh'].update(compression='redundancy_selected_8_of64_blocks_per_camera',external_tokens=232,wrist_tokens=232)
        if self.config.get('m3_ordinary_fast_positions'):
            offset=self.position_policy['action_position_offset'] if mode!='full' else 0
            aligned=getattr(self.pipeline,'_pb_original_visual_positions',False)
            request['context_refresh'].update(position_coordinates=(
                'original_camera_and_text_coordinates_v1' if aligned else 'contiguous_v1'),
                action_position_offset=offset,action_position_start=int(self.pipe.current_prompt_len)+self.pipeline.vision_seq_enc+offset)
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
