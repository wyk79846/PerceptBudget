"""Isolated M2 Full/recompute alternation; never the deployment default."""
import time
from perceptbudget.rtc_openpi import OriginalOpenPiRTCAdapter


class RecomputeRTCAdapter(OriginalOpenPiRTCAdapter):
    def _prepare_context(self,snapshot,prompt,request,*,guided):
        if not self.config.get('m2_expert_recompute'):
            raise ValueError('Expert recompute requires explicit M2 experiment')
        start=time.monotonic()
        step=request.get('observation_step',0)
        old=getattr(self,'_recompute_context',None)
        enabled=self.config.get('m2_reuse_enabled',False)
        reuse=bool(guided and enabled and old is not None and not old['reused']
                   and prompt==old['prompt'] and 0<=step-old['step']<self.horizon//2)
        if reuse:
            valid=old['valid']
            old['reused']=True
        else:
            valid=super()._prepare_context(snapshot,prompt,request,guided=guided)
            old=dict(valid=valid,step=step,prompt=prompt,reused=False,
                raw_state=list(map(float,snapshot['raw_state'])),
                prediction_id=request.get('prediction_id'),
                observation_monotonic=request.get('observation_monotonic'))
            self._recompute_context=old
        request['kv_recompute']=dict(mode='expert_only' if reuse else 'full',
            source_observation_step=old['step'],source_prediction_id=old['prediction_id'],
            source_observation_monotonic=old['observation_monotonic'],
            source_raw_state=old['raw_state'],age_steps_at_request=step-old['step'],
            nominal_expiry_step=old['step']+self.horizon,
            current_state_encoded=not reuse,current_images_encoded=not reuse,
            context_seconds=time.monotonic()-start,
            limitation='H-step nominal expiry is an upper bound, not a validated safe buffer')
        return valid

    def validate_execution(self,request,step):
        info=request.get('kv_recompute')
        if info is None or step>=info['nominal_expiry_step']:
            raise RuntimeError('Refusing to execute beyond original KV nominal horizon')
