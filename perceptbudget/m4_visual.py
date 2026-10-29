"""Explicit per-request visual selection on the existing dual-geometry RTC."""
from perceptbudget.rtc_visual_alternation import FreshViewRTCAdapter


class M4ViewRTCAdapter(FreshViewRTCAdapter):
    def select_next_mode(self,mode):
        if mode not in ('Full','Fast'):
            raise ValueError('Continue does not submit neural inference')
        self._m4_next_mode='full' if mode=='Full' else self.fast_mode()

    def context_mode(self,prompt,step,*,guided):
        # Bootstrap remains complete; planning and prediction read the same mode.
        if not guided:return 'full'
        if not hasattr(self,'_m4_next_mode'):
            raise RuntimeError('M4 requires explicit mode before preparing request')
        return self._m4_next_mode
