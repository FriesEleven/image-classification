"""Observe actual non-fused optimizer updates without synchronizing AMP scale."""


class OptimizerStepTracker:
    def __init__(self, optimizer):
        self.did_step = False
        # Fused/custom AMP-aware optimizers can be called even on an overflow.
        # Keep the scale-based fallback for them; the current non-fused AdamW
        # takes GradScaler's ordinary skip path before optimizer.step is called.
        self.supported = not getattr(optimizer, "_step_supports_amp_scaling", False)
        self.handle = optimizer.register_step_post_hook(self._after_step) if self.supported else None

    def _after_step(self, _optimizer, _args, _kwargs):
        self.did_step = True

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        if self.handle is not None:
            self.handle.remove()
