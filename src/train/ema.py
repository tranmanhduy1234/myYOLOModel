import math
from copy import deepcopy
import torch

class ModelEMA:
    def __init__(self, model, decay=0.9998, warmup_updates=2000):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_updates = warmup_updates
        self.updates = 0

    def _current_decay(self):
        return self.decay * (1 - math.exp(-self.updates / max(1, self.warmup_updates)))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self._current_decay()
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)