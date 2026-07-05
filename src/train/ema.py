import math
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average cua trong so model (kieu Ultralytics/YOLO).
    Model EMA thuong on dinh hon va cho ket qua eval/inference tot hon model
    "song" dang duoc optimizer cap nhat truc tiep, dac biet quan trong khi
    pretrain lau (nhieu step) truoc khi chuyen sang finetune voi head moi.

    Cong thuc: ema = decay * ema + (1 - decay) * model
    decay duoc "warm up" dan theo so update (giong YOLOv5/v8) de EMA bam sat
    model that nhanh o nhung buoc dau, roi on dinh dan ve gan decay toi da.
    """

    def __init__(self, model, decay=0.9998, warmup_updates=2000):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_updates = warmup_updates
        self.updates = 0

    def _current_decay(self):
        # decay tang dan tu ~0 den self.decay theo so update (tranh EMA "dong bang" luc dau)
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