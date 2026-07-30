import torch
import torch.nn as nn
from src.model import NMSFreeDetector
from head_face_landmark_v3 import DetectHeadFaceLmk
from train_config_face_lmk import TrainConfig
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

class FaceLmkDetector(nn.Module):

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self._trunk_frozen = False
        scaffold = NMSFreeDetector(
            nc=cfg.face.nc,
            reg_max=cfg.face.reg_max,
            backbone_w=cfg.trunk_backbone_w,
            backbone_n=cfg.trunk_backbone_n,
            neck_n=cfg.trunk_neck_n,
            strides=cfg.face.strides,
        )
        self.backbone = scaffold.backbone
        self.neck = scaffold.neck
        if hasattr(scaffold, 'head'):
            del scaffold.head
        del scaffold
        self.head = DetectHeadFaceLmk(chs=cfg.trunk_feat_channels, cfg=cfg.face)

    def load_trunk(self, path: str, map_location='cpu', strict: bool=True):
        sd = torch.load(path, map_location=map_location)
        (all_missing, all_unexpected) = ([], [])
        for (name, module) in (('backbone', self.backbone), ('neck', self.neck)):
            if name not in sd:
                raise KeyError(f"Checkpoint '{path}' không có khoá '{name}'. Nếu save_trunk() lưu theo format khác (vd state_dict phẳng có prefix), sửa lại load_trunk() trong model_face_lmk.py cho khớp.")
            (missing, unexpected) = module.load_state_dict(sd[name], strict=strict)
            all_missing += [f'{name}.{k}' for k in missing]
            all_unexpected += [f'{name}.{k}' for k in unexpected]
        if all_missing or all_unexpected:
            print(f'[FaceLmkDetector] load_trunk: missing={all_missing}, unexpected={all_unexpected}')
        return (all_missing, all_unexpected)

    def freeze_trunk(self, freeze: bool=True):
        self._trunk_frozen = freeze
        for p in list(self.backbone.parameters()) + list(self.neck.parameters()):
            p.requires_grad = not freeze
        for m in list(self.backbone.modules()) + list(self.neck.modules()):
            if isinstance(m, _BN_TYPES):
                m.eval() if freeze else m.train()

    def train(self, mode: bool=True):
        super().train(mode)
        if mode and self._trunk_frozen:
            for m in list(self.backbone.modules()) + list(self.neck.modules()):
                if isinstance(m, _BN_TYPES):
                    m.eval()
        return self

    def forward(self, images: torch.Tensor):
        feats = self.neck(*self.backbone(images))
        return self.head(feats)
