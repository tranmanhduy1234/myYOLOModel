import torch
import torch.nn as nn
from src.backbone_neck import Backbone, PAFPN
from src.head import DetectHead


class NMSFreeDetector(nn.Module):
    """
    Backbone + Neck + Head.

    QUAN TRONG (giai doan pretrain -> sau nay doi HEAD):
    Model duoc coi la 2 phan tach biet ve mat checkpoint:
      - "trunk"  = backbone + neck  (phan muon giu lai / truyen sang task moi)
      - "head"   = DetectHead       (phan se bi thay khi doi muc tieu detect,
                                      vi du: doi nc, doi reg_max, doi kieu output...)
    Cac ham save_trunk / load_trunk / replace_head / freeze_trunk duoc them
    de phuc vu vong doi: pretrain (hoc trunk tot) -> freeze/finetune voi head moi.
    """

    def __init__(self, nc=80, reg_max=16,
                 backbone_w=(38, 76, 152, 304, 608),
                 backbone_n=(1, 3, 3, 1),
                 neck_n=1,
                 strides=(8, 16, 32)):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.backbone_w = backbone_w
        self.backbone_n = backbone_n
        self.neck_n = neck_n
        self.strides = strides

        self.backbone = Backbone(w=backbone_w, n=backbone_n)
        c3, c4, c5 = backbone_w[2], backbone_w[3], backbone_w[4]
        self.neck_chs = (c3, c4, c5)
        self.neck = PAFPN(chs=(c3, c4, c5), n=neck_n)
        self.head = DetectHead(chs=(c3, c4, c5), nc=nc, reg_max=reg_max, strides=strides)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])

    # ------------------------------------------------------------------
    # Tien ich phuc vu doi HEAD sau khi pretrain xong
    # ------------------------------------------------------------------
    def trunk_state_dict(self):
        """State dict CHI chua backbone + neck (khong co head)."""
        return {
            "backbone": self.backbone.state_dict(),
            "neck": self.neck.state_dict(),
            "meta": {
                "backbone_w": self.backbone_w,
                "backbone_n": self.backbone_n,
                "neck_n": self.neck_n,
                "neck_chs": self.neck_chs,
                "strides": self.strides,
            },
        }

    def save_trunk(self, path):
        """Luu rieng backbone+neck (dung sau khi pretrain xong, truoc khi doi head)."""
        torch.save(self.trunk_state_dict(), path)

    def load_trunk(self, path_or_dict, strict=True, map_location="cpu"):
        """Nap backbone+neck tu file/dict da luu boi save_trunk. Head khong bi dong."""
        ckpt = path_or_dict if isinstance(path_or_dict, dict) else torch.load(path_or_dict, map_location=map_location)
        self.backbone.load_state_dict(ckpt["backbone"], strict=strict)
        self.neck.load_state_dict(ckpt["neck"], strict=strict)
        return self

    def replace_head(self, nc=None, reg_max=None, strides=None):
        """
        Thay HEAD moi (vi du doi so class, doi reg_max...) trong khi GIU NGUYEN
        backbone + neck da pretrain. Goi sau khi da load_trunk(...).
        """
        nc = self.nc if nc is None else nc
        reg_max = self.reg_max if reg_max is None else reg_max
        strides = self.strides if strides is None else strides
        self.head = DetectHead(chs=self.neck_chs, nc=nc, reg_max=reg_max, strides=strides)
        self.nc, self.reg_max, self.strides = nc, reg_max, strides
        return self

    def freeze_trunk(self, freeze=True):
        """Dong bang backbone+neck (dung khi finetune chi head moi giai doan dau)."""
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)
        for p in self.neck.parameters():
            p.requires_grad_(not freeze)
        return self


if __name__ == "__main__":
    m = NMSFreeDetector(nc=80).to("cuda")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Tong tham so: {n_params:,} ({n_params/1e6:.2f}M)")
    import time
    x = torch.randn(2, 3, 480, 480).to("cuda")
    start = time.time()
    for i in range(0, 100):
        out = m(x)
    end = time.time()
    print(end - start)
    
    print("o2o cls:", out["o2o"]["cls"].shape)   # (B, A, nc)
    print("o2o box:", out["o2o"]["box"].shape)   # (B, A, 4)
    print("anchors:", out["anchors"].shape)
    print("A (tong so anchor points) =", out["anchors"].shape[0], "= 60*60+30*30+15*15 =", 60*60+30*30+15*15)