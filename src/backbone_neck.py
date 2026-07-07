import torch
import torch.nn as nn
from src.blocks import Conv, C2f, SPPF, C2fPSA

class Backbone(nn.Module):
    """
    CSPDarknet-lite. Input (3, 480, 480).
    Stride tổng: 8 (P3), 16 (P4), 32 (P5)
      480 -> /2 -> 240 (stem)
           -> /2 -> 120 (stage1)
           -> /2 ->  60 (stage2) = P3
           -> /2 ->  30 (stage3) = P4
           -> /2 ->  15 (stage4) = P5
    """

    def __init__(self, w=(32, 64, 128, 256, 512), n=(1, 2, 2, 1)):
        super().__init__()
        c0, c1, c2, c3, c4 = w
        self.stem = Conv(3, c0, 3, 2)                       # 480 -> 240
        self.stage1 = nn.Sequential(
            Conv(c0, c1, 3, 2),                              # 240 -> 120
            C2f(c1, c1, n=n[0], shortcut=True),
        )
        self.stage2 = nn.Sequential(
            Conv(c1, c2, 3, 2),                              # 120 -> 60 (P3)
            C2f(c2, c2, n=n[1], shortcut=True),
        )
        self.stage3 = nn.Sequential(
            Conv(c2, c3, 3, 2),                              # 60 -> 30 (P4)
            C2f(c3, c3, n=n[2], shortcut=True),
        )
        self.stage4 = nn.Sequential(
            Conv(c3, c4, 3, 2),                              # 30 -> 15 (P5)
            C2fPSA(c4, c4, n=n[3], e=0.5),
            # C2f(c4, c4, n=n[3], shortcut=True),
            SPPF(c4, c4, k=5),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5

class PAFPN(nn.Module):
    """Path Aggregation FPN: top-down rồi bottom-up, giống YOLOv8/v10 neck."""

    def __init__(self, chs=(128, 256, 512), n=1):
        super().__init__()
        c3, c4, c5 = chs
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # top-down
        self.reduce5 = Conv(c5, c4, 1, 1)
        self.c2f_p4 = C2f(c4 + c4, c4, n=n, shortcut=False)
        self.reduce4 = Conv(c4, c3, 1, 1)
        self.c2f_p3 = C2f(c3 + c3, c3, n=n, shortcut=False)   # output P3 final

        # bottom-up
        self.down3 = Conv(c3, c3, 3, 2)
        self.c2f_n4 = C2f(c3 + c3, c4, n=n, shortcut=False)   # output P4 final
        self.down4 = Conv(c4, c4, 3, 2)
        self.c2f_n5 = C2f(c4 + c4, c5, n=n, shortcut=False)   # output P5 final

    def forward(self, p3, p4, p5):
        p5_red = self.reduce5(p5)
        x = self.up(p5_red)
        x = torch.cat([x, p4], 1)
        p4_td = self.c2f_p4(x)

        p4_red = self.reduce4(p4_td)
        x = self.up(p4_red)
        x = torch.cat([x, p3], 1)
        p3_out = self.c2f_p3(x)

        x = self.down3(p3_out)
        x = torch.cat([x, p4_red], 1)
        p4_out = self.c2f_n4(x)

        x = self.down4(p4_out)
        x = torch.cat([x, p5_red], 1)
        p5_out = self.c2f_n5(x)

        return p3_out, p4_out, p5_out