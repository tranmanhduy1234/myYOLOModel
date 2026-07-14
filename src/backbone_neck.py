import torch
import torch.nn as nn
from src.blocks import Conv, C2f, CIB, C2fCIB, SPPF, C2fPSA, SCDown

class Backbone(nn.Module):
    def __init__(self, w=(48, 96, 192, 384, 512), n=(2, 4, 4, 2)):
        super().__init__()
        c0, c1, c2, c3, c4 = w
        self.stem = Conv(3, c0, 3, 2)                     
        self.stage1 = nn.Sequential(
            SCDown(c0, c1, 3, 2),                            
            C2f(c1, c1, n=n[0], shortcut=True),
        )
        self.stage2 = nn.Sequential(
            SCDown(c1, c2, 3, 2),                        
            C2f(c2, c2, n=n[1], shortcut=True),
        )
        self.stage3 = nn.Sequential(
            SCDown(c2, c3, 3, 2),                   
            CIB(c3, c3, shortcut=True),                     
            C2fCIB(c3, c3, n=n[2], shortcut=True),           
        )
        self.stage4 = nn.Sequential(
            SCDown(c3, c4, 3, 2),                              
            C2fPSA(c4, c4, n=n[3], e=0.5),
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
    def __init__(self, chs=(192, 384, 512), n=2):
        super().__init__()
        c3,c4,c5 = chs
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce5 = Conv(c5,c4,1,1)
        
        self.c2f_p4 = C2fCIB(c4+c4, c4, n=n,shortcut=False)
        self.reduce4 = Conv(c4,c3,1,1)

        self.c2f_p3 = C2fCIB(c3+c3,c3,n=n,shortcut=False)
        self.down3 = SCDown(c3,c3,3,2)

        self.c2f_n4 = C2fCIB(c3+c3,c4,n=n,shortcut=False)
        self.down4 = SCDown(c4,c4,3,2)

        self.c2f_n5 = C2fCIB(c4+c4,c5,n=n,shortcut=False)

    def forward(self,p3,p4,p5):
        p5_reduce = self.reduce5(p5)
        x = self.up(p5_reduce)
        x = torch.cat([x,p4],1)
        
        p4_td = self.c2f_p4(x)
        p4_reduce = self.reduce4(p4_td)
        x = self.up(p4_reduce)
        x = torch.cat([x,p3],1)
        
        p3_out = self.c2f_p3(x)
        x = self.down3(p3_out)
        x = torch.cat([x,p4_reduce],1)
        
        p4_out = self.c2f_n4(x)
        x = self.down4(p4_out)
        x = torch.cat([x,p5_reduce],1)
        
        p5_out = self.c2f_n5(x)
        return p3_out,p4_out,p5_out