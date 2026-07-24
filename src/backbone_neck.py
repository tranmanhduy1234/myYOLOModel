import torch
import torch.nn as nn
from src.blocks import Conv, C2f, CIB, C2fCIB, SPPF, C2fPSA, SCDown

class Backbone(nn.Module):
    def __init__(self, w=(64, 128, 256, 512, 1024), n=(3, 6, 6, 3)):
        super().__init__()
        c0, c1, c2, c3, c4 = w
        n0, n1, n2, n3 = n

        self.stem = Conv(3, c0, 3, 2)                       
        self.stage1 = nn.Sequential(
            Conv(c0, c1, 3, 2),                             
            C2f(c1, c1, n=n0, shortcut=True),                
        )
        self.stage2 = nn.Sequential(
            Conv(c1, c2, 3, 2),                             
            C2f(c2, c2, n=n1, shortcut=True),                
        )
        self.stage3 = nn.Sequential(
            SCDown(c2, c3, 3, 2),                        
            C2fCIB(c3, c3, n=n2, shortcut=True),          
        )
        self.stage4 = nn.Sequential(
            SCDown(c3, c4, 3, 2),                        
            C2fCIB(c4, c4, n=n3, shortcut=True),           
            SPPF(c4, c4, k=5),                            
            C2fPSA(c4, c4, n=2, e=0.5),                   
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5

class PAFPN(nn.Module):
    def __init__(self, chs=(256, 512, 1024), n=3):
        super().__init__()
        c3, c4, c5 = chs
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down
        self.c2f_p4 = C2fCIB(c5 + c4, c4, n=n, shortcut=True)  
        self.c2f_p3 = C2f(c4 + c3, c3, n=n, shortcut=False)     

        # Bottom-up
        self.down3 = Conv(c3, c3, 3, 2)                          
        self.c2f_n4 = C2fCIB(c3 + c4, c4, n=n, shortcut=True)   

        self.down4 = SCDown(c4, c4, 3, 2)                       
        self.c2f_n5 = C2fCIB(c4 + c5, c5, n=n, shortcut=True)   

    def forward(self, p3, p4, p5):
        # Top-down
        x = self.up(p5)
        x = torch.cat([x, p4], 1)
        p4_td = self.c2f_p4(x)               

        x = self.up(p4_td)
        x = torch.cat([x, p3], 1)
        p3_out = self.c2f_p3(x)             

        # Bottom-up
        x = self.down3(p3_out)
        x = torch.cat([x, p4_td], 1)
        p4_out = self.c2f_n4(x)            

        x = self.down4(p4_out)
        x = torch.cat([x, p5], 1)
        p5_out = self.c2f_n5(x)          

        return p3_out, p4_out, p5_out