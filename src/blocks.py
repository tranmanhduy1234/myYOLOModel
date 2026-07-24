import torch
import torch.nn as nn
import math

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class CIB(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # kênh trung gian (hidden channels)
        self.block = nn.Sequential(
            Conv(c1, c1, 3, 1, g=c1),
            Conv(c1, 2 * c_, 1, 1),
            Conv(2 * c_, 2 * c_, 3, 1, g=2 * c_),
            Conv(2 * c_, c2, 1, 1),
            Conv(c2, c2, 3, 1, g=c2)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.block(x) if self.add else self.block(x)

class C2fCIB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__(c1, c2, n, shortcut, e)
        self.m = nn.ModuleList(
            CIB(self.c, self.c, shortcut=shortcut, e=1.0)
            for _ in range(n)
        )

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = x.view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        # x: (B, 4*c1, A) -> (B, 4, A)
        b, c, a = x.shape
        x = x.view(b, 4, self.c1, a).transpose(2, 1)  # (B, c1, 4, A)
        x = x.softmax(1)
        return self.conv(x).view(b, 4, a)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratio=2.0,
        layer_scale=1e-2,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV Projection
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)

        # Output Projection
        self.proj = Conv(dim, dim, 1, 1, act=False)

        # Positional Encoding cục bộ (chuẩn YOLOv10 PSA): depthwise conv trên value
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

        # Feed Forward Network
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            Conv(dim, hidden_dim, 1, 1),
            Conv(hidden_dim, dim, 1, 1, act=False),
        )

        # LayerScale (optional)
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(
            B, 3, self.num_heads, self.head_dim, N
        )
        q, k, v = qkv.unbind(1)

        q = q.transpose(-2, -1)        # (B,h,N,d)
        k = k                          # (B,h,d,N)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        v_spatial = v.reshape(B, C, H, W)   # dùng cho positional encoding
        v_t = v.transpose(-2, -1)           # (B,h,N,d)

        out = attn @ v_t                    # (B,h,N,d)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        out = out + self.pe(v_spatial)      # cộng positional encoding cục bộ
        out = self.proj(out)

        x = x + self.gamma1.view(1, -1, 1, 1) * out
        x = x + self.gamma2.view(1, -1, 1, 1) * self.ffn(x)
        return x

class C2fPSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, "C2fPSA (C2PSA) yêu cầu c1 == c2 theo chuẩn YOLOv10"

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1, 1)
        self.m = nn.Sequential(*[Attention(self.c) for _ in range(n)])

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)                       
        self.cv2 = Conv(c2, c2, k, s, p=p, g=c2, d=d, act=False) 

    def forward(self, x):
        return self.cv2(self.cv1(x))