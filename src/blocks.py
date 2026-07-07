import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module):
    """Conv2d + BN + SiLU (khối cơ bản kiểu YOLO)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    """Depthwise conv - dùng trong head để giảm tham số."""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=1, act=act)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=min(c1, c2), bias=False)

class Bottleneck(nn.Module):
    """Khối này thực hiện nhiệm vụ trích xuất đặc trưng sâu (deep features) thông qua cơ chế thắt nút cổ chai (bottleneck)"""
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
    """Khối CSP 2 nhánh dạng "fast" (giống Ultralytics YOLOv8/v10)."""
    """Khối này đóng vai trò là trung tâm trích xuất đặc trưng bậc cao và đa quy mô (Multi-scale Feature Fusion) trong phần Backbone và Neck của YOLOv10."""
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

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast, mở rộng receptive field rẻ tiền."""
    """Khối này làm nhiệm vụ Hội tụ đặc trưng đa quy mô toàn cục (Global Multi-scale Feature Fusion)."""
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
    """Distribution Focal Loss decode: chuyển phân phối rời rạc -> giá trị ltrb liên tục."""
    """Khối này làm nhiệm vụ chuyển đổi một phân phối xác suất rời rạc (Discrete Probability Distribution) 
    thành một giá trị hình học liên tục (Continuous Value)."""
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
    """Lớp Self-Attention thu gọn đã vá lỗi chia batch và tràn bộ nhớ."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = Conv(dim, dim, 1, 1, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        # qkv shape ban đầu: (B, 3, num_heads, head_dim, N)
        qkv = self.qkv(x).view(B, 3, self.num_heads, C // self.num_heads, N)
        
        # FIX LỖI 1: Dùng .unbind(1) để rã tensor theo chiều dimension 1 (nơi chứa đúng 3 phần tử Q, K, V)
        q, k, v = qkv.unbind(1) # Mỗi tensor tách ra sẽ có shape chuẩn: (B, num_heads, head_dim, N)

        # Tính toán ma trận tự chú ý không gian
        attn = (q.transpose(-2, -1) @ k) * self.scale # Shape: (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)

        # FIX LỖI 2: Thêm .contiguous() trước khi .view() để làm mượt lại các ô nhớ trên RAM/VRAM
        x_attn = (v @ attn.transpose(-2, -1)).contiguous().view(B, C, H, W)
        
        return self.proj(x_attn)

class C2fPSA(nn.Module):
    """Khối C2f tích hợp Partial Self-Attention đặc trưng của YOLOv10."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut=True, e=1.0) for _ in range(n))
        self.attn = Attention(self.c)

    def forward(self, x):
        # Mẹo toán học: Chẻ đôi số kênh màu làm 2 nhánh độc lập
        a, b = self.cv1(x).chunk(2, 1)
        # Nhánh b được mài giũa qua các khối Bottleneck
        for m in self.m:
            b = m(b)
        # CHÍNH NÓ: Áp dụng Self-Attention ĐƠN NHÁNH để tìm mối quan hệ toàn cục
        b = b + self.attn(b)
        # Ghép lại với nhánh gốc a và nén kênh về c2
        return self.cv2(torch.cat((a, b), 1))