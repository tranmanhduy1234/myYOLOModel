import torch
import torch.nn as nn
from src.backbone_neck import Backbone, PAFPN
from src.head import DetectHead
from src.config import TrainConfig

class NMSFreeDetector(nn.Module):
    def __init__(self, nc=TrainConfig.nc, reg_max=TrainConfig.reg_max,
                  backbone_w=TrainConfig.backbone_w,
                 backbone_n=TrainConfig.backbone_n,
                 neck_n=TrainConfig.neck_n,
                 strides=TrainConfig.strides):
        super().__init__()
        
        # Setting parameters
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
        results = self.head([p3, p4, p5])
        return results

    def trunk_state_dict(self):
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
        torch.save(self.trunk_state_dict(), path)

    def load_trunk(self, path_or_dict, strict=True, map_location="cuda"):
        ckpt = path_or_dict if isinstance(path_or_dict, dict) else torch.load(path_or_dict, map_location=map_location)
        self.backbone.load_state_dict(ckpt["backbone"], strict=strict)
        self.neck.load_state_dict(ckpt["neck"], strict=strict)
        return self

    def replace_head(self, nc=None, reg_max=None, strides=None):
        nc = self.nc if nc is None else nc
        reg_max = self.reg_max if reg_max is None else reg_max
        strides = self.strides if strides is None else strides

        # Ke thua device & dtype hien tai cua model (vi du sau khi da .to('cuda') hoac .half()),
        # neu khong head moi se luon mac dinh nam tren CPU/float32 -> gay loi mismatch device/dtype.
        ref_param = next(self.backbone.parameters())
        device, dtype = ref_param.device, ref_param.dtype

        self.head = DetectHead(
            chs=self.neck_chs, nc=nc, reg_max=reg_max, strides=strides
        ).to(device=device, dtype=dtype)
        self.nc, self.reg_max, self.strides = nc, reg_max, strides
        return self

    def freeze_trunk(self, freeze=True):
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)
        for p in self.neck.parameters():
            p.requires_grad_(not freeze)
        return self

if __name__ == "__main__":
    m = NMSFreeDetector().to("cuda").eval()

    n_params = sum(p.numel() for p in m.parameters())
    print(f"Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    import time
    x = torch.randn(5, 3, 640, 640).to("cuda")
    # Benchmark tốc độ inference
    with torch.inference_mode():
        start = time.time()
        for _ in range(1):
            out = m(x)
        end = time.time()
    print("Inference time:", end - start)
    # Kiểm tra kích thước đầu ra
    print("o2o cls:", out["o2o"]["cls"].shape)
    print("o2o box:", out["o2o"]["box"].shape)
    print("anchors:", out["anchors"].shape)
    
    # # 1. Chuyển mô hình sang chế độ eval và ÉP SANG FP16 (HALF PRECISION)
    # m.eval().half()
    # # 2. Tạo dữ liệu mẫu cũng phải ở định dạng dạng .half() mới khớp mạng
    # dummy_input = torch.randn(5, 3, 640, 640).to("cuda").half()   # <-- đổi 480 -> 640
    # onnx_filename = "yolov10_custom_fp16.onnx"
    # print("Đang xuất mô hình sang định dạng ONNX FP16...")
    # torch.onnx.export(
    #     m,
    #     dummy_input,
    #     onnx_filename,
    #     verbose=False,
    #     opset_version=18,
    #     input_names=["images"],
    #     output_names=["cls", "box", "reg_raw", "anchors", "strides"],
    #     do_constant_folding=True
    # )
    # print(f"Xuất ONNX FP16 thành công! File mới: {onnx_filename}")