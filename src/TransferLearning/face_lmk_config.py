"""
face_lmk_config.py
====================
Config DUY NHAT dùng chung cho cả 3 file: head_face_landmark.py,
loss_face_landmark.py, face_landmark_dataset.py.

LÝ DO TỒN TẠI FILE NÀY (fix bug của bản trước)
------------------------------------------------------------------------
Ở bản trước, `num_landmarks` và `lmk_margin` được khai báo ĐỘC LẬP ở cả
`DetectHeadFaceLmk.__init__` VÀ `FaceLandmarkDetectionLoss.__init__`
(default riêng, không liên quan gì nhau). Hệ quả:

  1. num_landmarks default=5 ở head/loss nhưng dataset thực tế (MediaPipe
     FaceLandmarker) có thể là 478 điểm -> nếu quên truyền tay, loss sẽ
     crash ngay bước đầu training (shape mismatch khi gán landmarks vào
     gt_landmarks trong preprocess_targets).
  2. lmk_margin nếu sửa ở 1 nơi mà quên sửa nơi kia -> encode lúc train
     (loss) và decode lúc inference (head) LỆCH NHAU ÂM THẦM (không lỗi
     runtime, chỉ sai vị trí landmark một cách hệ thống, rất khó debug).

FaceLmkConfig gom TẤT CẢ tham số cần dùng CHUNG giữa head/loss/dataset
vào 1 chỗ duy nhất. Quy trình dùng đúng:

    from face_lmk_config import FaceLmkConfig
    from face_landmark_dataset_v3 import FaceLandmarkDataset
    from head_face_landmark_v3 import DetectHeadFaceLmk
    from loss_face_landmark_v3 import FaceLandmarkDetectionLoss

    cfg = FaceLmkConfig(nc=1, reg_max=16)          # num_landmarks CHƯA đặt
    dataset = FaceLandmarkDataset(root_dir, image_size=224)
    cfg.sync_num_landmarks(dataset.num_landmarks)  # <-- BẮT BUỘC, xem dưới

    head = DetectHeadFaceLmk(chs=(128, 256, 512), cfg=cfg)
    loss_fn = FaceLandmarkDetectionLoss(cfg=cfg)

Cả head và loss giờ nhận `cfg` thay vì nhận `num_landmarks`/`lmk_margin`
riêng lẻ -> không còn chỗ nào để 2 giá trị này lệch nhau được nữa (chỉ
có 1 nguồn duy nhất để sửa).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FaceLmkConfig:
    # ---- kiến trúc / dữ liệu dùng CHUNG giữa head <-> loss <-> dataset ----
    nc: int = 1                                    # số class (mặc định 1: "face")
    reg_max: int = 16
    strides: Tuple[int, ...] = (8, 16, 32)
    num_landmarks: Optional[int] = None            # None = "chưa đặt" -> BẮT BUỘC sync_num_landmarks()
    lmk_margin: float = 0.15                        # dùng CHUNG cho decode (head) và encode (loss)

    # ---- loss gains ----
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    lmk_gain: float = 1.0
    geo_gain: float = 0.0                            # >0 để bật Geometric Consistency Loss

    # ---- TaskAlignedAssigner ----
    topk_o2m: int = 10
    topk_o2o: int = 1
    alpha: float = 0.5
    beta: float = 6.0

    # ---- trọng số 2 nhánh o2m / o2o ----
    o2m_weight: float = 1.0
    o2o_weight: float = 1.0

    # ---- landmark loss ----
    lmk_loss_type: str = "smooth_l1"                # "smooth_l1" | "wing"
    geo_constraints: List[tuple] = field(default_factory=list)
    geo_margin: float = 0.02

    def sync_num_landmarks(self, dataset_num_landmarks: int) -> None:
        """
        Gọi 1 LẦN DUY NHẤT, ngay sau khi tạo dataset, TRƯỚC KHI tạo head/loss.

        - Nếu cfg.num_landmarks còn là None (chưa đặt) -> lấy luôn giá trị
          từ dataset. Đây là trường hợp phổ biến nhất: không cần biết
          trước data có bao nhiêu điểm, để dataset tự dò rồi đồng bộ
          ngược lại config.
        - Nếu cfg.num_landmarks ĐÃ được đặt tay (ví dụ bạn có chủ đích chỉ
          lấy subset 5/68 điểm từ 478 điểm gốc) MÀ khác
          dataset_num_landmarks -> RAISE lỗi ngay, không im lặng, vì đây
          rất có thể là nhầm lẫn (quên sync) chứ không phải chủ ý.
        """
        if self.num_landmarks is None:
            self.num_landmarks = dataset_num_landmarks
            return
        if self.num_landmarks != dataset_num_landmarks:
            raise ValueError(
                f"FaceLmkConfig.num_landmarks ({self.num_landmarks}) khác với "
                f"số landmark thực tế trong dataset ({dataset_num_landmarks}). "
                "Nếu bạn CÓ CHỦ Ý dùng subset landmark: tự slice "
                "landmarks/labels trước khi đưa vào head/loss, rồi tạo "
                "FaceLmkConfig(num_landmarks=<số đã slice>) và KHÔNG gọi "
                "sync_num_landmarks() nữa (hoặc gọi với đúng số đã slice). "
                "Nếu KHÔNG chủ ý, đây là bug - hãy để trống 'num_landmarks=' "
                "khi tạo FaceLmkConfig để nó tự lấy đúng giá trị từ dataset."
            )

    def require_num_landmarks(self) -> int:
        if self.num_landmarks is None:
            raise ValueError(
                "FaceLmkConfig.num_landmarks chưa được đặt. Gọi "
                "cfg.sync_num_landmarks(dataset.num_landmarks) trước khi "
                "tạo DetectHeadFaceLmk / FaceLandmarkDetectionLoss."
            )
        return self.num_landmarks
