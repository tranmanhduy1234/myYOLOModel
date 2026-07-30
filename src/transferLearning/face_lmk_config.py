from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class FaceLmkConfig:
    nc: int = 1
    reg_max: int = 16
    strides: Tuple[int, ...] = (8, 16, 32)
    num_landmarks: Optional[int] = None
    lmk_margin: float = 0.15
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    lmk_gain: float = 1.0
    geo_gain: float = 0.0
    topk_o2m: int = 10
    topk_o2o: int = 1
    alpha: float = 0.5
    beta: float = 6.0
    o2m_weight: float = 1.0
    o2o_weight: float = 1.0
    lmk_loss_type: str = 'smooth_l1'
    geo_constraints: List[tuple] = field(default_factory=list)
    geo_margin: float = 0.02

    def sync_num_landmarks(self, dataset_num_landmarks: int) -> None:
        if self.num_landmarks is None:
            self.num_landmarks = dataset_num_landmarks
            return
        if self.num_landmarks != dataset_num_landmarks:
            raise ValueError(f"FaceLmkConfig.num_landmarks ({self.num_landmarks}) khác với số landmark thực tế trong dataset ({dataset_num_landmarks}). Nếu bạn CÓ CHỦ Ý dùng subset landmark: tự slice landmarks/labels trước khi đưa vào head/loss, rồi tạo FaceLmkConfig(num_landmarks=<số đã slice>) và KHÔNG gọi sync_num_landmarks() nữa (hoặc gọi với đúng số đã slice). Nếu KHÔNG chủ ý, đây là bug - hãy để trống 'num_landmarks=' khi tạo FaceLmkConfig để nó tự lấy đúng giá trị từ dataset.")

    def require_num_landmarks(self) -> int:
        if self.num_landmarks is None:
            raise ValueError('FaceLmkConfig.num_landmarks chưa được đặt. Gọi cfg.sync_num_landmarks(dataset.num_landmarks) trước khi tạo DetectHeadFaceLmk / FaceLandmarkDetectionLoss.')
        return self.num_landmarks
