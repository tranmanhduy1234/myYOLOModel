import os
import sys
from typing import Union, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import PIL.Image
import torch
from torchvision.ops import nms

# Thiết lập đường dẫn import cho dự án
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(src_dir, ".."))

for p in [current_dir, src_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from .config_lmk import TrainConfig
    from .model_lmk import FaceLmkDetector
except ImportError:
    from config_lmk import TrainConfig
    from model_lmk import FaceLmkDetector

def letterbox(
    img: np.ndarray,
    target_size: Tuple[int, int] = (480, 480),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Biến đổi ảnh theo phương pháp letterbox: giữ nguyên tỷ lệ khung hình (aspect ratio)
    và thêm padding xung quanh để đưa về kích thước target_size (h, w).

    Args:
        img: Ảnh đầu vào dạng numpy array (H, W, C) ở hệ màu RGB hoặc BGR.
        target_size: Kích thước mong muốn (target_h, target_w).
        color: Màu của phần padding fill (mặc định 114, 114, 114).

    Returns:
        padded_img: Ảnh đã qua letterbox có kích thước (target_h, target_w, C).
        info: Dictionary chứa các tham số phục vụ việc khôi phục tọa độ về ảnh gốc.
    """
    orig_h, orig_w = img.shape[:2]
    target_h, target_w = target_size

    # Tỷ lệ scale giữ nguyên aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)

    # Kích thước unpadded mới sau khi scale
    new_unpad_w = int(round(orig_w * scale))
    new_unpad_h = int(round(orig_h * scale))

    # Kích thước padding cần thêm vào 2 bên và trên dưới
    dw = target_w - new_unpad_w
    dh = target_h - new_unpad_h

    pad_w = dw / 2.0
    pad_h = dh / 2.0

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    # Resize ảnh theo kích thước unpadded mới
    if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
        img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    # Thêm padding viền
    padded_img = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    info = {
        'scale': scale,
        'pad_w': pad_w,
        'pad_h': pad_h,
        'top': top,
        'left': left,
        'new_unpad_w': new_unpad_w,
        'new_unpad_h': new_unpad_h,
        'orig_w': orig_w,
        'orig_h': orig_h,
        'target_w': target_w,
        'target_h': target_h,
    }

    return padded_img, info


class FaceLandmarkInferencer:
    """
    Lớp đóng gói quy trình Suy luận (Inference) cho mô hình Face & Landmark Detection.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        cfg: Optional[TrainConfig] = None,
        device: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        image_size: Optional[int] = None,
        num_landmarks: Optional[int] = None,
        lmk_margin: Optional[float] = None,
        max_det: Optional[int] = None,
    ):
        """
        Khởi tạo module suy luận với các tham số cấu hình cần thiết.

        Args:
            weights_path: Đường dẫn tới file trọng số (.pt) của mô hình (nếu có).
            device: Thiết bị chạy suy luận ('cuda', 'cpu' hoặc None để tự động nhận diện).
            conf_threshold: Ngưỡng độ tin cậy để lọc các detection.
            iou_threshold: Ngưỡng IoU cho NMS loại bỏ bbox trùng lặp.
            image_size: Kích thước ảnh đầu vào; pipeline chỉ chấp nhận 480.
            num_landmarks: Số lượng điểm landmark trên khuôn mặt (mặc định 478).
            lmk_margin: Hệ số margin mở rộng bbox cho decode landmark.
        """
        self.train_cfg = cfg or TrainConfig()
        if image_size is not None and image_size != self.train_cfg.image_size:
            raise ValueError(f'image_size phải đồng nhất ở {self.train_cfg.image_size}.')
        if num_landmarks is not None:
            self.train_cfg.face.num_landmarks = num_landmarks
        if lmk_margin is not None:
            self.train_cfg.face.lmk_margin = lmk_margin
        if self.train_cfg.face.num_landmarks is None:
            # Inference không có dataset để sync; checkpoint có thể điều chỉnh lại K bên dưới.
            self.train_cfg.face.num_landmarks = 478

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.conf_threshold = self.train_cfg.inference_conf_threshold if conf_threshold is None else conf_threshold
        self.iou_threshold = self.train_cfg.inference_iou_threshold if iou_threshold is None else iou_threshold
        self.max_det = self.train_cfg.inference_max_det if max_det is None else max_det
        if not 0 <= self.conf_threshold <= 1 or not 0 <= self.iou_threshold <= 1:
            raise ValueError('conf_threshold và iou_threshold phải nằm trong [0, 1].')
        if self.max_det <= 0:
            raise ValueError('max_det phải > 0.')
        self.image_size = self.train_cfg.image_size
        self.num_landmarks = self.train_cfg.face.require_num_landmarks()
        self.lmk_margin = self.train_cfg.face.lmk_margin

        if weights_path is not None and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số tại '{weights_path}'.")

        # Khởi tạo mô hình
        self.model = FaceLmkDetector(self.train_cfg)

        if weights_path is not None:
            self.load_weights(weights_path)

        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, weights_path: str) -> None:
        """
        Nạp trọng số từ file checkpoint .pt vào mô hình.
        """
        print(f"[FaceLandmarkInferencer] Nạp weights từ: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if 'ema' in checkpoint and checkpoint['ema'] is not None:
                state_dict = checkpoint['ema']
                print("[FaceLandmarkInferencer] Nạp trọng số từ bản sao EMA.")
            elif 'model' in checkpoint and checkpoint['model'] is not None:
                state_dict = checkpoint['model']
                print("[FaceLandmarkInferencer] Nạp trọng số từ checkpoint['model'].")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            raise TypeError('Checkpoint phải là dict hoặc state_dict.')

        if isinstance(state_dict, torch.nn.Module):
            state_dict = state_dict.state_dict()
        if isinstance(state_dict, dict) and 'ema' in state_dict and isinstance(state_dict['ema'], dict):
            state_dict = state_dict['ema']
        if not isinstance(state_dict, dict):
            raise TypeError('Không trích xuất được state_dict hợp lệ từ checkpoint.')

        normalized = {}
        for key, value in state_dict.items():
            clean = key
            while clean.startswith(('module.', 'model.', 'ema.')):
                clean = clean.split('.', 1)[1]
            normalized[clean] = value
        state_dict = normalized

        # Tự động cập nhật num_landmarks từ shape trọng số nếu phát hiện khớp
        for key in ['head.heads.0.lmk_o2o.weight', 'head.heads.0.lmk_o2m.weight']:
            if key in state_dict:
                detected_num_lmk = state_dict[key].shape[0] // 2
                if detected_num_lmk != self.num_landmarks:
                    print(f"[FaceLandmarkInferencer] Điều chỉnh num_landmarks từ {self.num_landmarks} -> {detected_num_lmk} theo weights.")
                    self.num_landmarks = detected_num_lmk
                    self.train_cfg.face.num_landmarks = detected_num_lmk
                    # Tái khởi tạo head với num_landmarks mới
                    self.model = FaceLmkDetector(self.train_cfg)
                break

        model_state = self.model.state_dict()
        model_keys = set(model_state)
        matched_keys = model_keys.intersection(state_dict)
        load_ratio = len(matched_keys) / max(len(model_keys), 1)
        missing = sorted(model_keys - set(state_dict))
        bad_shapes = sorted(
            key for key in matched_keys
            if not hasattr(state_dict[key], 'shape') or state_dict[key].shape != model_state[key].shape
        )
        if missing or bad_shapes:
            raise RuntimeError(
                f'Checkpoint không tương thích hoàn toàn: load ratio={load_ratio:.1%}, '
                f'missing={len(missing)}, sai shape={len(bad_shapes)}. '
                f'Ví dụ missing={missing[:5]}, sai shape={bad_shapes[:5]}.'
            )
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(
                f"[FaceLandmarkInferencer] load ratio={load_ratio:.1%}, "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )
        self.model.to(self.device).eval()

    def _prepare_image(self, image_input: Union[str, PIL.Image.Image, np.ndarray]) -> np.ndarray:
        """
        Đọc và chuẩn hóa ảnh đầu vào về định dạng numpy array RGB (H, W, 3).
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Không thể đọc file ảnh: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, PIL.Image.Image):
            img_rgb = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                img_rgb = image_input.copy()
                if self.train_cfg.numpy_input_color.lower() == 'bgr':
                    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Numpy array ảnh phải có kích thước (H, W, 3).")
        else:
            raise TypeError("image_input phải là đường dẫn str, PIL Image, hoặc numpy array.")

        return img_rgb

    def restore_coordinates(
        self,
        boxes: np.ndarray,
        landmarks: np.ndarray,
        letterbox_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Khôi phục tọa độ Bounding Boxes và Landmarks từ không gian ảnh letterbox
        về không gian tọa độ ảnh gốc ban đầu.

        Args:
            boxes: Array shape (N, 4) chứa (x1, y1, x2, y2) ở ảnh letterbox.
            landmarks: Array shape (N, K, 2) chứa (x, y) ở ảnh letterbox.
            letterbox_info: Dictionary lưu trữ tham số letterbox.

        Returns:
            boxes_orig: Array (N, 4) tọa độ trên ảnh gốc.
            landmarks_orig: Array (N, K, 2) tọa độ trên ảnh gốc.
        """
        top = letterbox_info['top']
        left = letterbox_info['left']
        scale = letterbox_info['scale']
        orig_w = letterbox_info['orig_w']
        orig_h = letterbox_info['orig_h']

        boxes_orig = boxes.copy()
        landmarks_orig = landmarks.copy()

        if len(boxes_orig) > 0:
            # Khôi phục Bbox: (x - left) / scale
            boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - left) / scale
            boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - top) / scale

            # Clip về phạm vi [0, orig_w] và [0, orig_h]
            boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_w)
            boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_h)

        if len(landmarks_orig) > 0:
            # Khôi phục Landmarks: (x - left) / scale
            landmarks_orig[:, :, 0] = (landmarks_orig[:, :, 0] - left) / scale
            landmarks_orig[:, :, 1] = (landmarks_orig[:, :, 1] - top) / scale

            # Clip về phạm vi ảnh gốc
            landmarks_orig[:, :, 0] = np.clip(landmarks_orig[:, :, 0], 0, orig_w)
            landmarks_orig[:, :, 1] = np.clip(landmarks_orig[:, :, 1], 0, orig_h)

        return boxes_orig, landmarks_orig

    def draw_detections(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: np.ndarray,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        lmk_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        point_radius: int = 1
    ) -> np.ndarray:
        """
        Vẽ Bounding Boxes, Score và Landmarks lên bản sao ảnh gốc RGB.
        """
        annotated_img = image_rgb.copy()

        for box, score, lmk in zip(boxes, scores, landmarks):
            x1, y1, x2, y2 = map(int, box)

            # Vẽ bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, thickness)

            # Vẽ label score
            label = f"Face: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, max(y1 - h - 4, 0)), (x1 + w, max(y1, h + 4)), box_color, -1)
            cv2.putText(annotated_img, label, (x1, max(y1 - 2, h + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Vẽ các điểm landmark
            for pt in lmk:
                px, py = map(int, pt)
                cv2.circle(annotated_img, (px, py), point_radius, lmk_color, -1)

        return annotated_img

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, PIL.Image.Image, np.ndarray],
        show: bool = True,
        conf_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Truyền 1 ảnh qua mô hình suy luận, vẽ bbox & landmark lên ảnh gốc nếu phát hiện được,
        hiển thị ảnh bằng matplotlib và trả về ảnh gốc kèm thông tin detect chi tiết.

        Args:
            image_input: Ảnh đầu vào (Đường dẫn str, PIL Image, hoặc numpy array RGB/BGR).
            show: Có hiển thị matplotlib plot hay không (True/False).
            conf_threshold: Ngưỡng tin cậy riêng cho lần predict này (nếu None dùng self.conf_threshold).

        Returns:
            orig_img_rgb: Ảnh gốc đầu vào (numpy array RGB).
            detections: Dictionary chứa thông tin đầu ra:
                - 'boxes': np.ndarray shape (N, 4) - Tọa độ [x1, y1, x2, y2] trên ảnh gốc.
                - 'scores': np.ndarray shape (N,) - Điểm độ tin cậy.
                - 'landmarks': np.ndarray shape (N, K, 2) - Tọa độ điểm landmark trên ảnh gốc.
            letterbox_info: Dictionary chứa các thành phần phụ của quá trình resize letterbox.
        """
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold

        # 1. Đọc ảnh gốc RGB
        orig_img_rgb = self._prepare_image(image_input)

        # 2. Xử lý Resize dạng Letterbox
        letterboxed_img, letterbox_info = letterbox(
            orig_img_rgb, target_size=(self.image_size, self.image_size)
        )

        # 3. Chuyển đổi thành PyTorch Tensor (B, C, H, W) chuẩn hóa [0, 1]
        img_tensor = torch.from_numpy(letterboxed_img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 4. Chạy mô hình forward
        preds = self.model(img_tensor)

        # Trích xuất đầu ra từ nhánh suy luận trực tiếp o2o (NMS-Free)
        o2o_output = preds['o2o']
        cls_logits = o2o_output['cls'][0]          # Shape: (N, nc)
        boxes_pixel = o2o_output['box'][0]         # Shape: (N, 4)
        landmarks_pixel = o2o_output['lmk'][0]     # Shape: (N, K, 2)

        # Tính xác suất phân lớp dùng Sigmoid
        scores = torch.sigmoid(cls_logits).squeeze(-1)  # Shape: (N,)

        # 5. Lọc theo ngưỡng độ tin cậy conf_threshold
        keep_mask = scores > conf_thresh
        filtered_scores = scores[keep_mask]
        filtered_boxes = boxes_pixel[keep_mask]
        filtered_lmks = landmarks_pixel[keep_mask]

        # Chặn số candidate trước NMS/vẽ để inference ổn định khi threshold thấp.
        if len(filtered_scores) > self.max_det:
            top_indices = filtered_scores.topk(self.max_det).indices
            filtered_scores = filtered_scores[top_indices]
            filtered_boxes = filtered_boxes[top_indices]
            filtered_lmks = filtered_lmks[top_indices]

        # Áp dụng NMS nếu có nhiều candidate
        if len(filtered_scores) > 0 and self.iou_threshold > 0:
            nms_indices = nms(filtered_boxes, filtered_scores, self.iou_threshold)
            filtered_scores = filtered_scores[nms_indices]
            filtered_boxes = filtered_boxes[nms_indices]
            filtered_lmks = filtered_lmks[nms_indices]

        # 6. Khôi phục tọa độ về khung ảnh gốc
        boxes_np = filtered_boxes.cpu().numpy()
        scores_np = filtered_scores.cpu().numpy()
        lmks_np = filtered_lmks.cpu().numpy()

        orig_boxes, orig_landmarks = self.restore_coordinates(
            boxes_np, lmks_np, letterbox_info
        )

        detections = {
            'boxes': orig_boxes,
            'scores': scores_np,
            'landmarks': orig_landmarks,
        }

        # 7. Chỉ tạo ảnh annotate khi thực sự cần hiển thị.
        if show:
            import matplotlib.pyplot as plt
            annotated_img = self.draw_detections(
                orig_img_rgb, orig_boxes, scores_np, orig_landmarks
            )
            fig = plt.figure(figsize=(9, 9))
            plt.imshow(annotated_img)
            plt.title(f"Face & Landmark Detection | Số lượng khuôn mặt: {len(scores_np)}")
            plt.axis('off')
            plt.show()
            plt.close(fig)

        return orig_img_rgb, detections, letterbox_info


if __name__ == '__main__':
    print("=== Demo khoi tao FaceLandmarkInferencer ===")
    inferencer = FaceLandmarkInferencer(
        weights_path=None,  # Co the truyen duong dan file .pt o đây
        conf_threshold=0.25,
        image_size=480,
        num_landmarks=478,
    )

    # Tao 1 anh synthetic gia lap de kiem thu luong khoi tao va predict
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (200, 100), (440, 380), (255, 200, 180), -1)  # Khung mat gia lap

    orig_img, detections, letterbox_info = inferencer.predict(dummy_img, show=False)

    print("Ket qua suy luan dummy:")
    print(f"- Kich thuoc anh goc: {orig_img.shape}")
    print(f"- So luong faces detect duoc: {len(detections['scores'])}")
    print(f"- Letterbox Info: {letterbox_info}")
