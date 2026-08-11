import os
import sys
import time
from typing import Union, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import PIL.Image
from PIL import ImageOps
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
    from .config_lmk import (
        DEFAULT_PADDING_COLOR,
        MEDIAPIPE_NUM_LANDMARKS,
        PIPELINE_IMAGE_SIZE,
        TrainConfig,
    )
    from .model_lmk import FaceLmkDetector
except ImportError:
    from config_lmk import (
        DEFAULT_PADDING_COLOR,
        MEDIAPIPE_NUM_LANDMARKS,
        PIPELINE_IMAGE_SIZE,
        TrainConfig,
    )
    from model_lmk import FaceLmkDetector


# Topology MediaPipe Face Mesh dùng để nối các vùng quan trọng khi render.
_MEDIAPIPE_LIP_CONNECTIONS = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
)
_MEDIAPIPE_EYE_CONNECTIONS = (
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
    (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
    (173, 133), (263, 249), (249, 390), (390, 373), (373, 374),
    (374, 380), (380, 381), (381, 382), (382, 362), (263, 466),
    (466, 388), (388, 387), (387, 386), (386, 385), (385, 384),
    (384, 398), (398, 362),
)
_MEDIAPIPE_EYEBROW_CONNECTIONS = (
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107),
    (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336),
)
_MEDIAPIPE_RENDER_CONNECTIONS = (
    _MEDIAPIPE_LIP_CONNECTIONS
    + _MEDIAPIPE_EYE_CONNECTIONS
    + _MEDIAPIPE_EYEBROW_CONNECTIONS
)
_MEDIAPIPE_DETAIL_INDICES = tuple(sorted(
    {
        point_index
        for connection in (_MEDIAPIPE_LIP_CONNECTIONS + _MEDIAPIPE_EYE_CONNECTIONS)
        for point_index in connection
    }
    | set(range(468, 478))
))


def letterbox(
    img: np.ndarray,
    target_size: Tuple[int, int] = (PIPELINE_IMAGE_SIZE, PIPELINE_IMAGE_SIZE),
    color: Tuple[int, int, int] = DEFAULT_PADDING_COLOR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Letterbox giống lúc train: resize Bilinear, sau đó chia padding dư
    theo phía trái/trên trước và phải/dưới sau.

    Args:
        img: Ảnh RGB dạng numpy array (H, W, 3).
        target_size: Kích thước mong muốn (target_h, target_w).
        color: Màu RGB của phần padding.

    Returns:
        padded_img: Ảnh đã qua letterbox có kích thước (target_h, target_w, C).
        info: Dictionary chứa các tham số phục vụ việc khôi phục tọa độ về ảnh gốc.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError('letterbox yêu cầu ảnh RGB có shape (H, W, 3).')

    orig_h, orig_w = img.shape[:2]
    target_h, target_w = target_size
    if min(orig_h, orig_w, target_h, target_w) <= 0:
        raise ValueError('Kích thước ảnh và target_size phải > 0.')

    scale = min(target_w / orig_w, target_h / orig_h)
    new_unpad_w = int(round(orig_w * scale))
    new_unpad_h = int(round(orig_h * scale))

    left = (target_w - new_unpad_w) // 2
    top = (target_h - new_unpad_h) // 2
    right = target_w - new_unpad_w - left
    bottom = target_h - new_unpad_h - top

    # Dùng cùng PIL Bilinear và ImageOps.expand như dataloader train.
    pil_image = PIL.Image.fromarray(np.ascontiguousarray(img))
    if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
        pil_image = pil_image.resize(
            (new_unpad_w, new_unpad_h),
            PIL.Image.Resampling.BILINEAR,
        )
    padded_img = np.asarray(
        ImageOps.expand(pil_image, (left, top, right, bottom), fill=color),
        dtype=np.uint8,
    ).copy()

    info = {
        'scale': scale,
        'pad_w': left,
        'pad_h': top,
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
        'new_unpad_w': new_unpad_w,
        'new_unpad_h': new_unpad_h,
        'orig_w': orig_w,
        'orig_h': orig_h,
        'target_w': target_w,
        'target_h': target_h,
    }

    if padded_img.shape[:2] != (target_h, target_w):
        raise RuntimeError(
            f'Letterbox tạo shape {padded_img.shape[:2]}, cần {(target_h, target_w)}.'
        )

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
        max_det: Optional[int] = None,
        enhance_details: bool = False,
        refine_eye_mouth: bool = False,
        max_refine_faces: int = 3,
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
            Landmark được decode trực tiếp từ anchor + signed offset; không dùng bbox margin.
            enhance_details: Dùng CLAHE + unsharp nhẹ trên crop refinement.
            refine_eye_mouth: Chạy thêm một lượt trên crop mặt để tinh chỉnh mắt/môi.
            max_refine_faces: Số khuôn mặt tối đa được refinement trong mỗi ảnh/frame.
        """
        self.train_cfg = cfg or TrainConfig()
        if image_size is not None and image_size != self.train_cfg.image_size:
            raise ValueError(f'image_size phải đồng nhất ở {self.train_cfg.image_size}.')
        if num_landmarks is not None:
            self.train_cfg.face.num_landmarks = num_landmarks
        if self.train_cfg.face.num_landmarks is None:
            # Inference không có dataset để sync; checkpoint có thể điều chỉnh lại K bên dưới.
            self.train_cfg.face.num_landmarks = MEDIAPIPE_NUM_LANDMARKS

        if device is None:
            requested_device = self.train_cfg.device
            self.device = torch.device(
                requested_device
                if requested_device == 'cpu' or torch.cuda.is_available()
                else 'cpu'
            )
        else:
            self.device = torch.device(device)

        self.conf_threshold = self.train_cfg.inference_conf_threshold if conf_threshold is None else conf_threshold
        self.iou_threshold = self.train_cfg.inference_iou_threshold if iou_threshold is None else iou_threshold
        self.max_det = self.train_cfg.inference_max_det if max_det is None else max_det
        if not 0 <= self.conf_threshold <= 1 or not 0 <= self.iou_threshold <= 1:
            raise ValueError('conf_threshold và iou_threshold phải nằm trong [0, 1].')
        if self.max_det <= 0:
            raise ValueError('max_det phải > 0.')
        if max_refine_faces <= 0:
            raise ValueError('max_refine_faces phải > 0.')
        self.image_size = self.train_cfg.image_size
        self.num_landmarks = self.train_cfg.face.require_num_landmarks()
        self.enhance_details = bool(enhance_details)
        self.refine_eye_mouth = bool(refine_eye_mouth)
        self.max_refine_faces = int(max_refine_faces)

        # Các hệ số cố định, cố ý nhẹ để không làm phân phối ảnh lệch xa lúc train.
        self._clahe_clip_limit = 2.0
        self._unsharp_amount = 0.25
        self._refine_margin = 0.20
        self._refine_match_iou = 0.25
        self._refine_blend = 0.75
        self._refine_max_median_shift = 0.18

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

        if not isinstance(checkpoint, dict):
            raise TypeError('Checkpoint phải là dict hoặc state_dict.')

        saved_signature = checkpoint.get('model_signature')
        if saved_signature is None:
            raise ValueError(
                'Checkpoint không có model_signature. Không thể xác nhận đây là '
                'checkpoint anchor-relative; từ chối nạp để tránh decode landmark sai.'
            )
        saved_encoding = saved_signature.get('landmark_encoding')
        expected_encoding = self.train_cfg.face.landmark_encoding
        if saved_encoding != expected_encoding:
            raise ValueError(
                f'Checkpoint dùng landmark_encoding={saved_encoding!r}, '
                f'nhưng model hiện tại yêu cầu {expected_encoding!r}.'
            )

        saved_num_landmarks = int(saved_signature.get('num_landmarks', self.num_landmarks))
        if saved_num_landmarks != self.num_landmarks:
            print(
                f"[FaceLandmarkInferencer] Điều chỉnh num_landmarks từ "
                f"{self.num_landmarks} -> {saved_num_landmarks} theo model_signature."
            )
            self.num_landmarks = saved_num_landmarks
            self.train_cfg.face.num_landmarks = saved_num_landmarks
            self.model = FaceLmkDetector(self.train_cfg)

        expected_signature = self.train_cfg.checkpoint_model_signature()
        if saved_signature != expected_signature:
            raise ValueError(
                'Model signature trong checkpoint khác config inference hiện tại: '
                f'checkpoint={saved_signature}, current={expected_signature}.'
            )

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

        # Kiểm tra thêm shape head để phát hiện checkpoint hỏng hoặc metadata sai.
        for key in ['head.heads.0.lmk_o2o.weight', 'head.heads.0.lmk_o2m.weight']:
            if key in state_dict:
                detected_num_lmk = state_dict[key].shape[0] // 2
                if detected_num_lmk != self.num_landmarks:
                    raise RuntimeError(
                        f'model_signature báo K={self.num_landmarks}, nhưng {key} '
                        f'có shape tương ứng K={detected_num_lmk}.'
                    )
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

        if img_rgb.size == 0:
            raise ValueError('Ảnh đầu vào không được rỗng.')
        if not np.issubdtype(img_rgb.dtype, np.number):
            raise TypeError(f'Dtype ảnh không được hỗ trợ: {img_rgb.dtype}.')
        if not np.isfinite(img_rgb).all():
            raise ValueError('Ảnh đầu vào chứa NaN hoặc Inf.')
        if np.issubdtype(img_rgb.dtype, np.floating):
            min_value = float(img_rgb.min())
            max_value = float(img_rgb.max())
            if 0.0 <= min_value and max_value <= 1.0:
                img_rgb = img_rgb * 255.0
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8, copy=False)
        return np.ascontiguousarray(img_rgb)

    def _enhance_detail_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Tăng tương phản cục bộ và độ nét nhẹ, không thay đổi hình học ảnh."""
        source = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip_limit,
            tileGridSize=(8, 8),
        )
        enhanced_lightness = clahe.apply(lightness)
        contrast_rgb = cv2.cvtColor(
            cv2.merge((enhanced_lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2RGB,
        )

        blurred = cv2.GaussianBlur(contrast_rgb, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(
            contrast_rgb,
            1.0 + self._unsharp_amount,
            blurred,
            -self._unsharp_amount,
            0.0,
        )
        # Giữ một phần ảnh gốc để giảm domain shift so với ảnh lúc train.
        return cv2.addWeighted(source, 0.25, sharpened, 0.75, 0.0)

    @torch.no_grad()
    def _infer_single_pass(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Một lượt model hoàn chỉnh và trả tọa độ trong ảnh RGB đầu vào."""
        letterboxed_img, letterbox_info = letterbox(
            image_rgb,
            target_size=(self.image_size, self.image_size),
            color=self.train_cfg.padding_color,
        )
        img_tensor = (
            torch.from_numpy(letterboxed_img)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        predictions = self.model(img_tensor, return_o2m=False)['o2o']
        scores = torch.sigmoid(predictions['cls'][0]).squeeze(-1)
        keep_mask = scores > conf_threshold
        filtered_scores = scores[keep_mask]
        filtered_boxes = predictions['box'][0][keep_mask]
        filtered_landmarks = predictions['lmk'][0][keep_mask]

        if len(filtered_scores) > self.max_det:
            top_indices = filtered_scores.topk(self.max_det).indices
            filtered_scores = filtered_scores[top_indices]
            filtered_boxes = filtered_boxes[top_indices]
            filtered_landmarks = filtered_landmarks[top_indices]

        if len(filtered_scores) > 0 and self.iou_threshold > 0:
            keep_indices = nms(
                filtered_boxes,
                filtered_scores,
                self.iou_threshold,
            )
            filtered_scores = filtered_scores[keep_indices]
            filtered_boxes = filtered_boxes[keep_indices]
            filtered_landmarks = filtered_landmarks[keep_indices]

        boxes, landmarks = self.restore_coordinates(
            filtered_boxes.detach().cpu().numpy(),
            filtered_landmarks.detach().cpu().numpy(),
            letterbox_info,
        )
        score_values = filtered_scores.detach().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        if len(boxes) > 0:
            valid = (
                np.isfinite(boxes).all(axis=1)
                & np.isfinite(landmarks).all(axis=(1, 2))
                & (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
            )
            boxes = boxes[valid]
            landmarks = landmarks[valid]
            score_values = score_values[valid]

        detections = {
            'boxes': boxes.astype(np.float32, copy=False),
            'scores': score_values,
            'landmarks': landmarks.astype(np.float32, copy=False),
        }
        return detections, letterbox_info

    @staticmethod
    def _box_iou_one_to_many(
        box: np.ndarray,
        other_boxes: np.ndarray,
    ) -> np.ndarray:
        """IoU giữa một box xyxy và N box xyxy, hoàn toàn bằng NumPy."""
        if len(other_boxes) == 0:
            return np.empty((0,), dtype=np.float32)
        x1 = np.maximum(box[0], other_boxes[:, 0])
        y1 = np.maximum(box[1], other_boxes[:, 1])
        x2 = np.minimum(box[2], other_boxes[:, 2])
        y2 = np.minimum(box[3], other_boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        box_area = max(float(box[2] - box[0]), 0.0) * max(
            float(box[3] - box[1]),
            0.0,
        )
        other_areas = (
            np.maximum(other_boxes[:, 2] - other_boxes[:, 0], 0.0)
            * np.maximum(other_boxes[:, 3] - other_boxes[:, 1], 0.0)
        )
        union = box_area + other_areas - intersection
        return (intersection / np.maximum(union, 1e-9)).astype(np.float32)

    def _refine_eye_mouth_landmarks(
        self,
        image_rgb: np.ndarray,
        detections: Dict[str, np.ndarray],
        conf_threshold: float,
    ) -> Dict[str, np.ndarray]:
        """Phóng crop mặt lên input model và chỉ blend lại mắt/iris/môi."""
        if (
            len(detections['scores']) == 0
            or self.num_landmarks != MEDIAPIPE_NUM_LANDMARKS
        ):
            return detections

        refined = {key: value.copy() for key, value in detections.items()}
        image_h, image_w = image_rgb.shape[:2]
        detail_indices = np.asarray(
            [index for index in _MEDIAPIPE_DETAIL_INDICES if index < self.num_landmarks],
            dtype=np.int64,
        )
        if len(detail_indices) == 0:
            return refined

        face_order = np.argsort(-refined['scores'])[:self.max_refine_faces]
        local_conf_threshold = max(0.05, min(conf_threshold, 0.15))
        for face_index in face_order:
            box = refined['boxes'][face_index]
            face_w = float(box[2] - box[0])
            face_h = float(box[3] - box[1])
            if face_w <= 1.0 or face_h <= 1.0:
                continue

            margin_x = face_w * self._refine_margin
            margin_y = face_h * self._refine_margin
            crop_x1 = max(0, int(np.floor(box[0] - margin_x)))
            crop_y1 = max(0, int(np.floor(box[1] - margin_y)))
            crop_x2 = min(image_w, int(np.ceil(box[2] + margin_x)))
            crop_y2 = min(image_h, int(np.ceil(box[3] + margin_y)))
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            if min(crop_w, crop_h) < 32:
                continue
            if crop_w * crop_h >= 0.90 * image_w * image_h:
                # Mặt đã chiếm gần toàn bộ frame; crop không tăng độ phân giải.
                continue

            crop_rgb = np.ascontiguousarray(
                image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            )
            crop_for_model = (
                self._enhance_detail_image(crop_rgb)
                if self.enhance_details
                else crop_rgb
            )
            local_detections, _ = self._infer_single_pass(
                crop_for_model,
                local_conf_threshold,
            )
            if len(local_detections['scores']) == 0:
                continue

            local_boxes_global = local_detections['boxes'].copy()
            local_boxes_global[:, [0, 2]] += crop_x1
            local_boxes_global[:, [1, 3]] += crop_y1
            overlaps = self._box_iou_one_to_many(box, local_boxes_global)
            candidate_indices = np.flatnonzero(overlaps >= self._refine_match_iou)
            if len(candidate_indices) == 0:
                continue
            candidate_quality = (
                local_detections['scores'][candidate_indices]
                * (0.25 + 0.75 * overlaps[candidate_indices])
            )
            best_local_index = int(candidate_indices[np.argmax(candidate_quality)])
            local_landmarks = local_detections['landmarks'][best_local_index].copy()
            local_landmarks[:, 0] += crop_x1
            local_landmarks[:, 1] += crop_y1

            base_details = refined['landmarks'][face_index, detail_indices]
            local_details = local_landmarks[detail_indices]
            face_diagonal = max(float(np.hypot(face_w, face_h)), 1.0)
            median_shift = float(np.median(
                np.linalg.norm(local_details - base_details, axis=1)
            )) / face_diagonal
            if median_shift > self._refine_max_median_shift:
                # Bbox có thể match nhưng landmark crop đã bám nhầm cấu trúc.
                continue
            refined['landmarks'][face_index, detail_indices] = (
                (1.0 - self._refine_blend) * base_details
                + self._refine_blend * local_details
            )

        return refined

    @staticmethod
    def restore_coordinates(
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
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f'boxes phải có shape (N, 4), nhận {boxes.shape}.')
        if landmarks.ndim != 3 or landmarks.shape[0] != boxes.shape[0] or landmarks.shape[2] != 2:
            raise ValueError(
                'landmarks phải có shape (N, K, 2) và cùng N với boxes, '
                f'nhận {landmarks.shape}.'
            )

        top = float(letterbox_info['top'])
        left = float(letterbox_info['left'])
        scale = float(letterbox_info['scale'])
        orig_w = int(letterbox_info['orig_w'])
        orig_h = int(letterbox_info['orig_h'])
        if scale <= 0 or orig_w <= 0 or orig_h <= 0:
            raise ValueError('Metadata letterbox không hợp lệ.')

        boxes_orig = boxes.astype(np.float32, copy=True)
        landmarks_orig = landmarks.astype(np.float32, copy=True)

        if len(boxes_orig) > 0:
            # Bỏ padding rồi chia scale để trở về hệ tọa độ ảnh gốc.
            boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - left) / scale
            boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - top) / scale

            # Clip về phạm vi [0, orig_w] và [0, orig_h]
            boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, orig_w)
            boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, orig_h)

        if len(landmarks_orig) > 0:
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
        box_color: Optional[Tuple[int, int, int]] = None,
        lmk_color: Optional[Tuple[int, int, int]] = None,
        thickness: Optional[int] = None,
        point_radius: Optional[int] = None,
    ) -> np.ndarray:
        """
        Vẽ Bounding Boxes, Score và Landmarks lên bản sao ảnh gốc RGB.
        """
        annotated_img = image_rgb.copy()
        box_color = box_color or self.train_cfg.inference_box_color
        lmk_color = lmk_color or self.train_cfg.inference_landmark_color
        thickness = thickness or self.train_cfg.inference_box_thickness
        point_radius = point_radius or self.train_cfg.inference_landmark_radius
        text_scale = self.train_cfg.inference_text_scale
        text_thickness = self.train_cfg.inference_text_thickness
        text_color = self.train_cfg.inference_text_color
        connection_thickness = max(1, point_radius)

        for box, score, lmk in zip(boxes, scores, landmarks):
            x1, y1, x2, y2 = map(int, box)

            # Vẽ bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, thickness)

            # Vẽ label score
            label = f"Face: {score:.2f}"
            (w, h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                text_thickness,
            )
            cv2.rectangle(annotated_img, (x1, max(y1 - h - 4, 0)), (x1 + w, max(y1, h + 4)), box_color, -1)
            cv2.putText(annotated_img, label, (x1, max(y1 - 2, h + 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color,
                        text_thickness, cv2.LINE_AA)

            # Nối landmark thành đường viền môi, mắt và lông mày. Nếu model
            # không đủ 478 điểm thì tự bỏ qua connection nằm ngoài phạm vi.
            num_points = len(lmk)
            for start_idx, end_idx in _MEDIAPIPE_RENDER_CONNECTIONS:
                if start_idx >= num_points or end_idx >= num_points:
                    continue
                start = lmk[start_idx]
                end = lmk[end_idx]
                if not np.isfinite(start).all() or not np.isfinite(end).all():
                    continue
                start_xy = tuple(np.rint(start).astype(int))
                end_xy = tuple(np.rint(end).astype(int))
                cv2.line(
                    annotated_img,
                    start_xy,
                    end_xy,
                    lmk_color,
                    connection_thickness,
                    cv2.LINE_AA,
                )

            # Vẽ các điểm landmark trên các đường nối.
            for pt in lmk:
                px, py = map(int, pt)
                cv2.circle(annotated_img, (px, py), point_radius, lmk_color, -1)

        return annotated_img

    def show_matplotlib(
        self,
        image_rgb: np.ndarray,
        title: str,
        *,
        block: bool = True,
    ) -> None:
        """Hiển thị ảnh RGB bằng Matplotlib."""
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=self.train_cfg.inference_figure_size)
        axis.imshow(image_rgb)
        axis.set_title(title)
        axis.axis('off')
        fig.tight_layout()
        plt.show(block=block)
        if block:
            plt.close(fig)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, PIL.Image.Image, np.ndarray],
        show: bool = True,
        conf_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Suy luận, bỏ padding, fit bbox/landmark về ảnh gốc và vẽ kết quả
        trên canvas có đúng kích thước ảnh ban đầu.

        Args:
            image_input: Ảnh đầu vào (Đường dẫn str, PIL Image, hoặc numpy array RGB/BGR).
            show: Có hiển thị matplotlib plot hay không (True/False).
            conf_threshold: Ngưỡng tin cậy riêng cho lần predict này (nếu None dùng self.conf_threshold).

        Returns:
            output_img_rgb: Ảnh RGB đã vẽ kết quả, cùng shape ảnh gốc.
            detections: Dictionary chứa thông tin đầu ra:
                - 'boxes': np.ndarray shape (N, 4) - Tọa độ [x1, y1, x2, y2] trên ảnh gốc.
                - 'scores': np.ndarray shape (N,) - Điểm độ tin cậy.
                - 'landmarks': np.ndarray shape (N, K, 2) - Tọa độ điểm landmark trên ảnh gốc.
            letterbox_info: Dictionary chứa các thành phần phụ của quá trình resize letterbox.
        """
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        if not 0 <= conf_thresh <= 1:
            raise ValueError('conf_threshold phải nằm trong [0, 1].')

        # 1. Đọc ảnh gốc RGB
        orig_img_rgb = self._prepare_image(image_input)

        # 2. Lượt chính luôn dùng ảnh gốc và letterbox y hệt lúc train.
        detections, letterbox_info = self._infer_single_pass(
            orig_img_rgb,
            conf_thresh,
        )

        # 3. CLAHE/unsharp toàn frame chỉ là fallback khi ảnh gốc không detect
        # được face; tránh thay phân phối đầu vào trong trường hợp bình thường.
        if len(detections['scores']) == 0 and self.enhance_details:
            enhanced_full_frame = self._enhance_detail_image(orig_img_rgb)
            detections, _ = self._infer_single_pass(
                enhanced_full_frame,
                conf_thresh,
            )

        # 4. Crop mặt có margin, phóng qua input 480 rồi chỉ blend mắt/iris/môi.
        if self.refine_eye_mouth:
            detections = self._refine_eye_mouth_landmarks(
                orig_img_rgb,
                detections,
                conf_thresh,
            )

        orig_boxes = detections['boxes']
        scores_np = detections['scores']
        orig_landmarks = detections['landmarks']

        # Luôn vẽ trên ảnh gốc, không resize ảnh output về 480x480.
        output_img_rgb = self.draw_detections(
            orig_img_rgb,
            orig_boxes,
            scores_np,
            orig_landmarks,
        )
        if output_img_rgb.shape != orig_img_rgb.shape:
            raise RuntimeError(
                f'Ảnh output có shape {output_img_rgb.shape}, '
                f'khác ảnh gốc {orig_img_rgb.shape}.'
            )

        # 5. Phần hiển thị chỉ dùng Matplotlib.
        if show:
            self.show_matplotlib(
                output_img_rgb,
                f"Face & Landmark Detection | Số khuôn mặt: {len(scores_np)}",
            )

        return output_img_rgb, detections, letterbox_info


def demo_image(
    inferencer: FaceLandmarkInferencer,
    image_input: Union[str, PIL.Image.Image, np.ndarray],
    conf_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Demo suy luận một ảnh và hiển thị kết quả bằng Matplotlib."""
    output, detections, info = inferencer.predict(
        image_input,
        show=True,
        conf_threshold=conf_threshold,
    )
    print(
        f'[Image] output={output.shape}, faces={len(detections["scores"])}, '
        f'scale={info["scale"]:.6f}, padding='
        f'({info["left"]}, {info["top"]}, {info["right"]}, {info["bottom"]})'
    )
    return output, detections, info


def demo_camera(
    inferencer: FaceLandmarkInferencer,
    camera_index: int = 0,
    conf_threshold: Optional[float] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> None:
    """Detect camera liên tục; đóng cửa sổ Matplotlib hoặc Ctrl+C để dừng."""
    import matplotlib.pyplot as plt

    capture = cv2.VideoCapture(camera_index)
    if frame_width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    if frame_height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'Không thể mở camera index={camera_index}.')

    plt.ion()
    fig, axis = plt.subplots(figsize=inferencer.train_cfg.inference_figure_size)
    artist = None
    axis.axis('off')
    plt.show(block=False)

    try:
        while plt.fignum_exists(fig.number):
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError('Không đọc được frame từ camera.')

            # OpenCV đọc BGR; chuyển sang RGB trước khi predict/Matplotlib.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            output, detections, _ = inferencer.predict(
                PIL.Image.fromarray(frame_rgb),
                show=False,
                conf_threshold=conf_threshold,
            )
            elapsed = max(time.perf_counter() - started, 1e-9)

            if artist is None:
                artist = axis.imshow(output)
            else:
                artist.set_data(output)
            axis.set_title(
                f'Camera {camera_index} | Faces: {len(detections["scores"])} '
                f'| FPS: {1.0 / elapsed:.1f}'
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
    except KeyboardInterrupt:
        print('\n[Camera] Đã dừng bởi người dùng.')
    finally:
        capture.release()
        plt.ioff()
        plt.close(fig)


def main(
    mode: str = 'image',
    image_input: Optional[Union[str, PIL.Image.Image, np.ndarray]] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_det: Optional[int] = None,
    enhance_details: bool = False,
    refine_eye_mouth: bool = False,
    max_refine_faces: int = 3,
    camera_index: int = 0,
    camera_width: Optional[int] = None,
    camera_height: Optional[int] = None,
):
    """Chạy demo trực tiếp trong Python, không dùng CLI."""
    mode = mode.lower()
    if mode not in {'image', 'camera'}:
        raise ValueError("mode phải là 'image' hoặc 'camera'.")
    if camera_width is not None and camera_width <= 0:
        raise ValueError('camera_width phải > 0.')
    if camera_height is not None and camera_height <= 0:
        raise ValueError('camera_height phải > 0.')

    config = TrainConfig(require_pretrained_trunk=False)
    inferencer = FaceLandmarkInferencer(
        weights_path=weights_path,
        cfg=config,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
        enhance_details=enhance_details,
        refine_eye_mouth=refine_eye_mouth,
        max_refine_faces=max_refine_faces,
    )

    if mode == 'image':
        if image_input is None:
            image_input = config.demo_model_image_path
        return demo_image(
            inferencer,
            image_input,
            conf_threshold=conf_threshold,
        )

    return demo_camera(
        inferencer,
        camera_index=camera_index,
        conf_threshold=conf_threshold,
        frame_width=camera_width,
        frame_height=camera_height,
    )

if __name__ == '__main__':
    # Chọn 'image' hoặc 'camera' và thay tham số trực tiếp tại đây.
    main(
        mode='camera',
        image_input="/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/src/image_sketch/demo/image.png",       # None -> dùng config.demo_model_image_path
        weights_path="/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/checkpoints_face_lmk_finetune/best.pt",      # Điền checkpoint .pt để detect thực tế
        device="cuda",            # None -> tự chọn theo TrainConfig
        conf_threshold=0.25,    # None -> dùng TrainConfig
        iou_threshold=0.4,
        max_det=None,
        enhance_details=False,     # False: giữ đúng phân phối ảnh như lúc train
        refine_eye_mouth=False,    # False: chỉ 1 forward/frame, chất lượng ổn định hơn
        max_refine_faces=1,        # Chỉ có tác dụng khi refine_eye_mouth=True
        camera_index=0,
        camera_width=None,
        camera_height=None,
    )
