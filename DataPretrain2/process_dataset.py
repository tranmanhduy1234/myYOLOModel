"""
process_dataset.py
===================

Script tải ảnh từ dataset HuggingFace "wuji3/face-recognition", lưu ảnh vào
thư mục con `Images/`, chạy MediaPipe Face Landmarker trên từng ảnh, và lưu
TOÀN BỘ dữ liệu landmark vào MỘT file JSON duy nhất (tách riêng khỏi thư mục
`Images/`). File JSON là một object, key là tên ảnh (id), value là dữ liệu
landmark tương ứng theo cấu trúc mô tả trong modify.md.

Yêu cầu cài đặt trước khi chạy:
    pip install datasets mediapipe opencv-python pillow

Bạn cũng cần tải sẵn model MediaPipe Face Landmarker (file .task), ví dụ:
    wget -O face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Cách chạy ví dụ:
    python3 process_dataset.py \
        --output-dir /run/media/tranmanhduy/Data/DataForCNN \
        --model face_landmarker.task \
        --split train \
        --num-samples 100 \
        --start 0 \
        --skip-no-face \
        --draw

Kết quả:
    <output-dir>/Images/<file_name>.jpg           (ảnh gốc)
    <output-dir>/Images/drawn/<file_name>.jpg     (ảnh có vẽ landmark, nếu --draw)
    <output-dir>/annotations.json                 (1 file JSON duy nhất chứa dữ liệu tất cả ảnh, key = tên ảnh)
"""

import argparse
import json
import os
import sys

# Tắt bớt log rác của TensorFlow Lite / absl / MediaPipe trước khi import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tải ảnh từ dataset HuggingFace và trích xuất face landmarks bằng MediaPipe."
    )
    parser.add_argument(
        "--dataset", type=str, default="wuji3/face-recognition",
        help="Tên dataset trên HuggingFace Hub."
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Split của dataset cần dùng (mặc định: train)."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Thư mục gốc để lưu (sẽ tạo thư mục con 'Images' bên trong)."
    )
    parser.add_argument(
        "--model", type=str, default="face_landmarker.task",
        help="Đường dẫn tới model MediaPipe Face Landmarker (.task)."
    )
    parser.add_argument(
        "--num-faces", type=int, default=1,
        help="Số khuôn mặt tối đa cần nhận diện trên mỗi ảnh."
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Vị trí bắt đầu lấy mẫu trong dataset (để xử lý theo lô)."
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Số lượng ảnh cần tải và xử lý (mặc định: 100). Dùng -1 để lấy tất cả."
    )
    parser.add_argument(
        "--draw", action="store_true",
        help="Nếu bật, sẽ lưu thêm ảnh có vẽ landmark vào thư mục con 'Images/drawn'."
    )
    parser.add_argument(
        "--skip-no-face", action="store_true",
        help="Nếu bật, bỏ qua (không lưu) những ảnh không phát hiện được khuôn mặt."
    )
    parser.add_argument(
        "--annotations-output", type=str, default=None,
        help="Đường dẫn file JSON tổng hợp (mặc định: <output-dir>/annotations.json)."
    )
    parser.add_argument(
        "--save-every", type=int, default=200,
        help="Lưu checkpoint annotations.json sau mỗi N ảnh xử lý (mặc định: 200). Quan trọng khi tải số lượng lớn để không mất dữ liệu nếu bị ngắt giữa chừng."
    )
    return parser.parse_args()


def build_landmarker(model_path: str, num_faces: int):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if not os.path.exists(model_path):
        print(f"Lỗi: không tìm thấy model '{model_path}'.")
        print("Tải model tại: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        sys.exit(1)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=num_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_face_data(result, width: int, height: int):
    """Chuyển kết quả detect() của MediaPipe thành list các dict theo cấu trúc modify.md."""
    faces_out = []

    if not result.face_landmarks:
        return faces_out

    for face_idx, landmarks in enumerate(result.face_landmarks):
        face_data = {
            "face_index": face_idx,
            "landmarks_normalized": [],
            "landmarks_pixel": [],
            "blendshapes": [],
            "facial_transformation_matrix": None,
        }

        for lm_idx, landmark in enumerate(landmarks):
            face_data["landmarks_normalized"].append({
                "index": lm_idx,
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
            })
            pixel_x = int(landmark.x * width)
            pixel_y = int(landmark.y * height)
            pixel_z = float(landmark.z * width)
            face_data["landmarks_pixel"].append({
                "index": lm_idx,
                "x": pixel_x,
                "y": pixel_y,
                "z": pixel_z,
            })

        x_norms = [float(lm.x) for lm in landmarks]
        y_norms = [float(lm.y) for lm in landmarks]
        xmin_norm, xmax_norm = min(x_norms), max(x_norms)
        ymin_norm, ymax_norm = min(y_norms), max(y_norms)

        face_data["bounding_box_normalized"] = {
            "xmin": xmin_norm,
            "ymin": ymin_norm,
            "xmax": xmax_norm,
            "ymax": ymax_norm,
            "width": xmax_norm - xmin_norm,
            "height": ymax_norm - ymin_norm,
        }

        xmin_px = int(xmin_norm * width)
        ymin_px = int(ymin_norm * height)
        xmax_px = int(xmax_norm * width)
        ymax_px = int(ymax_norm * height)

        face_data["bounding_box_pixel"] = {
            "xmin": xmin_px,
            "ymin": ymin_px,
            "xmax": xmax_px,
            "ymax": ymax_px,
            "width": xmax_px - xmin_px,
            "height": ymax_px - ymin_px,
        }

        if result.face_blendshapes and face_idx < len(result.face_blendshapes):
            for blendshape in result.face_blendshapes[face_idx]:
                face_data["blendshapes"].append({
                    "category_name": blendshape.category_name,
                    "score": float(blendshape.score),
                })

        if result.facial_transformation_matrixes and face_idx < len(result.facial_transformation_matrixes):
            matrix = result.facial_transformation_matrixes[face_idx]
            face_data["facial_transformation_matrix"] = [list(map(float, row)) for row in matrix]

        faces_out.append(face_data)

    return faces_out


def draw_landmarks_on_image(image_bgr, result):
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2

    drawn_image = image_bgr.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    height, width = image_bgr.shape[:2]

    for face_idx, face_landmarks in enumerate(result.face_landmarks):
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in face_landmarks
        ])

        mp_drawing.draw_landmarks(
            image=drawn_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=drawn_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp_drawing.draw_landmarks(
            image=drawn_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

        xs = [lm.x * width for lm in face_landmarks]
        ys = [lm.y * height for lm in face_landmarks]
        x_min_px, x_max_px = int(min(xs)), int(max(xs))
        y_min_px, y_max_px = int(min(ys)), int(max(ys))
        cv2.rectangle(drawn_image, (x_min_px - 10, y_min_px - 10), (x_max_px + 10, y_max_px + 10), (0, 200, 255), 2)
        cv2.putText(drawn_image, f"Face #{face_idx}", (x_min_px - 10, max(y_min_px - 15, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    return drawn_image


def main():
    args = parse_args()

    # Import ở đây để thông báo lỗi rõ ràng nếu thiếu thư viện `datasets`
    try:
        from datasets import load_dataset
    except ImportError:
        print("Thiếu thư viện 'datasets'. Cài đặt bằng: pip install datasets")
        sys.exit(1)

    import mediapipe as mp

    images_dir = os.path.join(args.output_dir, "Images")
    os.makedirs(images_dir, exist_ok=True)
    drawn_dir = os.path.join(images_dir, "drawn")
    if args.draw:
        os.makedirs(drawn_dir, exist_ok=True)

    annotations_path = args.annotations_output or os.path.join(args.output_dir, "annotations.json")

    # Resume: nếu đã có file annotations từ lần chạy trước, nạp lại để bỏ qua
    # những ảnh đã xử lý (tránh tải và detect lại từ đầu khi bị ngắt giữa chừng).
    annotations = {}
    if os.path.exists(annotations_path):
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            print(f"Tìm thấy checkpoint cũ: {len(annotations)} ảnh đã xử lý trước đó. Sẽ tiếp tục và bỏ qua các ảnh này.")
        except (json.JSONDecodeError, OSError):
            print("Cảnh báo: không đọc được checkpoint cũ, sẽ tạo mới.")
            annotations = {}

    print(f"Đang tải thông tin dataset '{args.dataset}' (split={args.split})...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    # Bỏ qua `start` phần tử đầu, rồi lấy `num_samples` phần tử
    if args.start > 0:
        ds = ds.skip(args.start)

    print("Khởi tạo Face Landmarker...")
    landmarker = build_landmarker(args.model, args.num_faces)

    processed = 0
    detected_count = 0
    skipped_count = 0
    resumed_count = 0

    def save_checkpoint():
        tmp_path = annotations_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, annotations_path)

    try:
        for i, sample in enumerate(ds):
            if args.num_samples != -1 and processed >= args.num_samples:
                break

            file_name = sample.get("file_name") or f"image_{args.start + i:07d}.jpg"

            # Resume: bỏ qua ảnh đã có trong checkpoint từ lần chạy trước
            if file_name in annotations:
                resumed_count += 1
                processed += 1
                continue

            pil_image = sample["image"]  # PIL.Image
            label = sample.get("label")
            class_name = sample.get("class_name")

            # Đảm bảo ảnh ở dạng RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            image_path = os.path.join(images_dir, file_name)
            pil_image.save(image_path)

            rgb_np = np.array(pil_image)
            width, height = pil_image.width, pil_image.height

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
            result = landmarker.detect(mp_image)

            faces_data = extract_face_data(result, width, height)

            if not faces_data and args.skip_no_face:
                os.remove(image_path)
                skipped_count += 1
                processed += 1
                continue

            annotations[file_name] = {
                "width": width,
                "height": height,
                "label": label,
                "class_name": class_name,
                "faces": faces_data,
            }

            if faces_data:
                detected_count += 1
                if args.draw:
                    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                    drawn = draw_landmarks_on_image(bgr_np, result)
                    cv2.imwrite(os.path.join(drawn_dir, file_name), drawn)

            processed += 1
            if processed % 50 == 0:
                print(f"Đã xử lý {processed} ảnh... (phát hiện mặt: {detected_count}, bỏ qua: {skipped_count})")

            if processed % args.save_every == 0:
                save_checkpoint()

    except KeyboardInterrupt:
        print("\nĐã ngắt bởi người dùng (Ctrl+C). Đang lưu checkpoint trước khi thoát...")
    finally:
        landmarker.close()
        save_checkpoint()

    print("Hoàn tất!")
    print(f"Tổng số ảnh xử lý trong lần chạy này: {processed}")
    if resumed_count:
        print(f"Số ảnh bỏ qua vì đã có sẵn trong checkpoint (resume): {resumed_count}")
    print(f"Số ảnh phát hiện được khuôn mặt: {detected_count}")
    if args.skip_no_face:
        print(f"Số ảnh bị bỏ qua (không có mặt): {skipped_count}")
    print(f"Ảnh được lưu tại: {images_dir}")
    print(f"File JSON tổng hợp được lưu tại: {annotations_path}")

if __name__ == "__main__":
    main()
    os._exit(0)