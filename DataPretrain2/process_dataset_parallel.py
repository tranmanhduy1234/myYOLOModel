"""
process_dataset_parallel.py
============================

Phiên bản CHỊU LỖI + SONG SONG của process_dataset.py, dùng cho việc tải
và xử lý số lượng lớn ảnh (ví dụ 1.000.000 ảnh) từ dataset HuggingFace
"wuji3/face-recognition" bằng MediaPipe Face Landmarker.

THIẾT KẾ (khác bản gốc):

1. CHIA SHARD SONG SONG
   - Toàn bộ num_samples được chia đều cho --num-workers tiến trình.
   - Mỗi tiến trình mở kết nối streaming RIÊNG tới HuggingFace và xử lý
     một đoạn liên tục [shard_start, shard_start+shard_len) của dataset.
   - Vì mỗi tiến trình tải mạng độc lập -> tốc độ tải tăng gần tuyến tính
     theo số worker (giới hạn bởi băng thông mạng và số nhân CPU).

2. GHI JSONL (APPEND-ONLY) THAY VÌ 1 FILE JSON DUY NHẤT
   - Mỗi shard ghi ra annotations_shard_XXX.jsonl, MỖI DÒNG là 1 object
     JSON cho 1 ảnh.
   - Ưu điểm so với việc load toàn bộ dict rồi json.dump() lại từ đầu:
     không phải rewrite toàn bộ file mỗi lần checkpoint (rất chậm khi
     file có hàng trăm nghìn record). Append + flush + fsync là đủ an toàn.

3. FILE PROGRESS RIÊNG (progress_shard_XXX.json)
   - Lưu đúng "vị trí trong dataset đã đi qua" (kể cả ảnh bị bỏ vì
     không có mặt), KHÔNG dùng số dòng trong file JSONL để tính resume
     (vì --skip-no-face có thể làm số dòng ít hơn số ảnh đã xử lý).
   - Khi restart, mỗi shard tự đọc file progress của nó và
     ds.skip(shard_start + consumed) để tiếp tục đúng chỗ.

4. PREFETCH THREAD TRONG MỖI SHARD
   - Một thread phụ liên tục lấy sample kế tiếp (I/O mạng: tải ảnh) và
     đẩy vào Queue.
   - Thread chính lấy từ Queue và chạy MediaPipe (CPU) trên ảnh hiện tại
     trong lúc thread phụ đã tải sẵn ảnh tiếp theo -> giảm thời gian chờ
     mạng xen giữa các lần detect.

5. TỰ ĐỘNG THỬ LẠI KHI LỖI MẠNG
   - Nếu iterator dataset ném exception (mất mạng, timeout...), shard sẽ:
     lưu progress hiện tại -> chờ (backoff tuyến tính) -> mở lại dataset
     từ đúng vị trí đã lưu -> tiếp tục. Tối đa --max-retries lần liên tiếp.

6. SUPERVISOR CHO TỪNG SHARD
   - Nếu cả tiến trình (không chỉ 1 lần detect) bị crash (OOM, segfault,
     lỗi driver...), supervisor sẽ tự spawn lại tiến trình đó. Tiến trình
     mới tự đọc progress file và resume, không mất dữ liệu.

CÀI ĐẶT:
    pip install datasets mediapipe opencv-python pillow

TẢI MODEL (nếu chưa có):
    wget -O face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

CHẠY VÍ DỤ (1 triệu ảnh, 8 tiến trình song song):
    python3 process_dataset_parallel.py \
        --output-dir /run/media/tranmanhduy/Data/DataPretrain \
        --model face_landmarker.task \
        --split train \
        --num-samples 1000000 \
        --start 0 \
        --num-workers 8 \
        --skip-no-face \
        --save-every 200

NẾU BỊ NGẮT (mất điện, Ctrl+C, crash...):
    Chạy lại ĐÚNG CÙNG LỆNH TRÊN. Mỗi shard tự đọc progress_shard_XXX.json
    và tiếp tục từ chỗ dở dang, không tải lại từ đầu.

GỘP KẾT QUẢ SAU KHI XONG:
    cat /run/media/tranmanhduy/Data/DataPretrain/annotations_shard_*.jsonl \
        > /run/media/tranmanhduy/Data/DataPretrain/annotations_all.jsonl

    (Không cần gộp thành 1 file JSON dict lớn - đọc trực tiếp JSONL bằng
    cách for line in file: json.loads(line) sẽ tiết kiệm RAM hơn nhiều
    khi dataset có 1 triệu ảnh.)

KẾT QUẢ:
    <output-dir>/Images/<file_name>.jpg              (ảnh gốc, dùng chung)
    <output-dir>/Images/drawn/<file_name>.jpg        (nếu --draw)
    <output-dir>/annotations_shard_000.jsonl ...     (mỗi shard 1 file)
    <output-dir>/progress_shard_000.json ...         (vị trí resume mỗi shard)
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback

# Tắt bớt log rác của TensorFlow Lite / absl / MediaPipe trước khi import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tham số dòng lệnh
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tải song song ảnh từ HuggingFace và trích xuất face landmarks bằng MediaPipe."
    )
    parser.add_argument("--dataset", type=str, default="wuji3/face-recognition")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output-dir", type=str,
        default="/run/media/tranmanhduy/Data/DataPretrain",
        help="Thư mục gốc để lưu."
    )
    parser.add_argument("--model", type=str, default="face_landmarker.task")
    parser.add_argument("--num-faces", type=int, default=1)
    parser.add_argument("--start", type=int, default=0,
                         help="Vị trí bắt đầu lấy mẫu trong dataset.")
    parser.add_argument("--num-samples", type=int, default=1_000_000,
                         help="Tổng số ảnh cần xử lý (chia đều cho các worker).")
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() - 1),
                         help="Số tiến trình song song (mặc định: số nhân CPU - 1).")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--skip-no-face", action="store_true")
    parser.add_argument("--save-every", type=int, default=200,
                         help="Flush + lưu progress sau mỗi N ảnh xử lý trong MỖI shard.")
    parser.add_argument("--prefetch-size", type=int, default=8,
                         help="Kích thước hàng đợi prefetch (số ảnh tải trước).")
    parser.add_argument("--max-retries", type=int, default=10,
                         help="Số lần thử lại liên tiếp khi lỗi mạng trong 1 shard trước khi bỏ shard đó.")
    parser.add_argument("--retry-delay", type=float, default=5.0,
                         help="Thời gian chờ cơ bản (giây) giữa các lần thử lại (tăng dần tuyến tính).")
    parser.add_argument("--max-process-restarts", type=int, default=20,
                         help="Số lần supervisor được phép khởi động lại 1 tiến trình shard nếu nó crash hẳn.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MediaPipe helpers (giống bản gốc)
# ---------------------------------------------------------------------------

def build_landmarker(model_path: str, num_faces: int):
    import mediapipe as mp_
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model '{model_path}'. Tải tại: "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

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
                "index": lm_idx, "x": float(landmark.x), "y": float(landmark.y), "z": float(landmark.z),
            })
            face_data["landmarks_pixel"].append({
                "index": lm_idx,
                "x": int(landmark.x * width),
                "y": int(landmark.y * height),
                "z": float(landmark.z * width),
            })

        x_norms = [float(lm.x) for lm in landmarks]
        y_norms = [float(lm.y) for lm in landmarks]
        xmin_norm, xmax_norm = min(x_norms), max(x_norms)
        ymin_norm, ymax_norm = min(y_norms), max(y_norms)

        face_data["bounding_box_normalized"] = {
            "xmin": xmin_norm, "ymin": ymin_norm, "xmax": xmax_norm, "ymax": ymax_norm,
            "width": xmax_norm - xmin_norm, "height": ymax_norm - ymin_norm,
        }

        xmin_px, ymin_px = int(xmin_norm * width), int(ymin_norm * height)
        xmax_px, ymax_px = int(xmax_norm * width), int(ymax_norm * height)
        face_data["bounding_box_pixel"] = {
            "xmin": xmin_px, "ymin": ymin_px, "xmax": xmax_px, "ymax": ymax_px,
            "width": xmax_px - xmin_px, "height": ymax_px - ymin_px,
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
    import mediapipe as mp_
    from mediapipe.framework.formats import landmark_pb2

    drawn_image = image_bgr.copy()
    mp_drawing = mp_.solutions.drawing_utils
    mp_drawing_styles = mp_.solutions.drawing_styles
    mp_face_mesh = mp_.solutions.face_mesh

    for face_idx, face_landmarks in enumerate(result.face_landmarks):
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in face_landmarks
        ])
        mp_drawing.draw_landmarks(
            image=drawn_image, landmark_list=proto,
            connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=drawn_image, landmark_list=proto,
            connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp_drawing.draw_landmarks(
            image=drawn_image, landmark_list=proto,
            connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

        h, w = image_bgr.shape[:2]
        xs = [lm.x * w for lm in face_landmarks]
        ys = [lm.y * h for lm in face_landmarks]
        x_min_px, x_max_px = int(min(xs)), int(max(xs))
        y_min_px, y_max_px = int(min(ys)), int(max(ys))
        cv2.rectangle(drawn_image, (x_min_px - 10, y_min_px - 10), (x_max_px + 10, y_max_px + 10), (0, 200, 255), 2)
        cv2.putText(drawn_image, f"Face #{face_idx}", (x_min_px - 10, max(y_min_px - 15, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    return drawn_image


# ---------------------------------------------------------------------------
# Progress (resume) helpers
# ---------------------------------------------------------------------------

def load_progress(progress_path: str) -> int:
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return int(json.load(f).get("consumed", 0))
        except Exception:
            return 0
    return 0


def save_progress(progress_path: str, consumed: int):
    tmp = progress_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"consumed": consumed}, f)
    os.replace(tmp, progress_path)


# ---------------------------------------------------------------------------
# Prefetch thread: tải sample kế tiếp trong lúc thread chính đang detect
# ---------------------------------------------------------------------------

def prefetch_thread_fn(ds_iter, out_queue: "queue.Queue", stop_event: threading.Event, errors: list):
    try:
        for sample in ds_iter:
            if stop_event.is_set():
                return
            out_queue.put(sample)
    except Exception as e:  # lỗi mạng / lỗi dataset sẽ được xử lý ở thread chính
        errors.append(e)
    finally:
        out_queue.put(None)  # sentinel: hết dữ liệu hoặc có lỗi


# ---------------------------------------------------------------------------
# Worker xử lý 1 shard (chạy trong 1 tiến trình riêng)
# ---------------------------------------------------------------------------

def shard_worker(shard_id: int, cfg: dict):
    from datasets import load_dataset
    import mediapipe as mp_

    shard_start = cfg["shard_start"]
    shard_len = cfg["shard_len"]
    output_dir = cfg["output_dir"]

    images_dir = os.path.join(output_dir, "Images")
    drawn_dir = os.path.join(images_dir, "drawn")
    os.makedirs(images_dir, exist_ok=True)
    if cfg["draw"]:
        os.makedirs(drawn_dir, exist_ok=True)

    ann_path = os.path.join(output_dir, f"annotations_shard_{shard_id:03d}.jsonl")
    progress_path = os.path.join(output_dir, f"progress_shard_{shard_id:03d}.json")

    consumed = load_progress(progress_path)
    if consumed >= shard_len:
        print(f"[Shard {shard_id}] Đã hoàn tất từ trước ({consumed}/{shard_len}). Bỏ qua.")
        return

    print(f"[Shard {shard_id}] Bắt đầu/tiếp tục từ {consumed}/{shard_len} "
          f"(offset dataset = {shard_start + consumed})")

    landmarker = build_landmarker(cfg["model"], cfg["num_faces"])
    ann_file = open(ann_path, "a", encoding="utf-8")

    processed_since_save = 0
    detected = 0
    skipped = 0
    consecutive_failures = 0
    cur_offset = shard_start + consumed

    def open_iter(offset):
        ds = load_dataset(cfg["dataset"], split=cfg["split"], streaming=True)
        remain = shard_len - (offset - shard_start)
        return iter(ds.skip(offset).take(remain))

    try:
        while consumed < shard_len:
            try:
                data_queue: "queue.Queue" = queue.Queue(maxsize=cfg["prefetch_size"])
                stop_event = threading.Event()
                prefetch_errors: list = []
                t = threading.Thread(
                    target=prefetch_thread_fn,
                    args=(open_iter(cur_offset), data_queue, stop_event, prefetch_errors),
                    daemon=True,
                )
                t.start()

                while True:
                    sample = data_queue.get()
                    if sample is None:
                        if prefetch_errors:
                            raise prefetch_errors[0]
                        break  # hết shard bình thường

                    file_name = sample.get("file_name") or f"image_{cur_offset:08d}.jpg"
                    label = sample.get("label")
                    class_name = sample.get("class_name")
                    pil_image = sample["image"]
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")

                    image_path = os.path.join(images_dir, file_name)
                    pil_image.save(image_path)

                    rgb_np = np.array(pil_image)
                    width, height = pil_image.width, pil_image.height
                    mp_image = mp_.Image(image_format=mp_.ImageFormat.SRGB, data=rgb_np)
                    result = landmarker.detect(mp_image)
                    faces_data = extract_face_data(result, width, height)

                    if not faces_data and cfg["skip_no_face"]:
                        os.remove(image_path)
                        skipped += 1
                    else:
                        record = {
                            "file_name": file_name, "width": width, "height": height,
                            "label": label, "class_name": class_name, "faces": faces_data,
                        }
                        ann_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if faces_data:
                            detected += 1
                            if cfg["draw"]:
                                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                                drawn = draw_landmarks_on_image(bgr_np, result)
                                cv2.imwrite(os.path.join(drawn_dir, file_name), drawn)

                    consumed += 1
                    cur_offset += 1
                    processed_since_save += 1
                    consecutive_failures = 0

                    if processed_since_save >= cfg["save_every"]:
                        ann_file.flush()
                        os.fsync(ann_file.fileno())
                        save_progress(progress_path, consumed)
                        processed_since_save = 0
                        print(f"[Shard {shard_id}] {consumed}/{shard_len} "
                              f"(mặt: {detected}, bỏ qua: {skipped})")

                break  # hết dữ liệu shard bình thường

            except KeyboardInterrupt:
                raise
            except Exception as e:
                consecutive_failures += 1
                print(f"[Shard {shard_id}] Lỗi ({consecutive_failures}/{cfg['max_retries']}): {e}")
                traceback.print_exc()
                ann_file.flush()
                save_progress(progress_path, consumed)
                if consecutive_failures >= cfg["max_retries"]:
                    print(f"[Shard {shard_id}] Vượt quá số lần thử lại. Dừng shard này "
                          f"(có thể chạy lại lệnh gốc sau để tiếp tục).")
                    break
                time.sleep(cfg["retry_delay"] * consecutive_failures)
                continue
    except KeyboardInterrupt:
        print(f"[Shard {shard_id}] Ngắt bởi người dùng, đang lưu checkpoint...")
    finally:
        ann_file.flush()
        try:
            os.fsync(ann_file.fileno())
        except Exception:
            pass
        ann_file.close()
        save_progress(progress_path, consumed)
        landmarker.close()

    print(f"[Shard {shard_id}] Kết thúc: {consumed}/{shard_len} "
          f"(mặt: {detected}, bỏ qua: {skipped})")


# ---------------------------------------------------------------------------
# Supervisor: tự khởi động lại tiến trình shard nếu nó crash hẳn
# ---------------------------------------------------------------------------

def run_with_supervisor(shard_id: int, cfg: dict, max_restarts: int):
    restarts = 0
    while restarts <= max_restarts:
        p = mp.Process(target=shard_worker, args=(shard_id, cfg))
        p.start()
        p.join()
        if p.exitcode == 0:
            return
        restarts += 1
        print(f"[Supervisor] Shard {shard_id} tiến trình dừng bất thường "
              f"(exitcode={p.exitcode}). Khởi động lại lần {restarts}/{max_restarts}...")
        time.sleep(5)
    print(f"[Supervisor] Shard {shard_id} vượt quá số lần khởi động lại cho phép. Bỏ cuộc.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Thiếu thư viện 'datasets'. Cài đặt bằng: pip install datasets")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    n_workers = max(1, args.num_workers)
    total = args.num_samples
    base_len = total // n_workers
    remainder = total % n_workers

    shards = []
    offset = args.start
    for i in range(n_workers):
        length = base_len + (1 if i < remainder else 0)
        if length > 0:
            shards.append((offset, length))
        offset += length

    print(f"Tổng số ảnh: {total} | Số worker: {len(shards)} | "
          f"~{total // max(1, len(shards))} ảnh/worker")

    common_cfg = dict(
        output_dir=args.output_dir,
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        draw=args.draw,
        skip_no_face=args.skip_no_face,
        save_every=args.save_every,
        num_faces=args.num_faces,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        prefetch_size=args.prefetch_size,
    )

    processes = []
    for i, (shard_start, shard_len) in enumerate(shards):
        cfg = dict(common_cfg, shard_start=shard_start, shard_len=shard_len)
        p = mp.Process(target=run_with_supervisor, args=(i, cfg, args.max_process_restarts))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nNhận Ctrl+C. Đang dừng tất cả worker, vui lòng đợi checkpoint được lưu...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()

    print("\n=== HOÀN TẤT (hoặc đã dừng an toàn) ===")
    print(f"Ảnh lưu tại : {os.path.join(args.output_dir, 'Images')}")
    print(f"Annotations : {os.path.join(args.output_dir, 'annotations_shard_*.jsonl')}")
    print("Nếu bị ngắt giữa chừng, chạy lại CHÍNH XÁC lệnh này để tiếp tục từ chỗ dở dang.")
    print("Gộp tất cả shard thành 1 file JSONL:")
    print(f"  cat {os.path.join(args.output_dir, 'annotations_shard_*.jsonl')} "
          f"> {os.path.join(args.output_dir, 'annotations_all.jsonl')}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()