"""
check_lmk_margin_coverage.py
==============================
Công cụ kiểm tra RIÊNG (không đụng vào head/loss/dataset), trả lời câu
hỏi quan trọng nhất trước khi train dài hạn với thiết kế v2/v3:

    "bounding_box_normalized" trong data có thực sự bao trọn landmark
    không, và margin=0.15 (mặc định) có đủ hay không?

LÝ DO CẦN SCRIPT NÀY:
------------------------------------------------------------------------
DetectHeadFaceLmk decode landmark = box_giãn_margin -> sigmoid trong
[0,1]. FaceLandmarkDetectionLoss encode target landmark theo đúng công
thức ngược lại rồi CLAMP vào [0,1]. Nếu "bounding_box_normalized" trong
JSONL là box từ 1 face-detector riêng (không phải bao chính xác min/max
của 478 điểm mesh), landmark ở viền hàm/trán/tai rất dễ rơi ra ngoài
vùng box+margin -> bị clamp về đúng biên 0 hoặc 1 -> model học "kẹp sát
biên" thay vì vị trí thật, mất độ phân giải cho đúng những điểm đó.

Script này KHÔNG cần model / head / loss - chỉ đọc thẳng annotations_all
.jsonl, tính tỉ lệ điểm landmark rơi ngoài box+margin cho vài giá trị
margin khác nhau, in ra bảng % để bạn chọn margin phù hợp CHO
FaceLmkConfig.lmk_margin trước khi train.

CHẠY:
    python3 check_lmk_margin_coverage.py --root-dir /duong/dan/DataPretrain

    (tuỳ chọn) --sample-size 5000   : số ảnh lấy mẫu (mặc định 5000, đủ
                                       để ước lượng ổn định, không cần
                                       quét hết dataset lớn)
    (tuỳ chọn) --margins 0.0 0.05 0.10 0.15 0.20 0.30
"""

import argparse
import json
import os
import random

import numpy as np


def _iter_sampled_records(jsonl_path: str, sample_size: int, seed: int = 0):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.Random(seed).shuffle(lines)
    for line in lines[:sample_size]:
        line = line.strip()
        if line:
            yield json.loads(line)


def check_coverage(root_dir: str, jsonl_name: str = "annotations_all.jsonl",
                    sample_size: int = 5000, margins=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30)):
    jsonl_path = os.path.join(root_dir, jsonl_name)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Không tìm thấy {jsonl_path}")

    # với mỗi margin, đếm số điểm landmark nằm NGOÀI vùng box+margin
    # (tính trong không gian normalized [0,1] gốc, không cần resize ảnh)
    n_points_total = 0
    n_outside = {m: 0 for m in margins}
    # đồng thời đo % OVERFLOW tối đa cần thiết (bao nhiêu margin mới đủ
    # chứa 99% / 99.9% điểm) để gợi ý margin "vừa đủ" thay vì chỉ test
    # vài giá trị cố định.
    overflow_fracs = []  # % w hoặc h cần giãn thêm, cho mỗi điểm ngoài box gốc (margin=0)

    n_faces_checked = 0
    for record in _iter_sampled_records(jsonl_path, sample_size):
        for face in record.get("faces", []):
            bb = face["bounding_box_normalized"]
            x1, y1, x2, y2 = bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]
            w, h = (x2 - x1), (y2 - y1)
            if w <= 0 or h <= 0:
                continue
            n_faces_checked += 1

            for p in face["landmarks_normalized"]:
                n_points_total += 1
                px, py = p["x"], p["y"]

                # khoảng cách (theo tỉ lệ w/h) điểm vượt ra ngoài box GỐC
                # (margin=0), âm nghĩa là đã nằm trong box.
                over_x = max((x1 - px) / w, (px - x2) / w, 0.0)
                over_y = max((y1 - py) / h, (py - y2) / h, 0.0)
                overflow_fracs.append(max(over_x, over_y))

                for m in margins:
                    x1e, x2e = x1 - m * w, x2 + m * w
                    y1e, y2e = y1 - m * h, y2 + m * h
                    if not (x1e <= px <= x2e and y1e <= py <= y2e):
                        n_outside[m] += 1

    if n_points_total == 0:
        print("[check_lmk_margin_coverage] Không tìm thấy landmark nào trong mẫu đã lấy.")
        return

    overflow_arr = np.array(overflow_fracs)
    print(f"Số mặt kiểm tra: {n_faces_checked} | số điểm landmark: {n_points_total}\n")

    print("Tỉ lệ điểm landmark NẰM NGOÀI box+margin (sẽ bị clamp trong loss):")
    for m in margins:
        pct = 100.0 * n_outside[m] / n_points_total
        print(f"  margin={m:>5.2f}  ->  {pct:6.2f}% điểm bị clamp")

    print("\nGợi ý margin để chứa X% điểm (dựa trên phân vị overflow thực tế):")
    for pct_target in (95, 99, 99.9, 100):
        needed = np.percentile(overflow_arr, pct_target if pct_target < 100 else 100)
        print(f"  chứa {pct_target:>5}% điểm  ->  cần margin >= {needed:.3f}")

    print(
        "\nLưu ý: margin quá lớn (vd cần >0.5 để chứa 99.9%) có thể do vài "
        "outlier annotation hiếm gặp (bbox lệch hẳn khỏi mesh) - nên nhìn "
        "cả 95%/99% lẫn 99.9% thay vì chỉ chọn theo giá trị max tuyệt đối, "
        "và cân nhắc lọc bỏ hẳn các face có overflow bất thường thay vì "
        "kéo margin lên quá cao cho toàn bộ dataset."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--jsonl-name", type=str, default="annotations_all.jsonl")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--margins", type=float, nargs="+",
                         default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    args = parser.parse_args()

    check_coverage(
        args.root_dir, args.jsonl_name, args.sample_size, tuple(args.margins)
    )


if __name__ == "__main__":
    main()
