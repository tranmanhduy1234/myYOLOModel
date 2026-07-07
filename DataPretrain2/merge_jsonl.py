"""
merge_jsonl.py
===============

Tiện ích nhỏ để làm việc với các file annotations_shard_*.jsonl sinh ra từ
process_dataset_parallel.py.

1) Gộp tất cả shard thành 1 file JSONL duy nhất (khuyên dùng, tốn ít RAM):
    python3 merge_jsonl.py merge --output-dir /run/media/tranmanhduy/Data/DataPretrain

2) Kiểm tra nhanh: đếm tổng số ảnh đã có annotation trong tất cả shard:
    python3 merge_jsonl.py count --output-dir /run/media/tranmanhduy/Data/DataPretrain

3) (Chỉ nếu thực sự cần 1 dict JSON lớn, ví dụ code cũ yêu cầu) chuyển
   sang 1 file .json duy nhất - CHÚ Ý: với 1 triệu ảnh file này sẽ rất lớn
   và tốn nhiều RAM khi load lại, chỉ dùng khi bắt buộc:
    python3 merge_jsonl.py to-json --output-dir /run/media/tranmanhduy/Data/DataPretrain
"""

import argparse
import glob
import json
import os


def iter_shard_files(output_dir):
    return sorted(glob.glob(os.path.join(output_dir, "annotations_shard_*.jsonl")))


def cmd_merge(output_dir):
    files = iter_shard_files(output_dir)
    out_path = os.path.join(output_dir, "annotations_all.jsonl")
    total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for fp in files:
            with open(fp, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")
                        total += 1
    print(f"Đã gộp {len(files)} shard -> {out_path} ({total} ảnh)")


def cmd_count(output_dir):
    files = iter_shard_files(output_dir)
    total = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        print(f"  {os.path.basename(fp)}: {n} ảnh")
        total += n
    print(f"Tổng: {total} ảnh trong {len(files)} shard")


def cmd_to_json(output_dir):
    files = iter_shard_files(output_dir)
    merged = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                merged[record["file_name"]] = record
    out_path = os.path.join(output_dir, "annotations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {len(merged)} ảnh -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["merge", "count", "to-json"])
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    if args.action == "merge":
        cmd_merge(args.output_dir)
    elif args.action == "count":
        cmd_count(args.output_dir)
    elif args.action == "to-json":
        cmd_to_json(args.output_dir)

if __name__ == "__main__":
    main()