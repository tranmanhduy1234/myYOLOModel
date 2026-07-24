"""
run_all_validation.py
======================
Điểm vào DUY NHẤT của bộ công cụ validate cho toàn bộ NMSFreeDetector.
Chạy tuần tự TẤT CẢ các validator con và gộp lại thành 1 báo cáo tổng hợp:

    validate_loss.py       - bbox_iou/CIoU, TAL assigner, BboxLoss, DetectionLoss
    validate_model.py      - blocks.py, backbone_neck.py, head.py, model.py
    validate_pipeline.py   - ema.py, engine.py (optimizer/LR/checkpoint), config.py
    validate_dataloader.py - letterbox/collate_fn/DetectionAugmenter (không cần data thật)

Cấu trúc thư mục giả định (đặt tất cả file validate_*.py CÙNG
1 cấp thư mục, có thể chạy từ bất kỳ đâu vì repo dùng "src." làm gốc import
tuyệt đối):

    <project_root>/
        src/
            model.py, backbone_neck.py, blocks.py, head.py, config.py
            train/
                loss.py, ema.py, engine.py, dataloader1_obj365.py
            validation_tool/
                validate_common.py
                validate_loss.py
                validate_model.py
                validate_pipeline.py
                validate_dataloader.py
                run_all_validation.py       <- file này

Chạy:
    python -m src.validation_tool.run_all_validation
    python -m src.validation_tool.run_all_validation --device cuda
    python -m src.validation_tool.run_all_validation --skip loss,dataloader   # bỏ qua 1 vài nhóm
    python -m src.validation_tool.run_all_validation --verbose-traceback       # in full traceback khi ERROR

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ
------------------------------
1. run_loss_suite(): không còn monkey-patching biến global R/VERBOSE_TRACEBACK
   nữa. validate_loss.py đã được refactor để hàm run() nhận Reporter qua
   tham số → an toàn khi refactor, không phụ thuộc tên biến global.
2. In thống kê từng suite (PASS/FAIL/ERROR/SKIP) ngay sau mỗi suite chạy xong.
3. In thời gian từng suite riêng lẻ bên cạnh tổng thời gian.
"""

import argparse
import sys
import time

from src.validation_tool.validate_common import Reporter, get_device

SUITES = ["loss", "model", "pipeline", "dataloader"]


# ==============================================================================
# Hàm chạy từng suite
# ==============================================================================

def run_loss_suite(device: str, verbose_traceback: bool) -> Reporter:
    """Chạy suite LOSS (validate_loss.py).

    KHÔNG còn monkey-patching biến global. validate_loss.run() nhận
    verbose_traceback và trả về Reporter đã điền đầy đủ kết quả.
    """
    from src.validation_tool import validate_loss
    return validate_loss.run(device, verbose_traceback)


def run_model_suite(device: str, verbose_traceback: bool) -> Reporter:
    from src.validation_tool import validate_model
    return validate_model.run(device, verbose_traceback)


def run_pipeline_suite(device: str, verbose_traceback: bool) -> Reporter:
    from src.validation_tool import validate_pipeline
    return validate_pipeline.run(device, verbose_traceback)


def run_dataloader_suite(verbose_traceback: bool) -> Reporter:
    from src.validation_tool import validate_dataloader
    return validate_dataloader.run(verbose_traceback)


# ==============================================================================
# Helper in tóm tắt nhanh sau mỗi suite
# ==============================================================================

def _print_suite_summary(name: str, r: Reporter, elapsed: float):
    p = r.count("PASS")
    f = r.count("FAIL")
    e = r.count("ERROR")
    s = r.count("SKIP")
    total = len(r.results)
    status = "✓ OK" if (f == 0 and e == 0) else "✗ CÓ LỖI"
    print(
        f"\n  [{name}] {status} — "
        f"{p}/{total} PASS, {f} FAIL, {e} ERROR, {s} SKIP "
        f"({elapsed:.1f}s)"
    )


# ==============================================================================
# main()
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chạy toàn bộ bộ công cụ validate cho NMSFreeDetector."
    )
    parser.add_argument("--device", type=str, default=None,
                        help="cuda|cpu (mặc định: tự động phát hiện)")
    parser.add_argument("--skip", type=str, default="",
                        help=f"danh sách suite bỏ qua, cách nhau bởi dấu phẩy. Các suite: {SUITES}")
    parser.add_argument("--verbose-traceback", action="store_true",
                        help="In full traceback khi gặp ERROR")
    args = parser.parse_args()

    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown  = skip_set - set(SUITES)
    if unknown:
        print(f"[Cảnh báo] --skip có tên suite không hợp lệ, sẽ bị bỏ qua: {unknown}")

    device = get_device(args.device)
    print(f"{'#' * 78}\nBỘ CÔNG CỤ VALIDATE - NMSFreeDetector\nThiết bị: {device}\n{'#' * 78}")

    master = Reporter(args.verbose_traceback)
    t_total = time.time()
    suite_times = {}

    # ------------------------------------------------------------------ LOSS --
    suite = "loss"
    if suite not in skip_set:
        print(f"\n>>> Suite: LOSS (validate_loss.py) <<<")
        t0 = time.time()
        r  = run_loss_suite(device, args.verbose_traceback)
        elapsed = time.time() - t0
        master.merge(r)
        suite_times[suite] = elapsed
        _print_suite_summary("LOSS", r, elapsed)
    else:
        print(f"\n>>> Suite: LOSS - BỎ QUA (--skip) <<<")

    # ----------------------------------------------------------------- MODEL --
    suite = "model"
    if suite not in skip_set:
        print(f"\n>>> Suite: MODEL ARCHITECTURE (validate_model.py) <<<")
        t0 = time.time()
        r  = run_model_suite(device, args.verbose_traceback)
        elapsed = time.time() - t0
        master.merge(r)
        suite_times[suite] = elapsed
        _print_suite_summary("MODEL", r, elapsed)
    else:
        print(f"\n>>> Suite: MODEL ARCHITECTURE - BỎ QUA (--skip) <<<")

    # -------------------------------------------------------------- PIPELINE --
    suite = "pipeline"
    if suite not in skip_set:
        print(f"\n>>> Suite: TRAINING PIPELINE (validate_pipeline.py) <<<")
        t0 = time.time()
        r  = run_pipeline_suite(device, args.verbose_traceback)
        elapsed = time.time() - t0
        master.merge(r)
        suite_times[suite] = elapsed
        _print_suite_summary("PIPELINE", r, elapsed)
    else:
        print(f"\n>>> Suite: TRAINING PIPELINE - BỎ QUA (--skip) <<<")

    # ------------------------------------------------------------ DATALOADER --
    suite = "dataloader"
    if suite not in skip_set:
        print(f"\n>>> Suite: DATALOADER UTILS (validate_dataloader.py) <<<")
        t0 = time.time()
        r  = run_dataloader_suite(args.verbose_traceback)
        elapsed = time.time() - t0
        master.merge(r)
        suite_times[suite] = elapsed
        _print_suite_summary("DATALOADER", r, elapsed)
    else:
        print(f"\n>>> Suite: DATALOADER UTILS - BỎ QUA (--skip) <<<")

    # ---------------------------------------------------------- TỔNG KẾT -----
    total_elapsed = time.time() - t_total
    ok = master.summary("TỔNG KẾT TOÀN BỘ BỘ CÔNG CỤ VALIDATE")

    print("Thời gian từng suite:")
    for name, t in suite_times.items():
        print(f"  {name:<12}: {t:.1f}s")
    print(f"Tổng thời gian  : {total_elapsed:.1f}s")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()