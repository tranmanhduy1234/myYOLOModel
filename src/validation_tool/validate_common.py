"""
validate_common.py
===================
Thành phần dùng chung cho toàn bộ "bộ công cụ validate" của NMSFreeDetector:

  - Reporter : chạy từng test-case riêng lẻ, bắt exception, in PASS/FAIL/ERROR/SKIP,
    tổng kết cuối cùng. Mọi validator trong repo đều dùng chung lớp này để đảm
    bảo định dạng output nhất quán (kể cả validate_loss.py đã được refactor).

  - Skip / skip() : đánh dấu một test là "bỏ qua" (thiếu dependency tùy chọn
    như cv2, albumentations, cuda) thay vì tính là FAIL.

  - get_device() : phát hiện thiết bị tự động (cuda/cpu).

Mọi file validate_*.py trong bộ công cụ import Reporter từ đây,
KHÔNG tự định nghĩa lại để tránh trùng lặp code và đảm bảo đồng bộ.
"""

import sys
import traceback


class Skip(Exception):
    """Ném exception này bên trong một test-case để Reporter ghi nhận là SKIP
    (khác với FAIL) — dùng khi thiếu dependency tùy chọn (cv2, albumentations,
    cuda...) hoặc thiếu dữ liệu thật (dataset thật trên đĩa)."""
    pass


def skip(msg: str):
    """Shorthand để raise Skip từ bên trong lambda hoặc test function."""
    raise Skip(msg)


class Reporter:
    """Thu thập kết quả test và in tổng kết cuối phiên.

    Mỗi test-case được đăng ký qua `check(section, name, fn)`.
    Kết quả lưu dưới dạng tuple (section, name, status, detail) với
    status ∈ {"PASS", "FAIL", "ERROR", "SKIP"}.
    """

    def __init__(self, verbose_traceback: bool = False):
        self.results = []          # list[(section, name, status, detail)]
        self.verbose_traceback = verbose_traceback

    # ------------------------------------------------------------------
    # Chạy một test-case
    # ------------------------------------------------------------------
    def check(self, section: str, name: str, fn):
        """Chạy fn(), bắt exception và phân loại kết quả."""
        try:
            detail = fn()
            self.results.append((section, name, "PASS", detail or ""))
            print(f"  [PASS] {name}" + (f" -> {detail}" if detail else ""))
        except Skip as e:
            self.results.append((section, name, "SKIP", str(e)))
            print(f"  [SKIP] {name} -> {e}")
        except AssertionError as e:
            self.results.append((section, name, "FAIL", str(e)))
            print(f"  [FAIL] {name} -> AssertionError: {e}")
        except Exception as e:
            self.results.append((section, name, "ERROR", f"{type(e).__name__}: {e}"))
            print(f"  [ERROR] {name} -> {type(e).__name__}: {e}")
            if self.verbose_traceback:
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Tiện ích trình bày
    # ------------------------------------------------------------------
    def section(self, title: str):
        """In tiêu đề phần."""
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    # ------------------------------------------------------------------
    # Gộp kết quả
    # ------------------------------------------------------------------
    def merge(self, other: "Reporter"):
        """Gộp kết quả từ một Reporter khác (dùng trong run_all_validation.py)."""
        self.results.extend(other.results)

    # ------------------------------------------------------------------
    # Thống kê nhanh
    # ------------------------------------------------------------------
    def count(self, status: str) -> int:
        """Trả về số test có status cho trước."""
        return sum(1 for r in self.results if r[2] == status)

    # ------------------------------------------------------------------
    # Tổng kết
    # ------------------------------------------------------------------
    def summary(self, title: str = "TỔNG KẾT") -> bool:
        """In bảng tổng kết và trả về True nếu không có FAIL/ERROR."""
        total   = len(self.results)
        passed  = self.count("PASS")
        failed  = self.count("FAIL")
        errored = self.count("ERROR")
        skipped = self.count("SKIP")

        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print(f"Tổng số kiểm tra   : {total}")
        print(f"  Đạt (PASS)       : {passed}")
        print(f"  Không đạt (FAIL) : {failed}")
        print(f"  Lỗi (ERROR)      : {errored}")
        print(f"  Bỏ qua (SKIP)    : {skipped}")

        if failed or errored:
            print("\nDanh sách KHÔNG ĐẠT / LỖI:")
            for section, name, status, detail in self.results:
                if status in ("FAIL", "ERROR"):
                    print(f"  - [{status}][{section}] {name}: {detail}")

        if skipped:
            print("\nDanh sách BỎ QUA:")
            for section, name, status, detail in self.results:
                if status == "SKIP":
                    print(f"  - [{section}] {name}: {detail}")

        print()
        return failed == 0 and errored == 0


def get_device(preferred: str | None = None) -> str:
    """Trả về thiết bị ưu tiên; nếu không có CUDA thì fallback về 'cpu'."""
    import torch
    if preferred is not None:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"
