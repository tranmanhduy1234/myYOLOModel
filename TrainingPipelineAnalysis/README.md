# BÁO CÁO NGHIÊN CỨU KĨ THUẬT VỀ TOÀN BỘ PIPELINE HUẤN LUYỆN NMS-FREE OBJECT DETECTOR (YOLOv10-STYLE)

## TỔNG QUAN VỀ BỘ TÀI LIỆU HỌC THUẬT

Bộ tài liệu này được biên soạn theo tiêu chuẩn cao nhất của một chương luận văn tốt nghiệp cao học / báo cáo nghiên cứu kỹ thuật chuyên sâu (Deep Learning Technical Research Report). Nội dung tập trung phân tích toàn diện, toàn thể và chuyên sâu về **Pipeline Huấn luyện (Training Pipeline)** của kiến trúc mô hình phát hiện vật thể không sử dụng NMS (**NMS-Free Detector**) được cài đặt trong mã nguồn project.

Bộ báo cáo bao gồm **11 chương chuyên đề độc lập nhưng liên kết chặt chẽ**, bao phủ toàn bộ chuỗi xử lý từ dữ liệu đầu vào, khởi tạo trọng số toán học, thuật toán tối ưu, cơ chế suy giảm tốc độ học, cấu trúc hàm mất mát kép (Dual-Head Loss), hệ thống theo dõi & giám sát (Logging System), cơ chế quản lý checkpoint cho đến các đánh giá kỹ thuật chuyên sâu.

---

## MỤC LỤC CHI TIẾT CÁC CHƯƠNG

| Chương | Tên Chương | Nội Dung Trọng Tâm |
| :--- | :--- | :--- |
| **01** | [Kiến trúc & Vòng đời Training Pipeline](01_Training_Pipeline.md) | Vòng đời 15 bước từ Dataset $\rightarrow$ DataLoader $\rightarrow$ Collate $\rightarrow$ Forward $\rightarrow$ Loss $\rightarrow$ AMP Backward $\rightarrow$ Clipping $\rightarrow$ Optimizer $\rightarrow$ Scheduler $\rightarrow$ EMA $\rightarrow$ Validation $\rightarrow$ Checkpoint. |
| **02** | [Data Pipeline & Tối ưu hóa Bộ nhớ](02_Data_Pipeline.md) | Cơ chế Byte-Offset Indexing qua JSONL, Thuật toán Letterbox Padding, Albumentations Augmentations, PyTorch Multi-processing & Zero-RAM-leak Dataloading. |
| **03** | [Lý thuyết Toán học & Khởi tạo Trọng số](03_Weight_Initialization.md) | Quy trình 3 Pha (3-Phase Init), Toán học Kaiming Normal (fan_out), Constant BN, DFL Frozen Weight, Focal Prior Logit Initialization & Stride-Aware Class Bias Scaling. |
| **04** | [Thuật toán Tối ưu & Động học Cập nhật Trọng số](04_Optimizer.md) | Phân tách Parameter Grouping (Decay vs No-Decay), Toán học AdamW vs SGD, Decoupled Weight Decay, Gradient Clipping & Chống bùng nổ Gradient. |
| **05** | [Chiến lược Suy giảm Tốc độ Học (LR Scheduler)](05_Learning_Rate_Scheduler.md) | Toán học Linear Warmup, Cosine Annealing Decay, Cập nhật theo Iteration (Per-step Decay), Tương tác với AMP Skips. |
| **06** | [Cấu trúc Hàm Mất mát & Động học Không gian Tọa độ](06_Loss_Function.md) | Dual-Head Architecture (o2m vs o2o), Task-Aligned Assigner (TAL) Alignment Metric $m = s^\alpha \cdot \text{IoU}^\beta$, Chuyển đổi Không gian [PIXEL] $\leftrightarrow$ [GRID], CIoU Loss, Distribution Focal Loss (DFL). |
| **07** | [Hệ thống Logging Multi-Tier & Real-Time Monitoring](07_Logging_System.md) | Dual Logger (File Log vs TensorBoard SummaryWriter), Quản lý Tần suất (Scalar Interval vs Histogram Interval), RMSNorm Gradient/Weight, Update-to-Weight Ratio, Memory Profiling. |
| **08** | [Cơ chế Quản lý Checkpoint & Resume Training](08_Checkpoint_System.md) | Cấu trúc Checkpoint 7 Thành phần, Cơ chế Phôi tái lập State, Trích xuất Trunk State-Dict (`best_trunk.pt`) cho Transfer Learning, Atomic State Saving. |
| **09** | [Hệ sinh thái Công nghệ & Tăng tốc Phần cứng](09_Frameworks_and_Technologies.md) | Phân tích PyTorch Autograd, Mixed Precision AMP (`torch.autocast`, `GradScaler`), OpenCV / Albumentations, CUDA Memory Pining, Thread Synchronization. |
| **10** | [Đánh giá Kỹ thuật, Bottlenecks & Trade-offs](10_Discussion.md) | Thảo luận chuyên sâu về đánh đổi Compute vs Convergence Rate, Ảnh hưởng của Batch Size & Stride, Điểm nghẽn I/O & Memory Overhead, Tính ổn định khi Train NMS-Free Head. |
| **11** | [Kết luận & Định hướng Phát triển](11_Conclusion.md) | Tổng kết toàn bộ khám phá kỹ thuật, Đóng góp của kiến trúc Pipeline, Các hướng cải tiến nâng cao cho bài toán thương mại và nghiên cứu sinh. |

---

## CẤU TRÚC THƯ MỤC BÁO CÁO

```text
TrainingPipelineAnalysis/
├── README.md                           # Master Table of Contents & Executive Overview
├── 01_Training_Pipeline.md             # Chương 1: Architecture of the End-to-End Training Pipeline
├── 02_Data_Pipeline.md                 # Chương 2: Data Pipeline, Augmentations, & Memory Optimization
├── 03_Weight_Initialization.md         # Chương 3: Mathematical Theory & Practice of Weight Initialization
├── 04_Optimizer.md                     # Chương 4: Optimization Algorithms & Parameter Update Dynamics
├── 05_Learning_Rate_Scheduler.md       # Chương 5: Learning Rate Scheduling & Warmup Strategies
├── 06_Loss_Function.md                 # Chương 6: Mathematical Formulation & Coordinate Dynamics of Loss Functions
├── 07_Logging_System.md                # Chương 7: Multi-Tier Logging System & Real-Time Monitoring
├── 08_Checkpoint_System.md             # Chương 8: State Persistence, Resumability, & Trunk Extraction
├── 09_Frameworks_and_Technologies.md   # Chương 9: Technology Stack, Hardware Acceleration, & AMP
├── 10_Discussion.md                    # Chương 10: Technical Discussion, Performance Analysis, & Trade-Offs
├── 11_Conclusion.md                    # Chương 11: Summary & Future Research Directions
│
├── diagrams/                           # Sơ đồ Mermaid & Kiến trúc Pipeline
│   ├── pipeline_flow.mermaid
│   ├── coordinate_transformation.mermaid
│   └── loss_structure.mermaid
├── figures/                            # Biểu đồ minh họa ASCII & Mô phỏng trực quan
├── formulas/                           # Tổng hợp Công thức Toán học theo chuẩn LaTeX
│   ├── loss_mathematics.md
│   └── weight_init_mathematics.md
└── assets/                             # Các tệp tài nguyên phụ trợ
```

---

## NGUYÊN TẮC PHÂN TÍCH HỌC THUẬT

1. **Giữ nguyên tuyệt đối mã nguồn dự án**: Không chỉnh sửa, không refactor, không xóa/sửa bất kỳ file mã nguồn nào của project. Tất cả tài liệu phân tích được tạo độc lập trong thư mục `TrainingPipelineAnalysis/`.
2. **Chứng minh Toán học & Logic Thiết kế**: Mọi khẳng định về hiệu năng, tốc độ hội tụ hoặc mức độ ổn định đều được hỗ trợ bởi công thức toán học (Variance preservation, Gradient scaling, Softmax interpolation) và đối chiếu trực tiếp với mã nguồn cài đặt (`src/train/engine.py`, `src/train/loss.py`, `src/utils/init_weights.py`, v.v.).
3. **Liên kết không gian tọa độ**: Phân tích chi tiết ranh giới giữa hai không gian tọa độ then chốt: **[PIXEL]** (dùng cho Task-Aligned Assigner) và **[GRID]** (dùng cho Bbox Loss & DFL).
