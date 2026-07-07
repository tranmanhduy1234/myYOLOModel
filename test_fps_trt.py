import time
import torch
import tensorrt as trt

def benchmark_engine(engine_path):
    # 1. Khởi tạo Runtime và nạp file Engine
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    
    print(f"Đang nạp file TensorRT Engine: {engine_path}...")
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    # Tạo ngữ cảnh thực thi (Execution Context)
    context = engine.create_execution_context()

    # 2. Khởi tạo dữ liệu đầu vào bằng PyTorch (FP16 nằm sẵn trên CUDA)
    # Cấu hình chuẩn khớp với lúc xuất mạng: Batch=5, C=3, H=480, W=480
    input_tensor = torch.randn(5, 3, 480, 480, device="cuda", dtype=torch.float16)

    # 3. Khởi tạo bộ nhớ đệm cho Đầu ra (Output Tensors) bằng PyTorch
    # Kích thước phải khớp chính xác 100% với các cổng ra của DetectHead
    cls_tensor = torch.empty(5, 4725, 80, device="cuda", dtype=torch.float16)
    box_tensor = torch.empty(5, 4725, 4, device="cuda", dtype=torch.float16)
    reg_raw_tensor = torch.empty(5, 64, 4725, device="cuda", dtype=torch.float16)
    anchors_tensor = torch.empty(4725, 2, device="cuda", dtype=torch.float16)
    strides_tensor = torch.empty(4725, 1, device="cuda", dtype=torch.float16)

    # 4. RÀNG BUỘC Ô NHỚ (Tensor Address Binding) - Cú pháp chuẩn của TensorRT 10.x
    # Ép TensorRT đọc/ghi trực tiếp trên con trỏ vùng nhớ của PyTorch
    context.set_tensor_address("images", input_tensor.data_ptr())
    context.set_tensor_address("cls", cls_tensor.data_ptr())
    context.set_tensor_address("box", box_tensor.data_ptr())
    context.set_tensor_address("reg_raw", reg_raw_tensor.data_ptr())
    context.set_tensor_address("anchors", anchors_tensor.data_ptr())
    context.set_tensor_address("strides", strides_tensor.data_ptr())

    # Lấy ID luồng CUDA hiện tại của PyTorch để đồng bộ hóa
    cuda_stream = torch.cuda.current_stream().cuda_stream

    # 5. Khởi động GPU (Warm-up 10 vòng) để GPU đạt xung nhịp đỉnh
    print("Đang khởi động (Warm-up) GPU...")
    for _ in range(10):
        context.execute_async_v3(cuda_stream)

    # Đảm bảo GPU đã hoàn thành hết các lệnh chờ trước khi bấm giờ
    torch.cuda.synchronize()
    
    # 6. TIẾN HÀNH BẤM GIỜ CHÍNH THỨC (100 VÒNG LẶP)
    print("Bắt đầu cuộc đua tốc độ TensorRT Engine (100 loops)...")
    start_time = time.time()
    
    for _ in range(100):
        context.execute_async_v3(cuda_stream)
        
    # Ép CPU phải đợi GPU chạy xong hoàn toàn vòng lặp thứ 100 rồi mới dừng đồng hồ
    torch.cuda.synchronize()
    end_time = time.time()

    # 7. TỔNG HỢP KẾT QUẢ
    total_time = end_time - start_time
    fps = (100 * 5) / total_time
    ms_per_batch = (total_time / 100) * 1000

    print("\n⚡=== KẾT QUẢ TỐI ƯU HÓA TENSORRT 10.X ===")
    print(f"Tổng thời gian chạy (100 loops): {total_time:.4f} giây")
    print(f"Thời gian xử lý mỗi Batch (5 ảnh): {ms_per_batch:.2f} ms")
    print(f"TỐC ĐỘ ĐẠT ĐƯỢC: {fps:.2f} FPS")
    print("=========================================\n")

if __name__ == "__main__":
    benchmark_engine("yolov10_custom.engine")