import tensorrt as trt

def build_engine(onnx_path, engine_path, is_dynamic=False):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # FIX LỖI CHÍ MẠNG: Kích hoạt chế độ Strongly Typed cho TensorRT 10.x
    # Chế độ này giúp mạng tự động chạy FP16 vì nhận diện file ONNX đầu vào đã là FP16
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    
    print("Đang đọc file ONNX FP16...")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("LỖI: Không thể parse file ONNX.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    config = builder.create_builder_config()
    
    # ĐÃ XÓA dòng config.set_flag(trt.BuilderFlag.FP16) vì không còn cần thiết ở TRT 10!
    
    if is_dynamic:
        print("Đang cấu hình Dynamic Shape cho Batch Size (1 -> 5 -> 16)...")
        profile = builder.create_optimization_profile()
        profile.set_shape("images", (1, 3, 480, 480), (5, 3, 480, 480), (16, 3, 480, 480))
        config.add_optimization_profile(profile)
    
    print("Đang tối ưu hóa đồ thị mạng Strongly Typed (Mất khoảng 1-3 phút)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("LỖI: Biên dịch Engine thất bại.")
        return False
        
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f" Chúc mừng! Đã tạo file Engine thành công tại: {engine_path}")
    return True

if __name__ == "__main__":
    # Đổi tên file đầu vào sang file FP16 vừa xuất ở Bước 1 nhé
    build_engine("yolov10_custom_fp16.onnx", "yolov10_custom.engine", is_dynamic=False)