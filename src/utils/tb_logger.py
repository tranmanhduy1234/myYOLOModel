from torch.utils.tensorboard import SummaryWriter

def log_loss_items(writer: SummaryWriter, items: dict, step: int, phase: str = "train"):
    assert phase in ("train", "val"), f"phase phai la 'train' hoac 'val', duoc '{phase}'"

    # Bieu do 1: tong loss + loss tung nhanh (3 duong cung 1 bieu do)
    writer.add_scalars(f"{phase}/loss_total", {
        "total": items["loss"],
        "o2m": items["loss_o2m"],
        "o2o": items["loss_o2o"],
    }, step)

    # Bieu do 2: 3 thanh phan cua nhanh o2m (3 duong cung 1 bieu do)
    writer.add_scalars(f"{phase}/loss_o2m_parts", {
        "iou": items["o2m/iou"],
        "cls": items["o2m/cls"],
        "dfl": items["o2m/dfl"],
    }, step)

    # Bieu do 3: 3 thanh phan cua nhanh o2o (3 duong cung 1 bieu do)
    writer.add_scalars(f"{phase}/loss_o2o_parts", {
        "iou": items["o2o/iou"],
        "cls": items["o2o/cls"],
        "dfl": items["o2o/dfl"],
    }, step)

    # Bieu do 4: so luong anchor duong cua 2 nhanh (tach rieng vi scale la SO NGUYEN,
    # khac han scale float nho cua loss - gop chung se lam bieu do loss bi det).
    writer.add_scalars(f"{phase}/n_pos", {
        "o2m": items["o2m/n_pos"],
        "o2o": items["o2o/n_pos"],
    }, step)

# ==============================================================================
# VI DU SU DUNG trong vong lap training/validation
# ==============================================================================
if __name__ == "__main__":
    import random
    import shutil
    import os

    log_dir = "runs/demo_loss_logging"
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # --- gia lap 1 vai epoch train + validate, loss giam dan de test truc quan ---
    global_step = 0
    for epoch in range(5):
        # ---- giai doan TRAIN: log moi batch (step = global_step tang dan) ----
        for batch_idx in range(20):
            fake_items = {
                "loss": 10.0 / (epoch + 1) + random.uniform(-0.3, 0.3),
                "loss_o2m": 6.0 / (epoch + 1) + random.uniform(-0.2, 0.2),
                "loss_o2o": 4.0 / (epoch + 1) + random.uniform(-0.2, 0.2),
                "o2m/iou": 2.0 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2m/cls": 3.0 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2m/dfl": 1.0 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2o/iou": 1.5 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2o/cls": 2.0 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2o/dfl": 0.5 / (epoch + 1) + random.uniform(-0.1, 0.1),
                "o2m/n_pos": random.randint(50, 200),
                "o2o/n_pos": random.randint(5, 20),
            }
            log_loss_items(writer, fake_items, step=global_step, phase="train")
            global_step += 1

        # ---- giai doan VAL: thuong log 1 lan/epoch, step = epoch (khong dung
        # global_step cua train vi tan so ghi khac nhau hoan toan) ----
        fake_val_items = {
            "loss": 9.0 / (epoch + 1),
            "loss_o2m": 5.5 / (epoch + 1),
            "loss_o2o": 3.5 / (epoch + 1),
            "o2m/iou": 1.8 / (epoch + 1),
            "o2m/cls": 2.8 / (epoch + 1),
            "o2m/dfl": 0.9 / (epoch + 1),
            "o2o/iou": 1.3 / (epoch + 1),
            "o2o/cls": 1.8 / (epoch + 1),
            "o2o/dfl": 0.4 / (epoch + 1),
            "o2m/n_pos": random.randint(50, 200),
            "o2o/n_pos": random.randint(5, 20),
        }
        log_loss_items(writer, fake_val_items, step=epoch, phase="val")

    writer.close()
    print(f"Da ghi log demo vao '{log_dir}'. Chay: tensorboard --logdir {log_dir}")