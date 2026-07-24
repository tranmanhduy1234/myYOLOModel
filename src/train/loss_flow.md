flowchart TD

    %% ======================= INPUT =======================
    PREDS["preds (từ DetectHead.forward)<br/>o2m/o2o: cls(bs,A,nc) logit<br/>box(bs,A,4) <b>[PIXEL] xyxy</b> đã decode<br/>reg_raw(bs,4·reg_max,A) <b>[GRID]</b> logit DFL<br/>anchors(A,2) [GRID] offset 0.5<br/>strides(A,1)"]
    TARGETS["targets: list[dict], len=batch_size<br/>boxes(N,4) <b>[PIXEL] xyxy</b>, labels(N,)<br/>N khác nhau mỗi ảnh"]

    %% ======================= PREPROCESS =======================
    TARGETS --> PRE["preprocess_targets()<br/>tìm n_max = max(N) trong batch<br/>PAD về cùng kích thước"]
    PRE --> GT["gt_bboxes (bs,n_max,4) <b>[PIXEL] xyxy</b>, pad=0<br/>gt_labels (bs,n_max,1) long, pad=0<br/>mask_gt (bs,n_max,1) bool: True=GT thật"]

    PREDS --> BR1
    GT --> BR1

    %% ======================= _branch_loss (chạy 2 LẦN: o2m topk=10, o2o topk=1) =======================
    subgraph BRANCH["_branch_loss(assigner, ...) — GỌI 2 LẦN: nhánh o2m (topk=10) & nhánh o2o (topk=1)"]
        direction TB

        SIG["cls_raw.detach().sigmoid()<br/>→ pd_scores_sig (bs,A,nc)<br/>(không lan truyền grad qua assigner)"]
        APIX["anchors_pixel = anchors · strides<br/>(A,2) <b>[GRID]→[PIXEL]</b>"]

        subgraph ASSIGNER["TaskAlignedAssigner.forward — TOÀN BỘ chạy ở <b>[PIXEL]</b>, xyxy"]
            direction TB
            IN_A["input: pd_scores_sig(bs,A,nc)<br/>pd_bboxes=box_pixel.detach() (bs,A,4) <b>[PIXEL]</b> xyxy<br/>anc_points=anchors_pixel (A,2) <b>[PIXEL]</b><br/>gt_labels/gt_bboxes/mask_gt"]

            IN_A --> CAND["select_candidates_in_gts()<br/>so khoảng cách anchor→4 cạnh GT (ltrb)<br/>→ mask_in_gts (bs,M,A) bool"]
            IN_A --> METRIC["get_box_metrics()<br/>bbox_scores = pd_scores tại đúng class GT<br/>overlaps = CIoU(gt_boxes, pd_boxes) <b>[PIXEL]</b><br/>align_metric = score^α · CIoU^β<br/>→ (bs,M,A), (bs,M,A)"]
            CAND & METRIC --> TOPK["select_topk_candidates()<br/>lấy top-k align_metric mỗi GT<br/>(k=10 cho o2m, k=1 cho o2o)<br/>→ mask_topk (bs,M,A)"]
            TOPK --> POS["mask_pos = mask_topk · mask_in_gts · mask_gt<br/>(bs,M,A) — anchor dương ứng viên"]

            POS --> CONFLICT["select_highest_overlaps()<br/>nếu 1 anchor bị NHIỀU GT chọn trùng<br/>→ chỉ giữ GT có CIoU cao nhất<br/>(1 anchor chỉ thuộc 1 GT,<br/>1 GT có thể có nhiều anchor)"]
            CONFLICT --> IDXFG["target_gt_idx (bs,A): GT gán cho từng anchor<br/>fg_mask (bs,A) bool: anchor là foreground?<br/>mask_pos (bs,M,A) đã dọn xung đột"]

            IDXFG --> GATHER["get_targets()<br/>gather gt_bboxes/gt_labels theo target_gt_idx<br/>(M) → (A), KHÔNG đổi không gian"]
            GATHER --> TGT0["target_labels (bs,A)<br/>target_bboxes (bs,A,4) <b>[PIXEL]</b> xyxy<br/>target_scores (bs,A,nc) one-hot, 0 tại background"]

            TGT0 --> NORM["chuẩn hoá target_scores:<br/>norm = align_metric·max_iou/max_align (mỗi GT)<br/>target_scores *= norm<br/>→ anchor gán tốt hơn có trọng số lớn hơn"]
        end

        SIG --> IN_A
        APIX --> IN_A

        NORM --> TS["target_labels, <b>target_bboxes_pixel</b> (bs,A,4) [PIXEL],<br/>target_scores, fg_mask"]

        TS --> CLS["Classification loss (BCE)<br/>bce(cls_raw, target_scores)<br/>.sum() / target_scores_sum<br/>→ loss_cls (không liên quan toạ độ)"]

        TS --> CONVERT{"CHUYỂN KHÔNG GIAN<br/><b>[PIXEL] → [GRID]</b><br/>chia cho stride_b = strides.unsqueeze(0)"}
        PREDS --> CONVERT
        CONVERT --> PBG["pred_bboxes_grid = box_pixel / stride<br/>(bs,A,4) <b>[GRID]</b> xyxy"]
        CONVERT --> TBG["target_bboxes_grid = target_bboxes_pixel / stride<br/>(bs,A,4) <b>[GRID]</b> xyxy"]
        PREDS --> PDIST["pred_dist = reg_raw.transpose(1,2)<br/>(bs,A,4·reg_max) <b>[GRID]</b> logit"]

        subgraph BBOXLOSS["BboxLoss.forward — TOÀN BỘ chạy ở <b>[GRID]</b>"]
            direction TB
            IN_B["input: pred_dist [GRID] logit<br/>pred_bboxes_grid, target_bboxes_grid [GRID] xyxy<br/>anchors [GRID] offset 0.5 (CHƯA nhân stride)<br/>target_scores, fg_mask"]
            IN_B --> WEIGHT["weight = target_scores.sum(-1)[fg_mask]<br/>(N_pos,1) — trọng số Task-Aligned"]
            IN_B --> IOU["CIoU(pred_bboxes_grid[fg], target_bboxes_grid[fg])<br/>loss_iou = Σ(1-iou)·weight / target_scores_sum<br/>(tỉ lệ nên GRID hay PIXEL đều ra cùng số)"]
            IN_B --> ENC["bbox2dist(anchors, target_bboxes_grid, reg_max-1)<br/>xyxy → <b>ltrb</b> [GRID], clamp[0, reg_max-1-eps]<br/>→ target_ltrb (bs,A,4) số thực"]
            ENC --> DFL["_df_loss(pred_dist[fg], target_ltrb[fg])<br/>tách 2 bin nguyên liền kề (tl, tr=tl+1)<br/>nội suy cross-entropy theo trọng số (wl,wr)<br/>loss_dfl = Σ(...)·weight / target_scores_sum"]
        end
        WEIGHT -.-> IOU
        WEIGHT -.-> DFL
        PBG --> IN_B
        TBG --> IN_B
        PDIST --> IN_B

        IOU --> OUT_B["loss_iou, loss_cls, loss_dfl, n_pos<br/>(đầu ra của _branch_loss)"]
        DFL --> OUT_B
        CLS --> OUT_B
    end

    OUT_B --> GAIN["loss_o2x = box_gain·iou + cls_gain·cls + dfl_gain·dfl<br/>(box_gain=7.5, cls_gain=0.5, dfl_gain=1.5)"]

    GAIN -- "nhánh o2m<br/>(assigner topk=10)" --> LOM["loss_o2m"]
    GAIN -- "nhánh o2o<br/>(assigner topk=1)" --> LOO["loss_o2o"]

    LOM & LOO --> TOTAL["total = o2m_weight·loss_o2m + o2o_weight·loss_o2o<br/>(mặc định weight=1.0 cho cả 2)"]
    TOTAL --> ITEMS["items = {loss, loss_o2m, loss_o2o,<br/>o2m/iou, o2m/cls, o2m/dfl, o2m/n_pos,<br/>o2o/iou, o2o/cls, o2o/dfl, o2o/n_pos}<br/>(đã .item(), dùng để log)"]
    TOTAL --> RETURN["return total (Tensor có grad, để backward),<br/>items (dict để log)"]

    style APIX fill:#fff3cd
    style CONVERT fill:#fff3cd
    style PBG fill:#d0e8ff
    style TBG fill:#d0e8ff
    style PDIST fill:#d0e8ff
    style IN_A fill:#ffe0e0
    style TGT0 fill:#ffe0e0
    style IN_B fill:#e0d0ff
    style TOTAL fill:#d0ffd0