"""
loss.py
=======
Ham loss cho NMSFreeDetector: Task-Aligned Assigner (TAL) +
CIoU loss + Distribution Focal Loss (DFL) + BCE classification loss, ap dung
rieng cho CA HAI nhanh du doan cua head:

  - o2m (one-to-many): topk lon (mac dinh 10) -> nhieu anchor duong cho 1 GT,
    giup training on dinh & hoi tu nhanh.
  - o2o (one-to-one)  : topk = 1 -> moi GT chi co DUY NHAT 1 anchor duong,
    day la nhanh giup model bo duoc NMS luc inference.

Tong loss:
    total = o2m_weight * loss_o2m + o2o_weight * loss_o2o
    voi   loss_o2x = box_gain * loss_iou + cls_gain * loss_cls + dfl_gain * loss_dfl

Dinh dang input mong doi:
    preds  : dict tra ve boi DetectHead.forward() khi model.train() dang bat,
             tuc la {"o2m": {"cls","box","reg_raw"}, "o2o": {...},
                     "anchors": (A,2), "strides": (A,1)}
    targets: list do dai = batch_size, moi phan tu la dict:
                 {"boxes": FloatTensor (N,4) xyxy o khong gian PIXEL cua anh dau vao,
                  "labels": LongTensor (N,)}
             N co the = 0 (anh khong co object nao).

------------------------------------------------------------------------------
QUY UOC VE KHONG GIAN TOA DO (RAT QUAN TRONG - de tam do khi doc code):
------------------------------------------------------------------------------
Trong file nay co 2 "khong gian" (coordinate space) khac nhau, moi tensor
lien quan toi toa do/box DEU duoc chu thich ro no dang o khong gian nao:

  [PIXEL]  Khong gian pixel cua anh dau vao goc (vi du anh 640x640 thi toa do
           chay tu 0 -> 640). Day la khong gian cua GT box (targets["boxes"])
           va cua box du doan sau khi head da decode (box_pixel).
           TaskAlignedAssigner (gan GT <-> anchor) LUON LUON chay trong khong
           gian nay, vi can so sanh truc tiep voi GT box goc.

  [GRID]   Khong gian feature map / grid cell, tuc la toa do PIXEL da chia
           cho stride cua level tuong ung (grid = pixel / stride). 1 don vi
           trong khong gian nay = 1 o luoi cua feature map. Day la khong gian
           dung de tinh CIoU loss + DFL loss (BboxLoss), vi DFL bieu dien
           khoang cach anchor->canh box duoi dang phan phoi roi rac tren
           reg_max bin, va cac bin nay duoc dinh nghia theo don vi grid cell
           (0, 1, 2, ..., reg_max-1), khong theo pixel.

  Phep chuyen doi giua 2 khong gian: PIXEL = GRID * stride  <=>  GRID = PIXEL / stride
  (stride la (A,1), moi anchor co 1 gia tri stride rieng tuy thuoc no thuoc
  level feature map nao - vi du 8/16/32).

Ngoai ra, box/khoang cach con co "dinh dang" (format) khac nhau:
  xyxy  : (x1, y1, x2, y2)      - toa do goc tren-trai va goc duoi-phai.
  cxcywh: (cx, cy, w, h)        - toa do tam + kich thuoc (dung khi xywh=True
                                   trong bbox_iou/dist2bbox).
  ltrb  : (left, top, right, bottom) - khoang cach TUONG DOI tu 1 anchor point
                                   toi 4 canh cua box (dung cho DFL). Don vi
                                   cua ltrb phai khop voi don vi cua khong gian
                                   dang xet (thuong la [GRID] vi DFL chi hoat
                                   dong trong khong gian nay).

Moi tensor toa do trong docstring/comment ben duoi deu duoc ghi ro theo mau:
    ten_bien: shape, [KHONG_GIAN], dinh_dang
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CAC HAM HINH HOC DUNG CHUNG (box IoU, decode/encode khoang cach)
#    Cac ham trong phan nay KHONG GAN CO DINH voi mot khong gian nao - chung
#    hoat dong dung voi bat ky khong gian nao (PIXEL hoac GRID), MIEN LA moi
#    tensor dau vao cua cung 1 lan goi ham phai o CHUNG mot khong gian.
# ==============================================================================

def bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Tinh IoU (hoac bien the GIoU / DIoU / CIoU) giua hai tap box, ho tro broadcasting.
    Ket qua IoU la mot ty le (khong don vi) nen ham nay dung duoc voi CA khong
    gian [PIXEL] lan [GRID], mien la box1 va box2 cung khong gian voi nhau.

    Tham so:
        box1, box2: Tensor (..., 4). Cac chieu dau (...) phai broadcast duoc
                    voi nhau. Cung mot khong gian toa do (PIXEL hoac GRID tuy
                    noi goi), dinh dang xac dinh boi tham so xywh.
        xywh      : True neu box dang cxcywh (cx, cy, w, h);
                    False neu dang xyxy (x1, y1, x2, y2).
        GIoU/DIoU/CIoU: chon bien the can tinh them (uu tien CIoU > DIoU > GIoU
                    neu nhieu co bat dong thoi, theo thu tu kiem tra ben duoi).

    Tra ve:
        Tensor (...,) - gia tri IoU (hoac GIoU/DIoU/CIoU) tuong ung, khong don
        vi (unitless), da bo chieu cuoi.
    """
    if xywh:
        # box dang cxcywh -> tach thanh 4 tensor (..., 1): tam (x1c,y1c) + nua-kich-thuoc
        (x1c, y1c, w1, h1), (x2c, y2c, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2

        # Quy doi ve xyxy noi bo de tinh toan thong nhat:
        # goc tren-trai (x1, y1) va goc duoi-phai (x2, y2)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1c - w1_, x1c + w1_, y1c - h1_, y1c + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2c - w2_, x2c + w2_, y2c - h2_, y2c + h2_
    else:
        # box da la dang xyxy -> tach truc tiep
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        # Tinh width/height (dung don vi voi khong gian dang xet - PIXEL hoac GRID)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Dien tich vung giao nhau (Overlap), don vi = (don_vi_khong_gian)^2
    # Chieu rong giao nhau = min(x2_box1, x2_box2) - max(x1_box1, x1_box2)
    # Chieu cao giao nhau  = min(y2_box1, y2_box2) - max(y1_box1, y1_box2)
    # .clamp(0) de dam bao neu 2 box khong cham nhau (gia tri am) thi dua ve bang 0.
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union  # ty le, khong don vi -> ket qua giong het du tinh o PIXEL hay GRID

    if CIoU or DIoU or GIoU:
        # Box bao nho nhat (smallest enclosing box) chua ca box1 va box2,
        # cung don vi voi khong gian dang xet.
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # chieu rong box bao
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # chieu cao box bao

        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # binh phuong duong cheo cua box bao chung C

            # Binh phuong khoang cach Euclid giua 2 tam cua box1 va box2 (rho^2).
            # Toa do tam X = (x1 + x2) / 2, doan code duoi gop chung phep chia
            # cho 2 binh phuong len thanh chia 4.
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

            if CIoU:
                # v: do lech ty le width/height giua 2 box (khong don vi vi la
                # ty so w/h), 0 <= v <= 1 (gia tri actan lon nhat luon < PI / 2)
                v = (4 / math.pi ** 2) * (
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
                ) ** 2
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return (iou - (rho2 / c2 + v * alpha)).squeeze(-1)

            return (iou - rho2 / c2).squeeze(-1)

        c_area = cw * ch + eps
        return (iou - (c_area - union) / c_area).squeeze(-1)

    return iou.squeeze(-1)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Decode khoang cach ltrb (left, top, right, bottom) tinh tu anchor point
    thanh bounding box. KHONG GIAN dau ra = KHONG GIAN cua distance va
    anchor_points dau vao (ham nay khong tu quy doi don vi).

    Tham so:
        distance     : Tensor (..., 4), dinh dang ltrb (left, top, right, bottom).
                       Don vi = don vi cua khong gian dang xet (PIXEL hoac GRID),
                       phai KHOP voi don vi cua anchor_points.
        anchor_points: Tensor (..., 2), toa do tam anchor (x, y), cung khong
                       gian voi distance.
        xywh         : True  -> tra ve dinh dang cxcywh (cx, cy, w, h);
                       False -> tra ve dinh dang xyxy (x1, y1, x2, y2).

    Tra ve:
        Tensor (..., 4) - bounding box da decode, CUNG khong gian voi input.
    """
    lt, rb = distance.chunk(2, dim)     # left-top, right-bottom (ltrb -> 2 nua)
    x1y1 = anchor_points - lt           # goc tren-trai = anchor - (left, top)
    x2y2 = anchor_points + rb           # goc duoi-phai = anchor + (right, bottom)
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox, reg_max):
    """
    Encode bounding box (xyxy) thanh khoang cach ltrb so voi anchor point,
    dung lam ground truth cho DFL. Luu y: day la gia tri hinh hoc dang SO
    THUC (float), CHUA phai phan phoi xac suat - viec bien no thanh phan
    phoi (soft label cho 2 bin lien ke) duoc thuc hien ben trong
    BboxLoss._df_loss.

    Ham nay CHI duoc goi trong khong gian [GRID] tren thuc te (xem
    BboxLoss.forward), vi reg_max la so bin DFL - moi bin tuong ung 1 don vi
    grid cell. Neu goi voi anchor_points/bbox o khong gian PIXEL, ket qua se
    SAI vi se bi clamp nham theo thang do pixel.

    Tham so:
        anchor_points: Tensor (A, 2)          - toa do tam anchor, [GRID].
        bbox         : Tensor (bs, A, 4) xyxy - bounding box can encode, [GRID].
        reg_max      : so bin - 1 (int), dung de clamp gia tri dau ra vao
                       [0, reg_max - eps], dam bao khoang cach nam gon trong
                       pham vi bieu dien duoc cua DFL (reg_max bin: 0..reg_max-1).

    Tra ve:
        Tensor (..., 4), dinh dang ltrb, [GRID], da clamp trong [0, reg_max - 0.01].
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    lt = anchor_points - x1y1   # left, top = anchor - goc tren-trai
    rb = x2y2 - anchor_points   # right, bottom = goc duoi-phai - anchor
    return torch.cat((lt, rb), -1).clamp_(0, reg_max - 0.01)


# ==============================================================================
# 2. TASK-ALIGNED ASSIGNER
#    QUAN TRONG: toan bo class nay hoat dong trong khong gian [PIXEL], dinh
#    dang xyxy cho moi box. Day la lua chon co chu dich: so sanh IoU/khoang
#    cach truc tiep voi GT box goc (von o pixel) ma khong can quy doi qua lai.
# ==============================================================================
class TaskAlignedAssigner(nn.Module):
    """
    Gan GT cho anchor dua tren align_metric = cls_score^alpha * IoU^beta.

    topk = N (N > 1) -> kieu one-to-many (nhieu anchor duong / GT).
    topk = 1          -> kieu one-to-one (dung cho nhanh NMS-free).
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Thuc hien Task-Aligned Assignment (TAL): gan Ground Truth cho tung Anchor,
        bien doi GT tu bieu dien theo Object/GT (M) sang bieu dien theo
        Prediction/Anchor (A), dong thoi tao ra cac soft target phuc vu tinh loss.
        Toan bo phep tinh trong ham nay dien ra trong khong gian [PIXEL].

        Dau vao:
            pd_scores : (bs, A, nc)
                Xac suat phan loai cua tung Anchor sau Sigmoid. Khong lien
                quan toa do nen khong co khai niem khong gian/dinh dang.
            pd_bboxes : (bs, A, 4), [PIXEL], xyxy
                Bounding Box du doan cua tung Anchor.
            anc_points: (A, 2), [PIXEL]
                Toa do tam cua cac Anchor = anchors_grid * stride (xem
                DetectionLoss._branch_loss).
            gt_labels : (bs, M, 1)
                Nhan lop cua cac Ground Truth (khong lien quan toa do).
            gt_bboxes : (bs, M, 4), [PIXEL], xyxy
                Bounding Box Ground Truth, cung khong gian voi anh dau vao goc.
            mask_gt   : (bs, M, 1)
                Mat na xac dinh GT hop le.
                True  : GT ton tai.
                False : GT padding.

        Dau ra:
            target_labels : (bs, A)
                Nhan lop duoc gan cho tung Anchor.
            target_bboxes : (bs, A, 4), [PIXEL], xyxy
                Bounding Box muc tieu ma tung Anchor phai hoi quy (lay truc
                tiep tu gt_bboxes nen giu nguyen khong gian PIXEL cua GT goc -
                noi goi (BboxLoss) se tu quy doi sang [GRID] neu can).
            target_scores : (bs, A, nc)
                Soft Classification Target sau khi duoc chuan hoa bang
                Task-Aligned Metric. Background co vector toan 0.
            fg_mask : (bs, A)
                Mat na Foreground.
                True  : Anchor duoc gan cho mot GT.
                False : Background Anchor.
            target_gt_idx : (bs, A)
                Chi so GT ma moi Anchor duoc gan sau khi giai quyet xung dot.
                Moi Anchor chi thuoc duy nhat mot GT, nhung mot GT co the so huu
                nhieu Anchor.
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device
        A = pd_scores.shape[1]

        # Truong hop khong co GT nao trong ca batch -> tra ve toan bo la background
        if self.n_max_boxes == 0:
            return (
                torch.full((self.bs, A), self.nc, dtype=torch.long, device=device),
                torch.zeros((self.bs, A, 4), device=device),
                torch.zeros((self.bs, A, self.nc), device=device),
                torch.zeros((self.bs, A), dtype=torch.bool, device=device),
                torch.zeros((self.bs, A), dtype=torch.long, device=device),
            )

        # Buoc 1: xac dinh tap anchor "ung vien duong" cho tung GT (mask_pos)
        # cung align_metric va overlaps (CIoU, ca hai deu tinh tren khong gian
        # PIXEL nhung ket qua khong don vi nen se giong het neu tinh o GRID).
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        # mask_pos:     [bs, M, A] (0/1)
        # align_metric: [bs, M, A] (khong don vi)
        # overlaps:     [bs, M, A] (CIoU, khong don vi)

        # Buoc 2: giai quyet xung dot - mot Anchor bat ky co the vo tinh lot vao
        # mat xanh cua nhieu hop nhan (GT) khac nhau (vi du: cac vat the nam de
        # len nhau hoac nam rat gan nhau). Tuy nhien, nguyen tac toan hoc cua
        # detector la: MOT Anchor chi duoc phep dai dien du doan cho DUY NHAT
        # mot vat the nhan that. Nguoc lai, mot GT van co the duoc chon boi
        # nhieu Anchor.
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )
        # target_gt_idx [bs, A]:    chi so GT duoc gan cho tung Anchor.
        # fg_mask       [bs, A]:    Anchor la Foreground (1) hay Background (0).
        # mask_pos      [bs, M, A]: ma tran phan bo GT-Anchor sau khi da don dep xung dot.

        # Buoc 3: chuyen toan bo Ground Truth dang to chuc theo GT (M) sang
        # to chuc theo Anchor (A) dua tren ket qua Assignment. target_bboxes
        # tra ve van la [PIXEL] xyxy (lay nguyen tu gt_bboxes dau vao).
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )
        # target_labels: [bs, A]
        # target_bboxes: [bs, A, 4], [PIXEL], xyxy
        # target_scores: [bs, A, nc]

        # Buoc 4: chuan hoa align_metric ve thang do CIoU tot nhat cua tung GT,
        # roi dung no de lam mem (soft) vector one-hot cua classification target.
        align_metric *= mask_pos  # chi giu lai cac Positive Anchor cua tung GT
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # [bs, M, 1]
        # Voi moi GT: Alignment Metric lon nhat trong cac Positive Anchor cua no.
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # [bs, M, 1]
        # Voi moi GT: CIoU lon nhat trong cac Positive Anchor cua no.

        # GT nao cung duoc chuan hoa ve Anchor tot nhat cua chinh no:
        #   - chia cho max_align : cong bang giua cac GT, loai bo khac biet ve scale.
        #   - nhan voi max_iou   : giu lai thong tin ve chat luong dinh vi cua tung GT.
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        # Ket qua: Anchor co chat luong tot hon se dong gop nhieu hon vao
        # classification loss, thay vi moi positive anchor deu co trong so bang nhau.

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Xac dinh mask_pos: tap anchor duong cuoi cung cho tung GT, la giao cua
        3 dieu kien: (1) anchor nam trong GT box, (2) GT do thuc su ton tai
        (khong phai padding), (3) anchor nam trong top-k align_metric cao nhat
        cua GT do. Tat ca deu tinh trong khong gian [PIXEL].

        Dau vao:
            pd_scores : (bs, A, nc)         - da qua sigmoid
            pd_bboxes : (bs, A, 4), [PIXEL], xyxy
            gt_labels : (bs, M, 1)
            gt_bboxes : (bs, M, 4), [PIXEL], xyxy
            anc_points: (A, 2), [PIXEL]     - toa do tam anchor
            mask_gt   : (bs, M, 1)          - True = GT that

        Tra ve:
            mask_pos     : (bs, M, A) - 1 neu anchor la positive cua GT tuong ung.
            align_metric : (bs, M, A) - m = score^alpha * IoU^beta.
            overlaps     : (bs, M, A) - CIoU giua GT va anchor.
        """
        # mask_in_gts * mask_gt => "Anchor nam trong GT" AND "GT do thuc su ton tai"
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)  # (bs,M,A)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )

        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps
        # mask_pos:     [bs, M, A]
        # align_metric: [bs, M, A]
        # overlaps:     [bs, M, A]

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Tinh align_metric va overlaps (CIoU) giua tung cap (GT, Anchor) hop le.
        pd_bboxes va gt_bboxes phai cung khong gian [PIXEL], xyxy (CIoU la ty
        le nen ket qua khong doi neu doi sang GRID, nhung trong pipeline nay
        luon truyen vao o PIXEL).

        Dau vao:
            pd_scores: (bs, A, nc)              - sigmoid
            pd_bboxes: (bs, A, 4), [PIXEL], xyxy
            gt_labels: (bs, M, 1)
            gt_bboxes: (bs, M, 4), [PIXEL], xyxy
            mask_gt  : (bs, M, A) bool - cap (GT, Anchor) nao can tinh.

        Tra ve:
            align_metric: (bs, M, A)
                Chi so Alignment Metric trong Task-Aligned Assigner:
                    m = score^alpha * IoU^beta
                dung de xep hang va chon Top-k positive anchors cho moi GT.
            overlaps: (bs, M, A)
                Gia tri CIoU (khong don vi) giua tung GT va tung prediction box:
                    overlaps[b, gt, a] = CIoU(GT_gt, Pred_a)
        """
        bs, M = self.bs, self.n_max_boxes
        A = pd_scores.shape[1]
        mask_gt = mask_gt.bool()

        # Luu CIoU giua anchor va GT
        overlaps = torch.zeros((bs, M, A), dtype=pd_bboxes.dtype, device=pd_bboxes.device)

        # Classification score cua anchor a doi voi class cua GT m
        # (ve co ban la anchor do co du doan gi cho dung class cua GT).
        bbox_scores = torch.zeros((bs, M, A), dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros((2, bs, M), dtype=torch.long, device=pd_scores.device)
        ind[0] = torch.arange(bs, device=pd_scores.device).view(-1, 1).expand(-1, M)
        # tensor([
        #     [0,0,0,0],
        #     [1,1,1,1],
        #     [2,2,2,2],
        # ]) - moi phan tu la ID sample trong batch du lieu

        ind[1] = gt_labels.squeeze(-1).clamp(0, self.nc - 1)
        # tensor([
        #     [5,2,8,1],   # batch 0
        #     [0,7,4,3],   # batch 1
        #     [2,1,6,5],   # batch 2
        # ]) - moi phan tu la ID label cua sample

        # bbox_scores: [bs,M,A]
        # pd_scores  : [bs,A,nc]
        # mask_gt    : [bs,M,A]
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
        # chua score du doan cua dung class GT tai cac anchor hop le;
        # cac anchor khong hop le co gia tri 0.

        # pd_bboxes: (bs,A,4), [PIXEL], xyxy
        # gt_bboxes: (bs,M,4), [PIXEL], xyxy
        pd_boxes_exp = pd_bboxes.unsqueeze(1).expand(-1, M, -1, -1)[mask_gt]
        # moi GT se deu "nhin thay" toan bo cac predict box
        gt_boxes_exp = gt_bboxes.unsqueeze(2).expand(-1, -1, A, -1)[mask_gt]
        # moi GT duoc lap lai A lan de ghep voi tat ca prediction box
        if pd_boxes_exp.numel():
            overlaps[mask_gt] = bbox_iou(gt_boxes_exp, pd_boxes_exp, xywh=False, CIoU=True).clamp(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask):
        """
        Voi moi GT, chon ra topk anchor co align_metric cao nhat. Ham nay
        khong lien quan toa do/khong gian, chi thao tac tren chi so (index).

        Dau vao:
            metrics  : (bs, M, A)     - align_metric.
            topk_mask: (bs, M, topk)  - mask GT hop le, mo rong theo chieu topk.

        Tra ve:
            count_tensor: (bs, M, A) - gia tri 1.0 neu anchor duoc chon vao top-k
                cua GT tuong ung, nguoc lai 0.0. Cac vi tri bi trung lap trong
                top-k (do trung idx) se bi loai bo (dat ve 0) de tranh dem 2 lan.
        """
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        # topk_metrics, topk_idxs: [bs, M, topk]

        if topk_mask is None:
            topk_mask = (topk_metrics.amax(-1, keepdim=True) > self.eps).expand_as(topk_idxs)

        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)  # [bs, M, A]

        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)

        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k: k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        # count_tensor: (bs, M, A). Chi chua 1.0 (anchor trung tuyen vao top-k)
        # va 0.0 (anchor bi loai, bao gom ca truong hop bi trung idx > 1 lan).

        return count_tensor.to(metrics.dtype)

    @staticmethod
    def select_candidates_in_gts(anc_points, gt_bboxes, eps=1e-9):
        """
        Xac dinh anchor nao nam ben trong tung GT box (dieu kien can de duoc
        xem xet lam anchor duong). Tinh bang cach so sanh khoang cach tu
        anchor point toi 4 canh cua GT box - CA HAI deu phai o cung khong
        gian [PIXEL] thi khoang cach moi co y nghia dung.

        Dau vao:
            anc_points: (A, 2), [PIXEL]      - toa do tam anchor.
            gt_bboxes : (bs, M, 4), [PIXEL], xyxy.

        Tra ve:
            mask_in_gts: (bs, M, A) bool - True neu anchor point nam trong GT box.
        """
        n_anchors = anc_points.shape[0]
        bs, M, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # [bs*M, 1, 2] - goc tren-trai / duoi-phai cua GT
        deltas = torch.cat(
            (anc_points.unsqueeze(0) - lt, rb - anc_points.unsqueeze(0)), dim=2
        ).view(bs, M, n_anchors, -1)
        # deltas: [bs, M, A, 4] (ltrb) - khoang cach tu anchor toi 4 canh cua GT box,
        # don vi = pixel (vi ca 2 dau vao deu [PIXEL]).
        return deltas.amin(3).gt_(eps)
        # Lay khoang cach nho nhat trong 4 khoang cach: neu > 0 thi anchor nam
        # trong GT box, nguoc lai (<= 0) la nam ngoai.

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Giai quyet xung dot khi mot Anchor duoc nhieu GT cung chon: chi giu lai
        GT co CIoU (overlaps) cao nhat voi anchor do. Nguyen tac: mot Anchor chi
        duoc dai dien cho DUY NHAT mot GT, nhung mot GT van co the co nhieu Anchor.
        Day la thao tac tren chi so/mask, khong lien quan truc tiep don vi toa do.

        Dau vao:
            mask_pos    : (bs, M, A) - mask positive truoc khi xu ly xung dot.
            overlaps    : (bs, M, A) - CIoU giua GT va anchor.
            n_max_boxes : int        - so GT toi da (M) trong batch.

        Tra ve:
            target_gt_idx: (bs, A)     - ID cua GT duoc gan cho tung Anchor.
            fg_mask      : (bs, A)     - so luong GT ma moi Anchor duoc gan
                (sau xu ly xung dot, gia tri chi con 0 hoac 1).
            mask_pos     : (bs, M, A)  - mask positive sau khi da don dep xung dot.
        """
        fg_mask = mask_pos.sum(-2)  # [bs, A] - tong theo chieu M: moi Anchor dang duoc gan cho bao nhieu GT
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # [bs, M, A]
            max_overlaps_idx = overlaps.argmax(1)  # [bs, A] - doc theo chieu M, tim GT tuong thich cao nhat voi tung Anchor
            is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes).permute(0, 2, 1).to(overlaps.dtype)  # [bs, M, A]

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)

        target_gt_idx = mask_pos.argmax(-2)
        # target_gt_idx [bs, A]:    ID cua GT duoc gan cho tung Anchor.
        # fg_mask       [bs, A]:    Anchor la Foreground (1) hay Background (0).
        # mask_pos      [bs, M, A]: mask GT-Anchor sau khi da don dep xung dot.
        # -> minh chung: 1 anchor chi duoc chon boi 1 GT, nhung 1 GT co the chon nhieu Anchor.
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Anh xa GT tu chieu M (so GT) sang chieu A (so Anchor) dua tren
        target_gt_idx da tinh o buoc truoc, tao ra target hoan chinh cho
        tung Anchor. Day chi la phep gather (lay lai) tu gt_bboxes theo index
        nen KHONG doi khong gian toa do: target_bboxes van la [PIXEL] xyxy
        y het gt_bboxes dau vao.

        Dau vao:
            gt_labels     : (bs, M, 1)  - long.
            gt_bboxes     : (bs, M, 4), [PIXEL], xyxy.
            target_gt_idx : (bs, A)     - ID cua GT duoc gan cho tung Anchor.
            fg_mask       : (bs, A)     - Foreground mask.

        Tra ve:
            target_labels: (bs, A)              - class ma moi anchor phai du doan.
            target_bboxes: (bs, A, 4), [PIXEL], xyxy - box ma moi anchor phai hoi quy.
            target_scores: (bs, A, nc)          - one-hot classification target
                                                   (float), bang 0 tai anchor background.
        """
        bs = gt_labels.shape[0]
        batch_ind = torch.arange(bs, dtype=torch.long, device=gt_labels.device).unsqueeze(-1)  # [bs, 1]

        target_gt_idx_flat = target_gt_idx + batch_ind * self.n_max_boxes  # [bs, A] - offset de flatten
        target_labels = gt_labels.long().flatten()[target_gt_idx_flat]  # [bs, A]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx_flat]  # [bs, A, 4], [PIXEL], xyxy
        # -> ky thuat anh xa (gather thong qua flatten index)

        target_labels = target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.nc)  # bien class ID thanh vector one-hot: [bs, A, nc]

        fg_scores_mask = fg_mask.unsqueeze(-1).repeat(1, 1, self.nc).bool()  # [bs, A, nc]
        target_scores = torch.where(fg_scores_mask, target_scores, torch.zeros_like(target_scores))
        # Anchor background -> vector target_scores toan 0.

        return target_labels, target_bboxes, target_scores.float()


# ==============================================================================
# 3. BBOX LOSS (CIoU + DFL)
#    QUAN TRONG: toan bo class nay hoat dong trong khong gian [GRID] (khac voi
#    Assigner o phan 2 - luon la [PIXEL]). Noi goi (DetectionLoss._branch_loss)
#    co trach nhiem quy doi box tu PIXEL sang GRID (chia cho stride) TRUOC khi
#    truyen vao day.
# ==============================================================================
class BboxLoss(nn.Module):
    """Tinh box regression loss gom CIoU loss va Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        """
        Dau vao:
            pred_dist : (bs, A, 4 * reg_max), [GRID], logits ltrb roi rac
                Logits phan phoi khoang cach (l, t, r, b) cua tung Anchor,
                chua qua Softmax. Moi canh (l/t/r/b) duoc bieu dien boi
                reg_max logits, tuong ung reg_max bin roi rac tren truc so
                thuc [0, reg_max-1] TINH THEO DON VI GRID CELL (khong phai pixel).
            pred_bboxes : (bs, A, 4), [GRID], xyxy
                Bounding Box du doan sau khi decode, da chia cho stride
                (xem DetectionLoss._branch_loss: pred_bboxes_grid = box_pixel / stride).
            anchor_points : (A, 2), [GRID]
                Toa do tam cua cac Anchor, dang GRID goc (offset 0.5), CHUA
                nhan stride - khac voi anc_points [PIXEL] dung trong Assigner.
            target_bboxes : (bs, A, 4), [GRID], xyxy
                Bounding Box muc tieu sau Assignment, DA duoc quy doi tu
                [PIXEL] (dau ra cua assigner) sang [GRID] o noi goi.
            target_scores : (bs, A, nc)
                Soft Classification Target sinh boi TaskAlignedAssigner. Duoc dung
                de tinh trong so (weight) cua tung Positive Anchor.
            target_scores_sum : float
                Tong trong so cua toan bo Positive Anchor trong batch, dung de
                chuan hoa Regression Loss.
            fg_mask : (bs, A)
                Mat na Foreground.
                True  : Anchor duoc gan cho mot GT.
                False : Background Anchor.

        Dau ra:
            loss_iou : Tensor - CIoU Loss sau khi chuan hoa (khong don vi, vi
                       CIoU la ty le nen tinh o GRID hay PIXEL deu cho cung
                       ket qua so hoc).
            loss_dfl : Tensor - Distribution Focal Loss sau khi chuan hoa (PHU
                       THUOC khong gian GRID, vi day la cross-entropy tren cac
                       bin roi rac theo don vi grid cell).
        """
        # Neu khong co positive anchor nao, tra ve loss = 0 nhung van giu graph
        # (nhan voi 0) de tranh loi khi backward.
        if fg_mask.sum() == 0:
            return pred_bboxes.sum() * 0, pred_dist.sum() * 0

        # target_scores.sum(-1): [bs, A]
        # Moi Positive Anchor chi thuoc mot class nen phep sum() chinh la Soft
        # Weight do TaskAlignedAssigner sinh ra. Chi lay cac Positive Anchor de
        # lam trong so khi tinh CIoU Loss va DFL Loss.
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # [N_pos, 1] - bat nguon tu chat luong Assignment giua Classification va
        # Localization (cong thuc m trong paper TAL).

        # CIoU giua box du doan va box target, ca hai deu [GRID] xyxy (ket qua
        # CIoU khong doi neu tinh o PIXEL, nhung o day nhat quan dung GRID).
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou).unsqueeze(-1) * weight).sum() / target_scores_sum

        # Encode target_bboxes [GRID] xyxy -> target_ltrb [GRID] ltrb (so thuc,
        # da clamp trong [0, reg_max-1-eps]) de lam nhan cho DFL.
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)  # [bs, A, 4], [GRID], ltrb

        # pred_dist: (bs, A, 4*reg_max) [GRID] logits; fg_mask: (bs, A)
        # -> chi lay cac Positive Anchor, reshape ve (N_pos*4, reg_max) de
        #    tinh cross-entropy cho tung canh (l/t/r/b) nhu mot bai toan
        #    phan loai reg_max lop.
        loss_dfl = self._df_loss(
            pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]
        ) * weight

        loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Distribution Focal Loss: quy khoang cach thuc (so thuc, don vi GRID
        cell) ve hai bin nguyen lien ke gan nhat va noi suy cross-entropy giua
        chung, giup mo hinh hoc phan phoi xac suat lien tuc thay vi hoi quy
        truc tiep 1 gia tri.

        Dau vao:
            pred_dist: (N_pos*4, reg_max), [GRID], logits
                Logits cua phan phoi xac suat tren reg_max bin (gia tri bin =
                0, 1, ..., reg_max-1 don vi grid cell) cho tung canh (Left,
                Top, Right, Bottom) cua moi Positive Anchor. Moi hang tuong
                ung voi mot canh va la dau vao cua CrossEntropy.
            target: (N_pos, 4), [GRID], ltrb
                Khoang cach Ground Truth (l, t, r, b) tu Anchor Point den 4 canh
                cua Bounding Box, o dang so thuc (don vi grid cell). Moi hang
                tuong ung voi mot Positive Anchor.

        Tra ve:
            Tensor (N_pos, 1) - DFL loss trung binh tren 4 canh cua tung anchor.
        """
        tl = target.long()          # bin nguyen ben trai (floor), vi du target=3.7 -> tl=3
        tr = tl + 1                 # bin nguyen ben phai, vi du tr=4
        wl = tr - target            # trong so cho bin trai (gan target hon thi trong so lon hon), vi du wl=0.3
        wr = 1 - wl                 # trong so cho bin phai, vi du wr=0.7
        loss_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        loss_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (loss_l * wl + loss_r * wr).mean(-1, keepdim=True)


# ==============================================================================
# 4. DETECTION LOSS: ghep TAL + BboxLoss + BCE cho ca 2 nhanh o2m / o2o
#    Day la noi DIEN RA PHEP QUY DOI GIUA 2 KHONG GIAN: goi Assigner o [PIXEL],
#    sau do chia cho stride de chuyen sang [GRID] truoc khi goi BboxLoss.
# ==============================================================================
class DetectionLoss(nn.Module):
    """
    Tong hop loss cho ca hai nhanh du doan (o2m, o2o) cua NMSFreeDetector.
    Moi nhanh dung rieng mot TaskAlignedAssigner (topk khac nhau) nhung dung
    chung BboxLoss va ham BCE classification.
    """

    def __init__(
        self, nc, reg_max=16,
        topk_o2m=10, topk_o2o=1,
        alpha=0.5, beta=6.0,
        box_gain=7.5, cls_gain=1.0, dfl_gain=1.5,
        o2m_weight=1.0, o2o_weight=1.0,
    ):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.box_gain, self.cls_gain, self.dfl_gain = box_gain, cls_gain, dfl_gain
        self.o2m_weight, self.o2o_weight = o2m_weight, o2o_weight

        self.assigner_o2m = TaskAlignedAssigner(topk=topk_o2m, num_classes=nc, alpha=alpha, beta=beta)
        self.assigner_o2o = TaskAlignedAssigner(topk=topk_o2o, num_classes=nc, alpha=alpha, beta=beta)
        self.bbox_loss = BboxLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def preprocess_targets(self, targets, batch_size, device):
        """
        Chuyen doi targets tu list[dict] (moi anh mot so luong GT khac nhau)
        sang tensor co padding, de xu ly dong loat theo batch. Khong gian
        toa do GIU NGUYEN [PIXEL] xyxy nhu dau vao goc (khong quy doi gi ca).

        Vi du dau vao:
            targets = [
                {"boxes": torch.tensor([[30., 30., 120., 150.],
                                         [150., 100., 280., 260.]], device=device),
                 "labels": torch.tensor([1, 3], device=device)},
                {"boxes": torch.tensor([[40., 40., 100., 100.]], device=device),
                 "labels": torch.tensor([0], device=device)},
            ]

        Tra ve:
            gt_bboxes: (batch_size, n_max, 4), [PIXEL], xyxy - da padding bang 0.
            gt_labels: (batch_size, n_max, 1) long           - da padding bang 0.
            mask_gt  : (batch_size, n_max, 1) bool           - True tai vi tri GT that.
        """
        n_max = max((t["boxes"].shape[0] for t in targets), default=0)  # so object lon nhat trong batch
        n_max = max(n_max, 1)  # tranh tensor rong gay loi shape khi khong co GT nao trong ca batch
        gt_bboxes = torch.zeros(batch_size, n_max, 4, device=device)
        gt_labels = torch.zeros(batch_size, n_max, 1, dtype=torch.long, device=device)
        mask_gt = torch.zeros(batch_size, n_max, 1, dtype=torch.bool, device=device)

        for i, t in enumerate(targets):
            n = t["boxes"].shape[0]
            if n == 0:
                continue
            gt_bboxes[i, :n] = t["boxes"].to(device)      # [num_object, 4], [PIXEL], xyxy
            gt_labels[i, :n, 0] = t["labels"].to(device)   # [num_object]
            mask_gt[i, :n, 0] = True
        return gt_bboxes, gt_labels, mask_gt

    def _branch_loss(self, assigner, cls_raw, box_pixel, reg_raw, anchors, strides,
                      gt_bboxes, gt_labels, mask_gt):
        """
        Tinh loss cho MOT nhanh (o2m hoac o2o): chay TaskAlignedAssigner (o
        khong gian [PIXEL]) de gan target, sau do QUY DOI sang [GRID] roi
        tinh classification loss (BCE) va box loss (CIoU + DFL).

        Dau vao:
            assigner  : TaskAlignedAssigner tuong ung nhanh (o2m hoac o2o).
            cls_raw   : (bs, A, nc), logit classification, CHUA sigmoid
                        (khong lien quan toa do).
            box_pixel : (bs, A, 4), [PIXEL], xyxy
                        Box du doan da duoc head decode san (tu reg_raw +
                        anchors + strides) ve khong gian pixel cua anh goc.
            reg_raw   : (bs, 4*reg_max, A), [GRID], logit DFL truoc softmax
                        (kenh thu 2 la 4*reg_max, se transpose lai thanh
                        (bs, A, 4*reg_max) ben duoi).
            anchors   : (A, 2), [GRID]
                        Toa do tam anchor tren grid feature map (offset 0.5),
                        CHUA nhan stride.
            strides   : (A, 1)
                        He so quy doi GRID -> PIXEL cho tung anchor (vi du
                        8/16/32 tuy anchor thuoc level feature map nao).
            gt_bboxes : (bs, M, 4), [PIXEL], xyxy (xem preprocess_targets).
            gt_labels : (bs, M, 1).
            mask_gt   : (bs, M, 1).

        Tra ve:
            loss_iou : Regression Loss dua tren CIoU (tinh o [GRID]).
            loss_cls : Classification Loss (BCE).
            loss_dfl : Distribution Focal Loss (tinh o [GRID]).
            n_pos    : So luong Positive Anchor sau Assignment, phuc vu thong ke/log.
        """
        stride_b = strides.unsqueeze(0)               # (1, A, 1) - de broadcast voi (bs, A, 4)
        anchors_pixel = anchors * strides              # (A, 2), [GRID] -> [PIXEL]: anchor_pixel = anchor_grid * stride

        pred_dist = reg_raw.transpose(1, 2).contiguous()  # (bs, 4*reg_max, A) -> (bs, A, 4*reg_max), van [GRID]

        # Assigner khong lan truyen gradient -> dung ban sao da detach + sigmoid
        with torch.no_grad():
            pd_scores_sig = cls_raw.detach().sigmoid()

        # Goi Assigner: TAT CA cac box (pd_bboxes, anc_points, gt_bboxes) deu
        # phai o [PIXEL] - day chinh la ly do can anchors_pixel thay vi anchors.
        target_labels, target_bboxes_pixel, target_scores, fg_mask, _ = assigner(
            pd_scores_sig, box_pixel.detach(), anchors_pixel,
            gt_labels, gt_bboxes, mask_gt,
        )
        # target_bboxes_pixel: (bs, A, 4), [PIXEL], xyxy

        target_scores_sum = max(target_scores.sum().item(), 1)  # tong trong so cua toan bo Positive Anchor trong batch

        # --- classification loss (tinh tren toan bo anchor, ca duong lan am) ---
        # Khong lien quan toa do nen khong can quan tam khong gian PIXEL/GRID.
        loss_cls = self.bce(cls_raw, target_scores).sum() / target_scores_sum

        # --- box + dfl loss (chi tinh tren anchor duong), quy ve khong gian [GRID] ---
        # Day la buoc QUY DOI KHONG GIAN then chot: PIXEL -> GRID bang cach
        # chia cho stride cua tung anchor (GRID = PIXEL / stride).
        pred_bboxes_grid = box_pixel / stride_b            # (bs, A, 4), [GRID], xyxy
        target_bboxes_grid = target_bboxes_pixel / stride_b  # (bs, A, 4), [GRID], xyxy
        # Sau buoc nay, pred_bboxes_grid / target_bboxes_grid / anchors / pred_dist
        # deu da CUNG mot khong gian [GRID], sẵn sang truyen vao BboxLoss.

        loss_iou, loss_dfl = self.bbox_loss(
            pred_dist, pred_bboxes_grid, anchors, target_bboxes_grid,
            target_scores, target_scores_sum, fg_mask,
        )

        n_pos = fg_mask.sum().item()

        return loss_iou, loss_cls, loss_dfl, n_pos

    def forward(self, preds, targets):
        """
        Tinh tong loss cua model tren ca hai nhanh o2m va o2o.

        Dau vao:
            preds : dict tra ve tu DetectHead o che do train:
                {                     (bs, A, nc)      (bs, A, 4) [PIXEL] xyxy   (bs, 4*reg_max, A) [GRID] logit
                    "o2m": {"cls": o2m_cls, "box": o2m_box, "reg_raw": o2m_reg},
                    "o2o": {"cls": o2o_cls, "box": o2o_box, "reg_raw": o2o_reg},
                    "anchors": anchors,   # (A, 2), [GRID], offset 0.5, CHUA nhan stride
                    "strides": stride_t,  # (A, 1), he so quy doi GRID -> PIXEL
                }
            targets : list[dict] do dai = batch_size:
                {"boxes": (N,4), [PIXEL], xyxy, "labels": (N,)}

        Tra ve:
            total : Tensor scalar co grad - tong loss dung de backward.
            items : dict[str, float | int] - cac thanh phan loss (da .item(),
                    dung de log, TensorBoard, WandB, progress bar, benchmark).
        """
        device = preds["anchors"].device
        batch_size = preds["o2o"]["cls"].shape[0]
        anchors, strides = preds["anchors"], preds["strides"]

        gt_bboxes, gt_labels, mask_gt = self.preprocess_targets(targets, batch_size, device)
        # gt_bboxes: [batch_size, n_max, 4], [PIXEL], xyxy
        # gt_labels: [batch_size, n_max, 1]
        # mask_gt  : [batch_size, n_max, 1]

        iou_m, cls_m, dfl_m, npos_m = self._branch_loss(
            self.assigner_o2m, preds["o2m"]["cls"], preds["o2m"]["box"], preds["o2m"]["reg_raw"],
            anchors, strides, gt_bboxes, gt_labels, mask_gt,
        )

        iou_o, cls_o, dfl_o, npos_o = self._branch_loss(
            self.assigner_o2o, preds["o2o"]["cls"], preds["o2o"]["box"], preds["o2o"]["reg_raw"],
            anchors, strides, gt_bboxes, gt_labels, mask_gt,
        )

        loss_o2m = self.box_gain * iou_m + self.cls_gain * cls_m + self.dfl_gain * dfl_m
        loss_o2o = self.box_gain * iou_o + self.cls_gain * cls_o + self.dfl_gain * dfl_o
        total = self.o2m_weight * loss_o2m + self.o2o_weight * loss_o2o

        # items: dictionary luu cac chi so (metrics) phuc vu ghi log, TensorBoard,
        # WandB, progress bar hoac benchmark. Cac gia tri deu da detach khoi
        # computation graph.
        items = {
            # Tong loss cuoi cung sau khi ket hop O2M va O2O.
            "loss": total.detach().item(),                # float

            # Tong loss cua tung nhanh.
            "loss_o2m": loss_o2m.detach().item(),          # float
            "loss_o2o": loss_o2o.detach().item(),          # float

            # Cac thanh phan loss cua nhanh One-to-Many.
            "o2m/iou": iou_m.detach().item(),               # float - CIoU Loss
            "o2m/cls": cls_m.detach().item(),               # float - BCE Classification Loss
            "o2m/dfl": dfl_m.detach().item(),               # float - Distribution Focal Loss

            # Cac thanh phan loss cua nhanh One-to-One.
            "o2o/iou": iou_o.detach().item(),               # float - CIoU Loss
            "o2o/cls": cls_o.detach().item(),               # float - BCE Classification Loss
            "o2o/dfl": dfl_o.detach().item(),               # float - Distribution Focal Loss

            # So luong Positive Anchor sau Assignment cua tung nhanh.
            "o2m/n_pos": npos_m,                            # int
            "o2o/n_pos": npos_o,                            # int
        }
        return total, items