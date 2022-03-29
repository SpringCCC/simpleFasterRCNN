import numpy as np
import torch



def toNumpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def toTensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).cuda()
    return x


def loc2bbox(loc, bbox):
    # loc: dy, dx, dh, dw,注意这里的dy，dx是相对于中心点
    # bbox: (y1 x1 y2 x2)
    dst_bbox = np.zeros(bbox.shape, dtype=loc.dtype)
    cy, cx = (bbox[:, 2] + bbox[:, 0])/2, (bbox[:, 3] + bbox[:, 1])/2
    h, w = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    dst_cy = h * loc[:, 0] + cy
    dst_cx = w * loc[:, 1] + cx
    dst_h = np.exp(loc[:, 2]) * h
    dst_w = np.exp(loc[:, 3]) * w
    dst_bbox[:, 0] = dst_cy - dst_h / 2
    dst_bbox[:, 1] = dst_cx - dst_w / 2
    dst_bbox[:, 2] = dst_cy + dst_h / 2
    dst_bbox[:, 3] = dst_cx + dst_w / 2
    return dst_bbox

def bbox2loc(anchor, gt_box):
    assert anchor.shape==gt_box.shape
    loc = np.zeros(anchor.shape, dtype=anchor.dtype)
    anchor_cy, anchor_cx = (anchor[:, 0]+ anchor[:, 2])/2, (anchor[:, 1]+ anchor[:, 3])/2
    anchor_h, anchor_w = anchor[:, 2] - anchor[:, 0], anchor[:, 3] - anchor[:, 1]
    gt_box_cy, gt_box_cx = (gt_box[:, 0]+ gt_box[:, 2])/2, (gt_box[:, 1]+ gt_box[:, 3])/2
    gt_box_h, gt_box_w = gt_box[:, 2] - gt_box[:, 0], gt_box[:, 3] - gt_box[:, 1]
    # dy = (y_gt - y)/h; dh = log(h_gt/h)
    loc[:, 0] = (gt_box_cy - anchor_cy) / anchor_h
    loc[:, 1] = (gt_box_cx - anchor_cx) / anchor_w
    loc[:, 2] = np.log(gt_box_h / anchor_h)
    loc[:, 3] = np.log(gt_box_w / anchor_w)
    return loc






def twobox_iou(bbox1, bbox2):
    tl = np.maximum(bbox1[:, None, :2], bbox2[None, :, :2])
    br = np.minimum(bbox1[:, None, 2:], bbox2[None, :, 2:])
    inter = np.prod(br - tl, axis=2) * (tl < br).all(axis=2) # N*K
    area1 = np.prod(bbox1[:, 2:] - bbox1[:, :2], axis=1)
    area2 = np.prod(bbox2[:, 2:] - bbox2[:, :2], axis=1)
    iou = inter / (area1[: ,None] + area2[None, :] - inter)
    return iou


def cycleconvert_y1x1y2x2_x1y1x2y2(bbox):
    newbbox = np.zeros(bbox.shape, dtype=bbox.dtype)
    newbbox[:, 0] = bbox[:, 1]
    newbbox[:, 1] = bbox[:, 0]
    newbbox[:, 2] = bbox[:, 3]
    newbbox[:, 3] = bbox[:, 2]
    return newbbox

def get_base_anchor(ratio, scale, stride):
    base_anchor = []
    cy, cx = stride / 2, stride / 2
    for r in ratio:
        for s in scale:
            h = stride * s * np.sqrt(r)
            w = stride * s * np.sqrt(1 / r)
            base_anchor.append([cy-h/2, cx-w/2, cy+h/2, cx+w/2])
    return np.asarray(base_anchor)

def get_all_anchor(base_anchor, h, w, stride):
    y = np.asarray([i * stride for i in range(h)])
    x = np.asarray([i * stride for i in range(w)])
    shiftx, shifty = np.meshgrid(x, y)
    shiftx, shifty = shiftx.flatten(), shifty.flatten()
    shift = np.stack([shifty, shiftx, shifty, shiftx], axis=1)
    N, K = len(shift), len(base_anchor)
    anchor = shift.reshape(N, 1, 4) + base_anchor.reshape(1, K, 4)
    anchor = anchor.reshape(-1, 4)
    return anchor

def init_weight(ms, mean, std):
    for m in ms:
        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(0)

