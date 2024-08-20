import math
import torch
from torch.nn.functional import cross_entropy, one_hot
import time
import torchvision
import numpy as np

from summary import Summary


class ComputeLoss:
    def __init__(self, model, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5):
        super().__init__()
        
        if hasattr(model, 'module'):
            model = model.module
        
        device = next(model.parameters()).device  # get model device
        
        m = model.head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device

        # task aligned assigner
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9

        self.bs = 1
        self.num_max_boxes = 0
        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)
        
    
    # outputs shape: 3 * (batch_size, 144, 80, 80)
    def __call__(self, outputs, targets):    
        x = outputs
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)
        
        pred_output = pred_output.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]

        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)

        # targets
        # no bounding box in target
        if targets.shape[0] == 0: 
            gt = torch.zeros(pred_scores.shape[0], 0, 5, device=self.device)
        # have bounding box in target
        else: 
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 5, device=self.device)
            for j in range(pred_scores.shape[0]):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]
            #gt[..., 1:5] = wh2xy_center(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))
            gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))

        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # boxes
        b, a, c = pred_output.shape
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        a, b = torch.split(pred_bboxes, 2, -1)
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)
        
        scores = pred_scores.detach().sigmoid()
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask = self.assign(
            scores, bboxes, gt_labels, gt_bboxes, mask_gt, anchor_points * stride_tensor
        )

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()
        # cls loss
        loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        loss_cls = loss_cls.sum() / target_scores_sum

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            # IoU loss
            weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum
            # DFL loss
            a, b = torch.split(target_bboxes, 2, -1)
            target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)  # distance (left_top, right_bottom)
            loss_dfl = self.df_loss(pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask])
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        loss_cls *= self.cls_gain
        loss_box *= self.box_gain
        loss_dfl *= self.dfl_gain
        return loss_cls + loss_box + loss_dfl
    
    
    @torch.no_grad()
    # Task-aligned One-stage Object Detection assigner
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors):
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)

        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))

        i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        i[1] = true_labels.long().squeeze(-1)

        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        overlaps = overlaps.squeeze(3).clamp(0)
        align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta)
        bs, n_boxes, _ = true_bboxes.shape
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        metrics = align_metric * mask_in_gts
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool()
        num_anchors = metrics.shape[-1]
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)
        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * true_mask

        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        # assigned target labels, (b, 1)
        batch_index = torch.arange(
            end=self.bs, dtype=torch.int64, device=true_labels.device
        )[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes
        target_labels = true_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = one_hot(target_labels, self.nc)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2)
        norm_align_metric = norm_align_metric.unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        
        return target_bboxes, target_scores, fg_mask.bool()
    
    
    @staticmethod
    #@torch.no_grad
    # Return sum of left and right DFL losses
    def df_loss(pred_dist, target):
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)
    
    
    @staticmethod
    #@torch.no_grad
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
    def iou(box1, box2, eps=1e-7):
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - intersection + eps

        # IoU
        iou = intersection / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        
        # convex diagonal squared
        c2 = cw ** 2 + ch ** 2 + eps  
        # center dist ** 2
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        alpha = v / (v - iou + (1 + eps))
        
        return iou - (rho2 / c2 + v * alpha) # CIoU
    
    
# Generate anchors from features
def make_anchors(x, strides, offset=0.5):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


# convert from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
@torch.no_grad()
def wh2xy(x):
    y = x.clone()
    y[..., 2] = x[..., 0] + x[..., 2] # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] # bottom right y
    return y


# convert from [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]
@torch.no_grad()
def wh2xy_center(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


# convert from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
@torch.no_grad()
def xy2wh(x):
    y = np.copy(x)
    y[:, 2] = (x[:, 2] - x[:, 0])  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) # height
    return y


@torch.no_grad()
def non_max_suppression(
        prediction, 
        conf_threshold=0.25, 
        iou_threshold=0.45,
        max_wh = 7680, # (pixels) maximum box width and height
        max_det = 300,  # the maximum number of boxes to keep after NMS
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    ):
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy_center(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)[:max_det]  # NMS
        outputs[index] = x[keep]
        
        if (time.time() - start) > 0.5 + 0.05 * prediction.shape[0]:
            print(f'WARNING: NMS time limit {0.5 + 0.05 * prediction.shape[0]:.3f}s exceeded')
            break  # time limit exceeded

    return outputs



def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


# cross stage partial networks
class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


# spatial pyramid pooling
class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0])]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1])]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2])]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


# Dark Feature Pyramid Network
class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


# Distribution Focal Loss
class DFL(torch.nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(
            Conv(x, c1, 3),
            Conv(c1, c1, 3),
            torch.nn.Conv2d(c1, self.nc, 1)) for x in filters
        )
        self.box = torch.nn.ModuleList(torch.nn.Sequential(
            Conv(x, c2, 3),
            Conv(c2, c2, 3),
            torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters
        )
    
    
    def get_stride(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
        return x
    
    
    def forward(self, x, dummy):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
        
        # calculate stride
        if dummy:
            return x
        
        output = list(x) # make a copy
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        box = torch.cat((box * self.strides, cls.sigmoid()), 1)
        return output, box
    

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)



# main yolo class
class YOLO(torch.nn.Module):
    def __init__(self, size='m', num_classes=80):
        super().__init__()
        
        # model size parameters
        if size == 'n':
            depth = [1, 2, 2]
            width = [3, 16, 32, 64, 128, 256]
            
        elif size == 's':
            depth = [1, 2, 2]
            width = [3, 32, 64, 128, 256, 512]
        
        elif size == 'm':
            depth = [2, 4, 4]
            width = [3, 48, 96, 192, 384, 576]
            
        elif size == 'l':
            depth = [3, 6, 6]
            width = [3, 64, 128, 256, 512, 512]
            
        elif size == 'x':
            depth = [3, 6, 6]
            width = [3, 80, 160, 320, 640, 640]
            
            
        self.size = size
        
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy, dummy=True)])
        self.stride = self.head.stride
        self.head.initialize_biases()
    
    
    def forward(self, x, dummy=False):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x), dummy)
    
    
    def name(self):
        return 'yolo_v8_' + self.size
    
    
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
    
    
    
def main():    
    device = 'cuda'
    
    # Creating model and testing output shapes 
    model = YOLO(size='s', num_classes=80) 
    model = model.to(device)
    Summary(model)

    
    
if __name__ == "__main__": 
    main()
    