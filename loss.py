import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLoss(nn.Module):
    def __init__(self, num_classes=80, lambda_box=7.5, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.nc = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        
        # BCE loss for objectness and classification
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
        
    def forward(self, preds, targets):
        """
        preds: list of 3 tensors, each [bs, 64+80, h, w] for different scales
               First 64 channels: bounding box coordinates (4 * 16 for DFL)
               Last 80 channels: class probabilities
        targets: list of tensors, one per image in batch
                 Each tensor is [num_objects, 5] containing [class_id, cx, cy, w, h]
                 All coordinates are normalized to [0, 1]
        """
        device = preds[0].device
        batch_size = preds[0].shape[0]
        
        # Initialize losses
        loss_box = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        
        # Get feature map sizes and create targets for each scale
        for scale_idx, pred in enumerate(preds):
            bs, channels, h, w = pred.shape
            
            # Split predictions: box (64 channels) and class (80 channels)
            pred_box = pred[:, :64, :, :]  # [bs, 64, h, w]
            pred_cls = pred[:, 64:, :, :]   # [bs, 80, h, w]
            
            # Reshape to [bs, h, w, channels] for easier processing
            pred_box = pred_box.permute(0, 2, 3, 1).contiguous()  # [bs, h, w, 64]
            pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous()  # [bs, h, w, 80]
            
            # Create objectness target (1 if object present, 0 otherwise)
            obj_target = torch.zeros(bs, h, w, 1, device=device)
            
            # Process each image in batch
            for b in range(batch_size):
                if len(targets[b]) == 0:
                    continue
                    
                target = targets[b]  # [num_objects, 5]
                
                # Get ground truth boxes for this image
                gt_cls = target[:, 0].long()  # [num_objects]
                gt_boxes = target[:, 1:]       # [num_objects, 4] - cx, cy, w, h
                
                # Convert normalized coordinates to grid coordinates
                # cx, cy are in [0, 1], scale to [0, w] and [0, h]
                gt_cx = gt_boxes[:, 0] * w
                gt_cy = gt_boxes[:, 1] * h
                gt_w = gt_boxes[:, 2] * w
                gt_h = gt_boxes[:, 3] * h
                
                # Find which grid cell each object belongs to
                grid_x = gt_cx.long().clamp(0, w - 1)
                grid_y = gt_cy.long().clamp(0, h - 1)
                
                # For each ground truth object
                for obj_idx in range(len(target)):
                    gx = grid_x[obj_idx]
                    gy = grid_y[obj_idx]
                    cls_id = gt_cls[obj_idx]
                    
                    # Mark this grid cell as containing an object
                    obj_target[b, gy, gx, 0] = 1.0
                    
                    # Create box target (simplified - using center offset and size)
                    # In real YOLO, you'd compute proper IoU-based assignment
                    box_target = torch.zeros(64, device=device)
                    
                    # Compute offset within grid cell
                    offset_x = gt_cx[obj_idx] - gx.float()
                    offset_y = gt_cy[obj_idx] - gy.float()
                    
                    # Simple box encoding (you can improve this)
                    # For now, use first 4 channels as cx, cy, w, h offsets
                    box_target[0] = offset_x
                    box_target[1] = offset_y
                    box_target[2] = gt_w[obj_idx] / w
                    box_target[3] = gt_h[obj_idx] / h
                    
                    # Box loss (MSE for simplicity, can use IoU/GIoU)
                    pred_box_cell = pred_box[b, gy, gx, :]
                    loss_box += F.mse_loss(pred_box_cell[:4], box_target[:4])
                    
                    # Class loss
                    cls_target = torch.zeros(self.nc, device=device)
                    cls_target[cls_id] = 1.0
                    pred_cls_cell = pred_cls[b, gy, gx, :]
                    loss_cls += self.bce_cls(pred_cls_cell, cls_target)
            
            # Objectness loss (all grid cells)
            # Create a pseudo objectness prediction from box confidence
            pred_obj = pred_box[..., :4].mean(dim=-1, keepdim=True).sigmoid()
            loss_obj += F.mse_loss(pred_obj, obj_target)
        
        # Normalize losses by number of scales
        num_scales = len(preds)
        loss_box = loss_box / num_scales
        loss_obj = loss_obj / num_scales
        loss_cls = loss_cls / num_scales
        
        # Weighted total loss
        total_loss = (self.lambda_box * loss_box + 
                     self.lambda_obj * loss_obj + 
                     self.lambda_cls * loss_cls)
        
        return total_loss

"""
Ignore ImprovedYoloLoss and would be implemented separately when we move to detections.
"""


class ImprovedYoloLoss(nn.Module):
    """
    More sophisticated YOLO loss with IoU-based assignment
    """
    def __init__(self, num_classes=80, lambda_box=7.5, lambda_obj=1.0, lambda_cls=0.5, 
                 iou_threshold=0.5):
        super().__init__()
        self.nc = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.iou_threshold = iou_threshold
        
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, preds, targets):
        """
        Enhanced loss with proper anchor assignment
        """
        device = preds[0].device
        batch_size = preds[0].shape[0]
        
        loss_box = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        
        total_targets = 0
        
        for scale_idx, pred in enumerate(preds):
            bs, channels, h, w = pred.shape
            stride = 2 ** (5 - scale_idx)  # Stride: 32, 16, 8 for 3 scales
            
            # Split predictions
            pred_box = pred[:, :64, :, :].permute(0, 2, 3, 1)  # [bs, h, w, 64]
            pred_cls = pred[:, 64:, :, :].permute(0, 2, 3, 1)  # [bs, h, w, nc]
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_y = grid_y.to(device).float()
            grid_x = grid_x.to(device).float()
            
            # Process each image
            for b in range(batch_size):
                if len(targets[b]) == 0:
                    continue
                
                target = targets[b]
                gt_cls = target[:, 0].long()
                gt_boxes = target[:, 1:]  # [num_obj, 4] in normalized coords
                
                # Scale to current feature map
                gt_cx = gt_boxes[:, 0] * w
                gt_cy = gt_boxes[:, 1] * h
                gt_w = gt_boxes[:, 2] * w
                gt_h = gt_boxes[:, 3] * h
                
                # Assign to grid cells
                grid_x_idx = gt_cx.long().clamp(0, w - 1)
                grid_y_idx = gt_cy.long().clamp(0, h - 1)
                
                for obj_idx in range(len(target)):
                    gx = grid_x_idx[obj_idx]
                    gy = grid_y_idx[obj_idx]
                    cls_id = gt_cls[obj_idx]
                    
                    # Box target
                    tx = gt_cx[obj_idx] - gx.float()
                    ty = gt_cy[obj_idx] - gy.float()
                    tw = torch.log(gt_w[obj_idx] + 1e-16)
                    th = torch.log(gt_h[obj_idx] + 1e-16)
                    
                    # Get predictions for this cell
                    pred_box_cell = pred_box[b, gy, gx, :4]
                    pred_cls_cell = pred_cls[b, gy, gx, :]
                    
                    # Box loss (simplified - first 4 channels)
                    box_target = torch.tensor([tx, ty, tw, th], device=device)
                    loss_box += F.smooth_l1_loss(pred_box_cell, box_target)
                    
                    # Classification loss
                    cls_target = torch.zeros(self.nc, device=device)
                    cls_target[cls_id] = 1.0
                    loss_cls += self.bce_cls(pred_cls_cell, cls_target).sum()
                    
                    # Objectness loss (positive sample)
                    obj_target = torch.ones(1, device=device)
                    pred_obj = pred_box_cell[0].unsqueeze(0)  # Use first channel as objectness
                    loss_obj += self.bce_obj(pred_obj, obj_target).sum()
                    
                    total_targets += 1
            
            # Negative objectness loss (no object in cell)
            # Sample some negative cells to balance
            num_neg_samples = min(h * w * batch_size // 10, 100)
            neg_mask = torch.rand(bs, h, w, device=device) > 0.9
            if neg_mask.sum() > 0:
                neg_pred = pred_box[neg_mask, 0]  # First channel as objectness
                neg_target = torch.zeros_like(neg_pred)
                loss_obj += self.bce_obj(neg_pred, neg_target).sum() * 0.1
        
        # Normalize
        if total_targets > 0:
            loss_box = loss_box / total_targets
            loss_cls = loss_cls / total_targets
            loss_obj = loss_obj / max(total_targets, 1)
        
        total_loss = (self.lambda_box * loss_box + 
                     self.lambda_obj * loss_obj + 
                     self.lambda_cls * loss_cls)
        
        return total_loss


# Helper functions for box operations
def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes
    box1, box2: [N, 4] in format [x1, y1, x2, y2]
    """
    # Intersection
    x1 = torch.max(box1[:, None, 0], box2[:, 0])
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    
    return inter / (union + 1e-6)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False):
    """
    Calculate IoU/GIoU/DIoU/CIoU
    box1, box2: [N, 4] 
    """
    if xywh:
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    # Union
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + 1e-16
    
    iou = inter / union
    
    if GIoU or DIoU or CIoU:
        # Convex hull
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        
        if GIoU:
            c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-16
            return iou - (c_area - union) / c_area
        
        if DIoU or CIoU:
            c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + 1e-16
            rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + 
                    (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
            
            if DIoU:
                return iou - rho2 / c_diag
            
            if CIoU:
                v = (4 / (torch.pi ** 2)) * torch.pow(
                    torch.atan(w2 / (h2 + 1e-16)) - torch.atan(w1 / (h1 + 1e-16)), 2)
                alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c_diag + v * alpha)
    
    return iou