import torch
import torch.nn as nn
import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#importing the classes
try:
    from model import Conv, C2f, SPPF, Head, Neck
    from loss import MyLoss
    from dataset import YoloMineDataset
except Exception as e:
    print(f"Error in importing⚠️... please try again ")


#Creating Backbone DarkNet53
class BackBone(nn.Module):
    def __init__(self, in_ch = 3, shortcut = True):
        super().__init__()
        self.in_ch = in_ch
        self.shortcut = shortcut
        self.d, self.w, self.r = 1/3, 1.0, 2.0 # 'version n'
        
        def C(v): return max(1, int(v))
        #Conv layers
        self.conv_0 = Conv(self.in_ch, C(64*self.w), kernel=3, stride=2, padding=1)
        self.conv_1 = Conv(C(64*self.w), C(64*self.w), kernel=3, stride=2, padding=1)
        self.conv_3 = Conv(C(64*self.w), C(128*self.w), kernel=3, stride=2, padding=1)
        self.conv_5 = Conv(C(128*self.w), C(256*self.w), kernel=3, stride=2, padding=1)
        self.conv_7 = Conv(C(256*self.w), C(512*self.w * self.r), kernel=3, stride=2, padding=1)
        
        #C2f layers
        self.c2f_2 = C2f(C(64*self.w), C(64*self.w), num_bottlenecks=max(1, int(3*self.d)))
        self.c2f_4 = C2f(C(128*self.w), C(128*self.w), num_bottlenecks=max(1, int(6*self.d)))  
        self.c2f_6 = C2f(C(256*self.w), C(256*self.w), num_bottlenecks=max(1, int(9*self.d)))
        self.c2f_8 = C2f(C(512*self.w * self.r), C(512*self.w * self.r), num_bottlenecks=max(1, int(3*self.d)))

        #SPPF
        self.sppf = SPPF(C(512*self.w * self.r), C(512*self.w * self.r))
    
    def forward(self, x):
        x = self.conv_0(x)
        
        x = self.conv_1(x)
        x = self.c2f_2(x)
        
        x = self.conv_3(x)
        out1 = self.c2f_4(x)

        x = self.conv_5(out1)
        out2 = self.c2f_6(x)

        x = self.conv_7(out2) #keeping these for output
        x = self.c2f_8(x)

        out3 = self.sppf(x)

        return out1, out2, out3
    

class Yolo_Mine(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackBone()
        self.neck = Neck()
        self.head = Head()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0], x[1], x[2])
        return self.head(list(x))
    
    def train(self, mode = True):
        return super().train(mode)
    def eval(self):
        return super().eval()

def box_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def yolo_to_coco_format(preds, image_ids, class_names):
    """
    Convert YOLO model predictions to COCO-style results JSON.
    
    preds: list of [N, 6] tensors → (x1, y1, x2, y2, conf, cls)
    image_ids: list of image IDs corresponding to preds
    class_names: list of all class names in the dataset
    """
    coco_results = []
    
    for i, det in enumerate(preds):
        if isinstance(det, torch.Tensor):
            det = det.detach().cpu().numpy()
        image_id = int(image_ids[i])
        
        for *bbox, conf, cls_id in det:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            coco_results.append({
                "image_id": image_id,
                "category_id": int(cls_id),  # must match category_id in annotations
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(conf),
            })
    
    return coco_results
    
def decode_preds(pred, conf_thresh=0.50):
    """
    pred: (batch, num_boxes, 5 + num_classes)
    Returns a tensor [x1, y1, x2, y2, conf, cls_id]
    """
    if pred.dim() == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)

    pred = pred.clone()
    pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])  # center xy
    pred[..., 4:] = torch.sigmoid(pred[..., 4:])     # conf + classes

    boxes = pred.clone()
    boxes[..., 0] = pred[..., 0] - pred[..., 2] / 2
    boxes[..., 1] = pred[..., 1] - pred[..., 3] / 2
    boxes[..., 2] = pred[..., 0] + pred[..., 2] / 2
    boxes[..., 3] = pred[..., 1] + pred[..., 3] / 2

    boxes = boxes.view(-1, boxes.shape[-1])
    mask = boxes[:, 4] > conf_thresh
    if mask.sum() == 0:
        return torch.empty((0, 6), device=pred.device)

    boxes = boxes[mask]
    class_scores = boxes[:, 5:]
    cls_conf, cls_ids = torch.max(class_scores, dim=1)
    out = torch.cat([
        boxes[:, 0:4],
        (boxes[:, 4:5] * cls_conf.unsqueeze(1)),  # combined confidence
        cls_ids.unsqueeze(1)
    ], dim=1)
    return out


def draw_preds(image, detections, class_names, conf_threshold=0.25, fill=True):
    """
    Draw highlighted damage regions (border + semi-transparent overlay) 
    instead of bounding boxes on detections.

    Args:
        image (np.ndarray): Original image (BGR)
        detections (list): Each detection as [x1, y1, x2, y2, conf, cls_id]
        class_names (list): List of class names
        conf_threshold (float): Minimum confidence to draw
        fill (bool): Whether to fill region with transparency
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:  # batch of images
            image = image[0]  # take first image in the batch
        image = image.detach().cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)


    overlay = image.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det

        if conf < conf_threshold:
            continue
        
        # Convert to int
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        
        if cls_id < 0 or cls_id >= len(class_names):
            continue

        # Define colors
        color = (0, 0, 255)  # red

        # Extract region of interest (ROI)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        
        # Edge detection inside bounding box to find contours of damage
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 150)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Offset contours to full-image coordinates
        for cnt in contours:
            cnt += [x1, y1]  # shift contour to original image position
            area = cv2.contourArea(cnt)
            if area > 80:
                cv2.drawContours(image, [cnt], -1, color, 2)
                if fill:
                    cv2.drawContours(overlay, [cnt], -1, color, -1)
    
    # Blend overlay if fill enabled
    if fill:
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    
    return image


def save_preds(image, detections, class_names, base_name, save_dir="output/inference"):
    """
    Save the highlighted prediction image and detection results.
    Compatible with draw_preds() that highlights regions instead of boxes.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save image with highlights
    img_path = os.path.join(save_dir, f"{base_name}.jpg")
    cv2.imwrite(img_path, image)

    # Save detections as .txt
    txt_path = os.path.join(save_dir, f"{base_name}.txt")
    with open(txt_path, "w") as f:
        for det in detections:
            # Expected format: [x1, y1, x2, y2, conf, cls_id]
            if len(det) < 6:
                continue  # skip malformed detections

            x1, y1, x2, y2, conf, cls_id = det

            # Clamp or round values
            cls_id = int(cls_id)
            conf = float(conf)

            if cls_id < 0 or cls_id >= len(class_names):
                continue

            f.write(f"{class_names[cls_id]} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf:.4f}\n")

    print(f"[✔️] Saved highlighted image and detections: {base_name}.jpg / {base_name}.txt")

def load_coco_classes(json_path):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    categories = coco.get("categories", [])
    #Sort categories by ID
    categories = sorted(categories, key = lambda x: x["id"])
    class_names = [cat["name"] for cat in categories]
    return class_names


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    model = Yolo_Mine().to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6} million parameters")
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_conf = {
        "model_name" : model.__class__.__name__,
        "layers" : str(model),
        "total_params" : sum(p.numel() for p in model.parameters()),
        "trainable params" : sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    with open("model_config.json", "w") as f:
        json.dump(model_conf, f, indent=4)

    #train_loader = 
    model_loss = MyLoss(num_classes=80, lambda_box=7.5, lambda_obj=1.0, lambda_cls=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
    num_epochs = 10

    train_dataset = YoloMineDataset(
        root_dir="data/train",
        json_path="data/train/COCO.json",
        split = "train"
    )

    val_dataset = YoloMineDataset(
        root_dir="data/val",
        json_path="data/val/COCO.json",
        split = "val"
    )

    test_dataset = YoloMineDataset(root_dir="data/test", split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn= lambda x: tuple(zip(*x)),
        num_workers=0
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    total_train_loss = []
    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1} / {num_epochs}]")
        train_loss = 0.0
        

        for imgs, targets in loop:
            imgs = torch.stack(imgs).to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            #Making predictionss
            preds = model(imgs)

            #Computing Yolo Loss
            loss = model_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            loop.set_postfix(loss = loss.item())
        
        print(f"Epoch {epoch + 1} | Train_Loss: {train_loss/len(train_loader): .4f}")
        total_train_loss.append(train_loss/len(train_loader))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(total_train_loss) + 1), total_train_loss, label='Train Loss', color='dodgerblue', linewidth = 2)
    plt.title('Training Loss')
    plt.xlabel('Range')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    #Now validation
    val_loss = 0.0
       # Directory for saving results
    os.makedirs("output/coco_val", exist_ok=True)
    results_path = "output/coco_val/val_predictions.json"

    results = []
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(val_loader):
            imgs = torch.stack(imgs).to(device)
            targets = [t.to(device) for t in targets]
            
            model.train()
            preds = model(imgs)
            loss = model_loss(preds, targets)
            val_loss += loss.item()

            for i, pred in enumerate(preds):
                detections = decode_preds(pred, conf_thresh=0.25)  # lower threshold
            if detections.numel() == 0:
                continue

            # Convert detections to COCO format: [x, y, width, height]
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det.tolist()
                width, height = x2 - x1, y2 - y1

                # Safety: match your COCO annotation IDs
                img_id = i

                results.append({
                    "image_id": img_id,
                    "category_id": int(cls_id),  # ensure matches COCO’s category_ids
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(conf)
                })
    val_loss /= len(val_loader)
    val_coco_path = "data/val/COCO.json"
    print(f"\n✅ Validation Loss: {val_loss:.4f}")
    print(f"Total Detections: {len(results)}")

    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"[✔️] Saved COCO-style predictions → {results_path}")

    # Load ground-truth and detections
    if os.path.exists(val_coco_path) and os.path.exists(results_path):
        coco_gt_path = val_coco_path  # Path to your ground-truth validation JSON
        coco_gt = COCO(coco_gt_path)
        coco_dt = coco_gt.loadRes(results_path)

        # Run COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Collect summarized metrics
        metrics = {
            "mAP@[.5:.95]": coco_eval.stats[0],
            "mAP@0.5": coco_eval.stats[1],
            "mAP@0.75": coco_eval.stats[2],
            "Recall": coco_eval.stats[8],
            "Precision": coco_eval.stats[9],
            "val_loss": val_loss / len(val_loader)
        }
        print("\nValidation Metrics (COCO):")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        keys = ["mAP@[.5:.95]", "mAP@0.5", "mAP@0.75", "Recall", "Precision"]
        values = [metrics[k] for k in keys]

        plt.figure(figsize=(8, 5))
        plt.bar(keys, values, color="cornflowerblue")
        plt.title("COCO Validation Metrics")
        plt.ylim(0, 1)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
        plt.show()
            
    #Now Testing
    model.eval()
    with torch.no_grad():
        class_names = [
            "headlamp",
            "rear_bumper",
            "door",
            "hood",
            "front_bumper"
        ]
        y_true, y_pred = [], []
        for imgs, names in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs)

            #now decoding and printing detections
            detections = decode_preds(preds[0], conf_thresh=0.5)

            valid_detections = []
            for det in detections:
                cls_id = int(det[5])
                conf = float(det[4])

                if not (0 <= cls_id < len(class_names)):
                    print(f"[SKIP] Invalid class ID {cls_id}")
                    continue
                det[5] = cls_id
                valid_detections.append(det)
            
            #print("Filtered class IDs:", [int(d[5]) for d in valid_detections])
            img_drawm = draw_preds(imgs, valid_detections, class_names)

            base_name = os.path.splitext(names[0])[0]
            save_preds(img_drawm, valid_detections, class_names, base_name)

    #------------------------------------------------------------This is the END OF FILE----------------------------------------------------------------------------------------#