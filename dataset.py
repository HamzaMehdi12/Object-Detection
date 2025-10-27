import os
import torch
import numpy as np
import json
import torchvision.transforms as T

from torch.utils.data import Dataset
from PIL import Image


class YoloMineDataset(Dataset):
    def __init__(self, root_dir, json_path = None, img_size = 640, transform = None, split = "train"):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.json_path = json_path
        self.split = split.lower()
        #Setting transformations
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

        self.split = split.lower()

        #opening annotations
        if json_path and os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                data = json.load(f)

            #creating mappings
            self.image_info = {img['id']: img for img in data['images']}
            
            #grouping annotations by image id
            self.image_annotations = {}
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.image_annotations:
                    self.image_annotations[img_id] = []
                self.image_annotations[img_id].append(ann)
            self.img_ids = list(self.image_info.keys())
        else:
            data = None
            self.image_annotations = None
            self.image_info = None
            self.img_ids = sorted([
                f for f in os.listdir(self.root_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        if self.split in ["train", "val"]:
            img_id = self.img_ids[index]
            img_info = self.image_info[img_id]
            img_path = os.path.join(self.root_dir, img_info['file_name'])

            #loading image
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            #Get bouding boxes
            anns = self.image_annotations.get(img_id, [])
            boxes, labels = [], []

            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

            #Convert to Tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            #Normalize (0-1 range)
            boxes[:, [0, 2]]  /= orig_w
            boxes[:, [1, 3]] /= orig_h

            image = self.transform(image)

            #Convert to Yolo format [class, x_center, y_center, width, height]
            yolo_target = []

            for box, label in zip(boxes, labels):
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                w = box[2] - box[0]
                h = box[3] - box[1]

                yolo_target.append([label, x_center, y_center, w, h])

            yolo_targets = torch.tensor(yolo_target, dtype=torch.float32)

            return image, yolo_targets
        
        elif self.split == "test":
            #inference mode: no labels
            img_info = self.img_ids[index]
            img_path = os.path.join(self.root_dir, img_info)
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            return image, img_info
        
