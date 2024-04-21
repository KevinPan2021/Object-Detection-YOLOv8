import torch
import pandas as pd
import random
import cv2
import os
import sys
import numpy as np
from PIL import Image 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import DataLoader, random_split

from model_YOLOv3 import convert_cells_to_bboxes, nms, iou, YOLOv3, YOLOLoss
from visualization import plot_image
from training import model_training, feedforward

# supports MacOS mps and CUDA
def GPU_Device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    
    
# Class labels 
class_labels = [ 
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# Image size 
image_size = 416
    
# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
] 

# Grid cell sizes 
grid_cell_sizes = [image_size // 32, image_size // 16, image_size // 8] 

# Defining the grid size and the scaled anchors 
scaled_anchors = (torch.tensor(ANCHORS) * 
    torch.tensor(grid_cell_sizes).unsqueeze(1).unsqueeze(1).repeat(1,3,2)).to(GPU_Device()) 


# augmumentation transform
aug_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        # Pad remaining areas with zeros 
        A.PadIfNeeded( 
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
        ), 
        A.ColorJitter(p=0.4),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)


# Transform for test and inference 
resize_transform = A.Compose( 
    [ 
        # Rescale an image so that maximum side is equal to image_size 
        A.LongestMaxSize(max_size=image_size), 
        # Pad remaining areas with zeros 
        A.PadIfNeeded( 
            min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT 
        ), 
        # Normalize the image 
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255), 
        # Convert the image to PyTorch tensor 
        ToTensorV2() 
    ],  
    # Augmentation for bounding boxes 
    bbox_params=A.BboxParams( format="yolo", min_visibility=0.4, label_fields=[] ) 
) 



# model inference
def inference(img, model):
    model.eval()
    with torch.no_grad():
        # unsqueeze the batch dim
        img = img.unsqueeze(0)
        
        # Getting the model predictions 
        output = model(img) 

        # Getting the boxes coordinates from the labels 
        # and converting them into bounding boxes without scaling 
        boxes = [] 
        for j in range(output[0].shape[1]): # 3 scales
            boxes += convert_cells_to_bboxes(output[j], scaled_anchors[j])[0]
        
        # Applying non-maximum suppression 
        boxes = nms(boxes, iou_threshold=0.45, confidence_threshold=0.6) 
          
    return boxes



# Create a dataset class to load the images and labels from the folder 
class Dataset(torch.utils.data.Dataset): 
    def __init__( 
        self, csv_file, image_dir, label_dir, anchors, image_size=416, 
        grid_sizes=[13, 26, 52], num_classes=20, transform=None ): 
        # Read the csv file with image names and labels 
        self.label_list = pd.read_csv(csv_file) 
        # Image and label directories 
        self.image_dir = image_dir 
        self.label_dir = label_dir 
        # Image size 
        self.image_size = image_size 
        # Transformations 
        self.transform = transform 
        # Grid sizes for each scale 
        self.grid_sizes = grid_sizes 
        # Anchor boxes 
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) 
        # Number of anchor boxes  
        self.num_anchors = self.anchors.shape[0] 
        # Number of anchor boxes per scale 
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes 
        self.num_classes = num_classes 
        # Ignore IoU threshold 
        self.ignore_iou_thresh = 0.5
  
    def __len__(self): 
        return len(self.label_list) 
    
    
    def __getitem__(self, idx):
        # Getting the label path 
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1]) 
        # We are applying roll to move class label to the last column 
        # 5 columns: x, y, width, height, class_label 
        bboxes = np.roll(np.loadtxt(fname=label_path, 
                         delimiter=" ", ndmin=2), 4, axis=1).tolist() 
          
        # Getting the image path 
        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0]) 
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Albumentations augmentations 
        if self.transform: 
            augs = self.transform(image=image, bboxes=bboxes) 
            image = augs["image"] 
            bboxes = augs["bboxes"] 
        
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
        # target : [probabilities, x, y, width, height, class_label] 
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) 
                   for s in self.grid_sizes] 
          
        # Identify anchor box and cell for each bounding box 
        for box in bboxes: 
            # Calculate iou of bounding box with anchor boxes 
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False) 
            # Selecting the best anchor box 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
            x, y, width, height, class_label = box 
  
            # At each scale, assigning the bounding box to the  
            # best matching anchor box 
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices: 
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale 
                  
                # Identifying the grid size for the scale 
                s = self.grid_sizes[scale_idx] 
                  
                # Identifying the cell to which the bounding box belongs 
                i, j = int(s * y), int(s * x) 
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] 
                  
                # Check if the anchor box is already assigned 
                if not anchor_taken and not has_anchor[scale_idx]: 
  
                    # Set the probability to 1 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
  
                    # Calculating the center of the bounding box relative to the cell 
                    x_cell, y_cell = s * x - j, s * y - i  
  
                    # Calculating the width and height of the bounding box  
                    # relative to the cell 
                    width_cell, height_cell = (width * s, height * s) 
  
                    # Idnetify the box coordinates 
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell] ) 
  
                    # Assigning the box coordinates to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 
  
                    # Assigning the class label to the target 
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 
  
                    # Set the anchor box as assigned for the scale 
                    has_anchor[scale_idx] = True
  
                # If the anchor box is already assigned, check if the  
                # IoU is greater than the threshold 
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: 
                    # Set the probability to -1 to ignore the anchor box 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        
        # Return the image and the target 
        return image, tuple(targets)
    




def main():    

    data_path = '../Datasets/PASCAL_VOC/'
    
    

    # Creating train dataset object 
    dataset = Dataset( 
        csv_file = data_path + "train.csv", 
        image_dir = data_path + "images/", 
        label_dir = data_path + "labels/", 
        grid_sizes=grid_cell_sizes, 
        anchors=ANCHORS, 
        transform=aug_transforms 
    )

    
    # visualize examples
    for i in range(5):
        # extract x, y
        x, y = dataset[i]
        
        # unsqueeze the batch dim
        x = x.unsqueeze(0)
        y = [item.unsqueeze(0) for item in y]

        # Getting the boxes coordinates from the labels 
        # and converting them into bounding boxes without scaling 
        boxes = [] 
        for j in range(y[0].shape[1]): 
            boxes += convert_cells_to_bboxes(y[j], scaled_anchors[j], is_predictions=False)[0] 
        
        # Applying non-maximum suppression 
        boxes = nms(boxes, iou_threshold=1, confidence_threshold=0.7) 
        
        # Plotting the image with the bounding boxes 
        plot_image(x[0].permute(1,2,0).to("cpu"), boxes, class_labels)


    # train/valid split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    
    # Split the dataset into training and validation sets
    random_seed = 42 # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=40, num_workers=5, pin_memory=True)  
    
    # create model 
    model = YOLOv3(num_classes=len(class_labels)).to(GPU_Device()) 
    if f'{type(model).__name__}.pth' in os.listdir():
        model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
        
    # model training
    model_training(train_loader, val_loader, model, YOLOLoss(), scaled_anchors.to(GPU_Device()), len(class_labels))
    
    # load the best model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    
    # test dataset
    test_dataset = Dataset( 
        csv_file = data_path+"test.csv", 
        image_dir = data_path+"images/", 
        label_dir = data_path+"labels/", 
        anchors = ANCHORS, 
        transform = resize_transform 
    ) 
    print('length dataset', len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=40, num_workers=5, pin_memory=True)
    loss, mAP = feedforward(test_loader, model, YOLOLoss(), scaled_anchors.to(GPU_Device()), len(class_labels))
    print(f"Test mAP: {mAP:.3f} | Test Loss: {loss:.3f}")
    
    # visualize test examples
    for i in range(6):
        # extract x
        x, _ = test_dataset[i]
        
        # move x to device
        x = x.to(GPU_Device())
        
        # inference
        boxes = inference(x, model)
        
        # Plotting the image with the bounding boxes 
        plot_image(x.permute(1,2,0).detach().cpu(), boxes, class_labels)
    
    
if __name__ == "__main__": 
    main()
