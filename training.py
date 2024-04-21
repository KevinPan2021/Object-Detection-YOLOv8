import torch
from tqdm import tqdm
import torch.optim as optim 
import os
import sys
from collections import Counter
from torchmetrics.detection import MeanAveragePrecision

from model_YOLOv3 import convert_cells_to_bboxes, nms, iou
from visualization import plot_training_curves



# compute the mean average precision score
def mAP(pred, target, scaled_anchors, num_classes):
    batch_size = pred[0].shape[0]
    batch_mAP = 0.0
    
    # iterate through batch_size
    for i in range(batch_size):
        # convert pred and target to bounding boxes
        target_bbox = [] 
        for j in range(len(target)): 
            target_bbox += convert_cells_to_bboxes(target[j][[i],...], scaled_anchors[j], is_predictions=False)[0] 
        
        pred_bbox = [] 
        for j in range(len(pred)): # 3 scales
            pred_bbox += convert_cells_to_bboxes(pred[j][[i],...], scaled_anchors[j])[0]
        
        # apply non-maximum suppresion
        target_bbox = nms(target_bbox, iou_threshold=1, confidence_threshold=0.6) 
        pred_bbox = nms(pred_bbox, iou_threshold=0.45, confidence_threshold=0.6)
        
        # construct the targets and preds for MeanAveragePrecision
        target_boxes, target_labels = [], []
        pred_boxes, pred_scores, pred_labels = [], [], []
        
        for bbox in target_bbox:
            class_ind, confidence, a, b, c, d = bbox
            target_boxes.append([a-c/2, b-d/2, a+c/2, b+d/2])
            target_labels.append(class_ind)
        
        for bbox in pred_bbox:
            class_ind, confidence, a, b, c, d = bbox
            pred_boxes.append([a-c/2, b-d/2, a+c/2, b+d/2])
            pred_scores.append(confidence) # has no effect
            pred_labels.append(class_ind)
        
        targets = [{'boxes':torch.tensor(target_boxes), 'labels':torch.tensor(target_labels, dtype=torch.long)}]
        preds = [{'boxes':torch.tensor(pred_boxes), 'scores':torch.tensor(pred_scores), 'labels':torch.tensor(pred_labels, dtype=torch.long)}]
        
        metric = MeanAveragePrecision()
        metric.update(preds, targets)

        # append to total
        batch_mAP += metric.compute()['map'].numpy()
    
    # averaging over all batch_size
    batch_mAP /= batch_size
    
    return batch_mAP



# calculate the loss and mAP
def feedforward(data_loader, model, loss_fn, scaled_anchors, num_classes):
    model.eval()
    device = scaled_anchors.device
    epoch_loss = 0.0
    epoch_mAP = 0.0
    
    #with torch.autocast(device_type='mps', dtype=torch.bfloat16):
    with torch.no_grad():
        # Iterating over the training data
        for _, (x, y) in enumerate(tqdm(data_loader)):
            # move data to device
            x = x.to(device)
            y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
    
            # Getting the model predictions
            outputs = model(x)
            # Calculating the loss at each scale
            loss = (loss_fn(outputs[0], y0, scaled_anchors[0])
                    + loss_fn(outputs[1], y1, scaled_anchors[1])
                    + loss_fn(outputs[2], y2, scaled_anchors[2]))
    
            # Add the loss to the list
            epoch_loss += loss.item()
            
            # compute mAP
            epoch_mAP += mAP(outputs, y, scaled_anchors, num_classes)
    
    # averaging over all batches
    epoch_loss /= len(data_loader)
    epoch_mAP /= len(data_loader)
    
    return epoch_loss, epoch_mAP
        
            
        
# calculate the loss and mAP, gradient update
def backpropagation(data_loader, model, loss_fn, scaled_anchors, optimizer, num_classes):
    model.train()
    device = scaled_anchors.device
    epoch_loss = 0.0
    epoch_mAP = 0.0

    # Iterating over the training data
    for _, (x, y) in enumerate(tqdm(data_loader)):
        # move data to device
        x = x.to(device)
        y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)

        # Getting the model predictions
        outputs = model(x)
        # Calculating the loss at each scale
        loss = (loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2]))

        # Add the loss to the list
        epoch_loss += loss.item()
        
        # compute mAP
        epoch_mAP += mAP(outputs, y, scaled_anchors, num_classes)

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Optimization step
        optimizer.step()
        
    epoch_loss /= len(data_loader)
    epoch_mAP /= len(data_loader)
        
    return epoch_loss, epoch_mAP




# Define the train function to train the model
def model_training(train_loader, valid_loader, model, loss_fn, scaled_anchors, num_classes):
    n_epochs = 100
    # Learning rate for training
    learning_rate = 1e-4
    # l2 regularization
    weight_decay = 5e-4
    # Defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if f'{type(model).__name__.lower()}_optimizer.pth' in os.listdir():
        optimizer.load_state_dict(torch.load(f'{type(model).__name__}_optimizer.pth'))
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    # append the results of model with no training
    train_loss, train_mAP = feedforward(train_loader, model, loss_fn, scaled_anchors, num_classes)
    valid_loss, valid_mAP = feedforward(valid_loader, model, loss_fn, scaled_anchors, num_classes)
    print(f"Epoch 0/{n_epochs} | Train mAP: {train_mAP:.3f} | Train Loss: {train_loss:.3f} | Valid mAP: {valid_mAP:.3f} | Valid Loss: {valid_loss:.3f}")
    
    # create lists to keep track
    train_losses = [train_loss]
    valid_losses = [valid_loss]
    train_mAPs = [train_mAP]
    valid_mAPs = [valid_mAP]
    
    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01

    for epoch in range(n_epochs):
        train_loss, train_mAP = backpropagation(train_loader, model, loss_fn, scaled_anchors, optimizer, num_classes)
        valid_loss, valid_mAP = feedforward(valid_loader, model, loss_fn, scaled_anchors, num_classes)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_mAPs.append(train_mAP)
        valid_mAPs.append(valid_mAP)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train mAP: {train_mAP:.3f} | Train Loss: {train_loss:.3f} | Valid mAP: {valid_mAP:.3f} | Valid Loss: {valid_loss:.3f}")
        
        # evaluate the current performance
        # strictly better
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            not_improved = 0
            # save the best model based on validation loss
            torch.save(model.state_dict(), f'{type(model).__name__}.pth')
            # also save the optimizer state for future training
            torch.save(optimizer.state_dict(), f'{type(model).__name__}_optimizer.pth')

        # becomes worse
        elif valid_loss > best_valid_loss + threshold:
            not_improved += 1
            if not_improved >= patience:
                print('Early Stopping Activated')
                break
            
    plot_training_curves(train_mAPs, train_losses, valid_mAPs, valid_losses)
            
            