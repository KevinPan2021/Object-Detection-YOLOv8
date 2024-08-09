import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from torch.optim.lr_scheduler import LambdaLR

from yolo_v8 import ComputeLoss, non_max_suppression, wh2xy
from visualization import plot_training_curves
from torch.cuda.amp import GradScaler, autocast



@torch.no_grad()
# compute the mean average precision score
# pred shape: torch.Size([batch_size, 84, 8400])
# target shape: torch.Size([num_bbox, 6])
def mAP(pred, target, scale):
    # apply nms
    # pred shape -> batch_size * torch.Size([(num_bbox, 6)])
    pred = non_max_suppression(pred, 0.1, 0.65, max_det=100, max_nms=2000)
    
    # rescale target from [0, 1] to [0, scale]
    target[:, 2:] *= scale

    # define torch lightening mAP metric
    metric = MeanAveragePrecision()
    metric.warn_on_many_detections = False
    
    # iterate through batch_size
    for i in range(len(pred)):
        target_data = target[target[:, 0] == i, 1:]
        pred_data = pred[i]
        
        # convert target_data from wh to xy
        target_data[:, 1:] = wh2xy(target_data[:, 1:])
        
        targets = [{
            'boxes':target_data[:, 1:], 
            'labels':target_data[:,0].long()
        }]
        preds = [{
            'boxes':pred_data[:, :4], 
            'scores':pred_data[:, 4], 
            'labels':pred_data[:, -1].long()
        }]
        
        metric.update(preds, targets)
        
    batch_mAP = metric.compute()['map']
    metric.reset()
    return batch_mAP


# Lambda function for learning rate warm-up
def lr_lambda(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps  # Gradual increase from 0 to 1
    return 1  # Continue with normal learning rate after warm-up


@torch.no_grad()
def feedforward(data_loader, model, mAP_skip=1):
    model.eval()
    
    criterion = ComputeLoss(model)
    device = next(model.parameters()).device
    epoch_loss = 0.0
    epoch_mAP = 0.0
    
    with tqdm(total=len(data_loader)) as pbar:
        # Iterate over the dataset
       for i, (X, Y) in enumerate(data_loader):
           # move to device
           X = X.to(device)
           Y = Y.to(device)
            
           # Forward
           with autocast(dtype=torch.float16):
               output, box = model(X)  # forward
               loss_score = criterion(output, Y)
               
               map_score = torch.tensor([0])
               if i % mAP_skip == 0: # only compute mAP when necessary
                   map_score = mAP(box, Y, X.shape[-1])
               
           # Add the loss to the list
           epoch_loss += loss_score.item()
           
           # compute mAP
           epoch_mAP += map_score.item()
           
           # Update tqdm description with loss and accuracy
           pbar.set_postfix({
                'Loss': f'{epoch_loss/(i+1):.3f}', 
                'mAP': f'{epoch_mAP/(i+1)*mAP_skip:.3f}'
           })
           pbar.update(1)
           
           torch.cuda.empty_cache()
           
    # averaging over all batches
    epoch_loss /= len(data_loader)
    epoch_mAP /= len(data_loader) / mAP_skip
    
    return epoch_mAP, epoch_loss



def backpropagation(data_loader, model, optimizer, scaler, scheduler, mAP_skip=1):
    model.train()
    
    criterion = ComputeLoss(model)
    device = next(model.parameters()).device
    epoch_loss = 0.0
    epoch_mAP = 0.0
    
    with tqdm(total=len(data_loader)) as pbar:
        # Iterate over the dataset
       for i, (X, Y) in enumerate(data_loader):
           # move to device
           X = X.to(device)
           Y = Y.to(device)

           # Forward
           with autocast(dtype=torch.float16):
               output, box = model(X)  # forward
               loss_score = criterion(output, Y)
               
               map_score = torch.tensor([0])
               if i % mAP_skip == 0: # only compute mAP when necessary
                   map_score = mAP(box, Y, X.shape[-1])
               
           # Add the loss to the list
           epoch_loss += loss_score.item()
           
           # compute mAP
           epoch_mAP += map_score.item()
           
           # Reset gradients
           optimizer.zero_grad()
           
           # Backpropagate the loss
           scaler.scale(loss_score).backward()
           
           # Optimization step
           scaler.step(optimizer)
           scheduler.step() 
           
           # Updates the scale for next iteration.
           scaler.update()
           
           # Update tqdm description with loss and accuracy
           pbar.set_postfix({
                'Loss': f'{epoch_loss/(i+1):.3f}', 
                'mAP': f'{epoch_mAP/(i+1)*mAP_skip:.3f}'
           })
           pbar.update(1)
           
           torch.cuda.empty_cache()
           
    # averaging over all batches
    epoch_loss /= len(data_loader)
    epoch_mAP /= len(data_loader) / mAP_skip
    
    return epoch_mAP, epoch_loss


# model training loop
def model_training(train_loader, valid_loader, model):
    # Define hyperparameters
    n_epochs = 100
    warmup_steps = 2
    
    # Learning rate for training
    learning_rate = 1e-4
    
    # l2 regularization
    weight_decay = 5e-4
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    
    # Create a learning scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps))

    # Creates a GradScaler
    scaler = GradScaler()

    # Early Stopping criteria
    patience = 3
    not_improved = 0
    threshold = 0.01
    best_valid_loss = float('inf')
    
    # Training loop
    train_loss_curve, train_mAP_curve = [], []
    valid_loss_curve, valid_mAP_curve = [], []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        train_mAP, train_loss = backpropagation(train_loader, model, optimizer, scaler, scheduler, mAP_skip=16)
        valid_mAP, valid_loss = feedforward(valid_loader, model, mAP_skip=2)

        train_loss_curve.append(train_loss)
        train_mAP_curve.append(train_mAP)
        valid_loss_curve.append(valid_loss)
        valid_mAP_curve.append(valid_mAP)

        # evaluate the current preformance
        if valid_loss - threshold < best_valid_loss:
            best_valid_loss = valid_loss
            not_improved = 0
            
            # save the best model in float16
            model.half()
            torch.save(model.state_dict(), model.name() + '.pth')
            model.float()
            
        else:
            not_improved += 1
            if not_improved >= patience:
                print('Early Stopping Activated')
                break
            
    plot_training_curves(train_mAP_curve, train_loss_curve, valid_mAP_curve, valid_loss_curve)
    