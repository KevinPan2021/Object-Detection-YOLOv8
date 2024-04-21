import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches 



# Function to plot images with bounding boxes and class labels 
def plot_image(image, boxes, class_labels): 
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    # Getting 20 different colors from the color map for 20 different classes 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 
  
    # Reading the image with OpenCV 
    img = np.array(image) 
    # Getting the height and width of the image 
    h, w, _ = img.shape 
  
    # Create figure and axes 
    fig, ax = plt.subplots(1) 
  
    # Add image to plot 
    ax.imshow(img) 
  
    # Plotting the bounding boxes and labels over the image 
    for box in boxes: 
        # Get the class from the box 
        class_pred = box[0] 
        # Get the center x and y coordinates 
        box = box[2:] 
        # Get the upper left corner coordinates 
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
    
        # Create a Rectangle patch with the bounding box 
        rect = patches.Rectangle( 
            (upper_left_x * w, upper_left_y * h), 
            box[2] * w, 
            box[3] * h, 
            linewidth=2, 
            edgecolor=colors[int(class_pred)], 
            facecolor="none", 
        ) 
          
        # Add the patch to the Axes 
        ax.add_patch(rect) 
          
        # Add class name to the patch 
        plt.text( 
            upper_left_x * w, 
            upper_left_y * h, 
            s=class_labels[int(class_pred)], 
            color="white", 
            verticalalignment="top", 
            bbox={"color": colors[int(class_pred)], "pad": 0}, 
        ) 
    
    plt.axis('off')
    
    # Display the plot 
    plt.show()
    


# plot the loss and acc curves
def plot_training_curves(train_mAP, train_loss, valid_mAP, valid_loss):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy loss')
    plt.legend()
    
    plt.figure()
    plt.plot(train_mAP, label='train')
    plt.plot(valid_mAP, label='valid')
    plt.title('mAP curves')
    plt.xlabel('epochs')
    plt.ylabel('mAP score')
    plt.legend()