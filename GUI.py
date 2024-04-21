application_name = 'Object Detection'
# pyqt packages
from PyQt5.QtGui import QPainter, QPixmap, QImage, QIcon, QColor, QFont
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog

import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
import torch
from PIL import Image
import albumentations as A 
from albumentations.pytorch import ToTensorV2 

from model_YOLOv3 import YOLOv3
from qt_main import Ui_Application
from main import class_labels, GPU_Device, inference, image_size



def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()
        
        
        
class QT_Action(Ui_Application, QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle(application_name) # set the title
        
        # runtime variable
        self.image_size = image_size
        self.class_labels = class_labels
        self.image = None
        self.model = None
        self.transform = A.Compose( 
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
        ) 
        
        
        # load the model
        self.load_model_action()
        
        
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.toolButton_import.clicked.connect(self.import_action)
        self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_process.clicked.connect(self.process_action)
        
    
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'YOLOv3':
            # load the model architechture
            self.model = YOLOv3(num_classes=len(self.class_labels))
            
            # loading the training model weights
            self.model.load_state_dict(torch.load(f'{self.model_name}.pth'))
            
        # move model to GPU
        self.model = self.model.to(GPU_Device())
        
        self.model.eval() # Set model to evaluation mode
        
        
        
    
    # clicking the import button action
    def import_action(self,):
        # show an "Open" dialog box and return the path to the selected file
        filename, _ = QFileDialog.getOpenFileName(None, "Select file", options=QFileDialog.Options())
        self.lineEdit_import.setText(filename)
        
        # didn't select any files
        if filename is None or filename == '': 
            return
    
        # selected .oct or .octa files
        if filename.endswith('.jpg'):
            self.image = Image.open(filename) 
            self.lineEdit_import.setText(filename)
            #X = [transform(img)]
            self.update_display()
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .jpg')
            self.import_action()
        
        
    def update_display(self):
        if not self.image is None:
            augs = self.transform(image=np.array(self.image))
            data = augs["image"] 
            data = data.permute(1,2,0).numpy()
            data = (data*255).astype(np.uint8)
            height, width, channels = data.shape
            q_image = QImage(data.tobytes(), width, height, width*channels, QImage.Format_RGB888)  # Create QImage
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_image.setPixmap(qpixmap)
            
            
    def process_action(self):
        if self.image is None:
            show_message(self, title='Process Error', message='Please load an image first')
            return
        
        augs = self.transform(image=np.array(self.image))
        data = augs["image"] 
        
        # move data to GPU
        data = data.to(GPU_Device())
        
        # model inference
        with torch.no_grad():  # Disable gradient calculation
            boxes = inference(data, self.model)
        

        # convert x to gray image
        img = data.permute(1,2,0).detach().cpu().numpy()
        gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) # convert to gray image
        gray_img = gray_img*255
        height, width = gray_img.shape
        q_image = QImage(gray_img.astype(np.uint8).tobytes(), width, height, width, QImage.Format_Grayscale8)  # Create QImage
        qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
        
        # draw bounding boxes
        qp = QPainter(qpixmap)
        
        # Getting the color map from matplotlib 
        colour_map = plt.get_cmap("tab20b") 
        # Getting 20 different colors from the color map for 20 different classes 
        colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 
        # Getting the height and width of the image 
        h, w, _ = img.shape 
        
        for box in boxes:
            # Get the class from the box 
            class_pred, confidence, a, b, c, d = box
            red, green, blue, _ = colors[int(class_pred)]
            qp.setPen(QColor.fromRgbF(red, green, blue))
            
            # Get the upper left corner coordinates 
            x0 = (a - c / 2)
            y0 = (b - d / 2)
            # draw bounding box
            qp.drawRect(int(x0*w), int(y0*h), int(c*w), int(d*h))
            
            # Write class label below the upper left corner of the bounding box
            class_label = self.class_labels[int(class_pred)]
            qp.setFont(QFont("Arial", 10))  # Set font size and type
            qp.drawText(int(x0*w)+5, int(y0*h)+10, class_label)
    
        qp.end()
        
        # Display the result in label_detection
        self.label_detection.setPixmap(qpixmap)
    
    
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()