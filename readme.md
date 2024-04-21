Introduction:
	This project aims to preform Object Detection using YOLOv3 from Scratch.



Dataset: 
	https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video/data


Build: 
	M1 Macbook Pro
	Miniforge 3 (Python 3.9)
	PyTorch version: 2.2.1

* Alternative Build:
	Windows (NIVIDA GPU)
	Anaconda 3
	PyTorch



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	model_transformer.py
	model_attention.py
	qt_main.py
	training.py
	visualization.py


Credits:
	YOLO-V3 model is referenced from "https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/"
	