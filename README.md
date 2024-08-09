# Object Detection with YOLOv8 and PyTorch

This project aims to perform Object Detection using YOLOv8 with the PyTorch MS-COCO dataset (GUI included).

### Dataset: 
[MS-COCO](http://images.cocodataset.org)

### Build: 

- **CPU:** Intel i9-13900H (14 cores)
- **GPU:** NVIDIA RTX 4060 (VRAM 8 GB)
- **RAM:** 32 GB

### Python Packages:

	* pytorch = 2.1.2
	* numpy = 1.23.5
	* OpenCV = 4.9.0.80
	* albumentations = 1.4.6
	* matplotlib = 3.7.0
	* pandas = 1.5.3
	* tqdm = 4.64.1

### Code Structure:

- `GUI.py` (Run to generate GUI)
- `main.py` (Run to train model)
- `dataset.py`
- `yolo_v8.py`
- `qt_main.ui`
- `training.py`
- `visualization.py`
- `summary.py`
- `class_ind_pair.pkl`

### Dataset Structure:
```bash
├── MS-COCO
    ├── annotations_test2017
        ├── image_info_test2017.json
        ├── image_info_test-dev2017.json
    ├── annotations_trainval2017
        ├── captions_train2017.json
        ├── captions_val2017.json
        ├── instances_train2017.json
        ├── instances_val2017.json
        ├── person_keypoints_train2017.json
        ├── person_keypoints_val2017.json
    ├── test2017
        ├── 000000000001.jpg
        ├── 000000000002.jpg
    ├── train2017
        ├── 000000000001.jpg
        ├── 000000000002.jpg
    ├── val2017
        ├── 000000000001.jpg
        ├── 000000000002.jpg
```

### Credits:

	* https://github.com/jahongir7174/YOLOv8-pt
	* https://github.com/ultralytics/ultralytics


	