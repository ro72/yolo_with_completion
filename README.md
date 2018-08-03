# YOLO v3 Object detector and Background Completion System

This project tries to connect the following two repos.

[YOLO v3 Object detector](https://github.com/ayooshkathuria/pytorch-yolo-v3)
and
[Globally and Locally Consistent Image Completion](https://github.com/satoshiiizuka/siggraph2017_inpainting)

## Requirements
1. Python 3.5
2. OpenCV
3. PyTorch 0.4

Using PyTorch 0.3 will break the detector.

## Download models
```
bash download_model.sh
```

## Running the code
```
python yolo_completion.py --images img2.jpg --det det
```
