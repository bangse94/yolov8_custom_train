from ultralytics import YOLO
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# Load a model
model = YOLO('/workspace/yolov8/ultralytics/models/v8/yolov8_custom.yaml')  # build a new model from scratch
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
# Use the model
model.train(data='/workspace/yolov8/ultralytics/datasets/engine_dataset.yaml', epochs=100, device=[0,1,2,3], batch=64)  # train the model