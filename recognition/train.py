import torch
from ultralytics import YOLO

if __name__ == '__main__':
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

    model = YOLO('yolov8n-cls.pt')

    model.train(data='datasets/classification', epochs=300, imgsz=640, batch=32,)
