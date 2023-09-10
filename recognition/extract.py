import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils import crop_minAreaRect


def extract(files, out_dir):
    model = YOLO('runs/segment/augmented_train/weights/best.pt')
    results = model.predict(files, save=False)
    i = 0
    for result in results:
        image = cv2.imread(result.path)
        masks = result.masks.xy
        for mask in masks:
            cv2.imwrite(os.path.join(out_dir, f'{i}.jpg'), crop_minAreaRect(image, cv2.minAreaRect(mask)))
            i += 1


if __name__ == "__main__":
    extract(["fives.jpg"], os.path.join(os.getcwd(), 'data/images/classification2'))
