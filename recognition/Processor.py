from ultralytics import YOLO
from recognition.utils import crop_minAreaRect, get_class
import os
import cv2


class Processor:
    def __init__(self, detector_path='weights/detect.py', classify_path='weights/classify.py'):
        self.detection_model = YOLO(detector_path)
        self.recognition_model = YOLO(classify_path)

    def find_tiles(self, image):
        detected = self.detection_model([image], conf=0.6)[0]

        masks = detected.masks.xy
        cropped_tiles = []
        rects = []
        for mask in masks:
            rect = cv2.minAreaRect(mask)
            cropped_tiles.append(crop_minAreaRect(image, rect))
            rects.append(rect)

        recognized = self.recognition_model(cropped_tiles)
        results = []
        for rect, label in list(zip(rects, recognized)):
            results.append(([int(x) for x in rect[0]], get_class(label, self.recognition_model.names)))

        return results
