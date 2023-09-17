from ultralytics import YOLO
from recognition.utils import crop_minAreaRect, get_class, remove_background
import os
import cv2


class Processor:
    def __init__(self, detector_path='weights/detect.py', classify_path='weights/classify.py', save=False):
        self.detection_model = YOLO(detector_path)
        self.recognition_model = YOLO(classify_path)
        self.i = 0
        self.save = save

    def detect_tiles(self, image):
        """
            image: BGR image
        """

        image = remove_background(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detected = self.detection_model([image], conf=0.6)[0]

        masks = detected.masks.xy
        cropped_tiles = []
        rects = []
        for mask in masks:
            rect = cv2.minAreaRect(mask)
            cropped_tiles.append(crop_minAreaRect(image, rect))
            rects.append(rect)

        return rects, cropped_tiles

    def get_tiles(self, image):

        rects, cropped_tiles = self.detect_tiles(image)

        if self.save:
            for tile in cropped_tiles:
                cv2.imwrite(f"/Users/artemkaramysev/Desktop/projects/nagasaki/recognition/tmp/{self.i}.png",
                            cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                self.i += 1

        recognized = self.recognition_model(cropped_tiles)
        results = []
        for rect, label in list(zip(rects, recognized)):
            results.append(([int(x) for x in rect[0]], get_class(label, self.recognition_model.names)))

        return results
