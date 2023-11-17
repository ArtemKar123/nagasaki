import copy

from ultralytics import YOLO
from recognition.utils import crop_minAreaRect, get_class, remove_background, masks2clusters
import os
import cv2
from scipy import stats


class Processor:
    def __init__(self, detector_path='weights/detect.py', classify_path='weights/classify.py', save=False):
        self.detection_model = YOLO(detector_path)
        self.recognition_model = YOLO(classify_path)
        self.i = 0
        self.save = save

    def detect_masks(self, image):
        image = remove_background(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detected = self.detection_model([image], conf=0.8)[0]

        masks = detected.masks.xy
        return masks

    def detect_tiles(self, image):
        """
            image: BGR image
        """

        image = remove_background(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detected = self.detection_model([image], conf=0.6)[0]

        masks = detected.masks.xy
        cropped_tiles = []
        rects = []
        clusters = masks2clusters(masks)
        rotations = []

        for mask in masks:
            rect = cv2.minAreaRect(mask)
            cropped, angle = crop_minAreaRect(image, rect)
            cropped_tiles.append(cropped)
            rotations.append(angle)
            rects.append(rect)

        # mode_angle = int(stats.mode(rotations).mode[0])
        # print("MODE", mode_angle)
        # if (mode_angle != 0):
        #     if mode_angle == 90:
        #         return self.detect_tiles(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
        #     elif mode_angle == -90:
        #         return self.detect_tiles(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

        return rects, clusters, rotations, cropped_tiles

    def get_tiles(self, image):

        rects, clusters, rotations, cropped_tiles = self.detect_tiles(image)

        if self.save:
            for tile in cropped_tiles:
                cv2.imwrite(f"/Users/artemkaramysev/Desktop/projects/nagasaki/recognition/tmp/tiles/{self.i}.png",
                            tile)
                self.i += 1

        recognized = self.recognition_model(cropped_tiles)
        results = []
        for rect, label, cluster, rotation in list(zip(rects, recognized, clusters, rotations)):
            results.append(
                {'center': [int(x) for x in rect[0]],
                 'tile': get_class(label, self.recognition_model.names),
                 'cluster': int(cluster),
                 'rotation': int(rotation),
                 'size': rect[1]
                 })

        return results
