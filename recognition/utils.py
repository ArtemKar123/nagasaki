import math
import shutil

import numpy as np
from ultralytics.utils.ops import xywhr2xyxyxyxy
import pandas as pd
import os
import json


def label_studio_csv2xyxyxyxy(path: str, save_path: str = None):
    """
    Converts labels from label studio cvs format to xyxyxyxy
    """

    def fix_format(x, y, w, h, r):
        cos, sin = (np.cos, np.sin)
        x_w = w / 2
        y_h = h / 2

        angle = fix_rotation(r)
        rotation = angle * math.pi / 180.0

        cos_rot = cos(rotation)
        sin_rot = sin(rotation)

        dx = (x_w * cos_rot - y_h * sin_rot)
        dy = (x_w * sin_rot + y_h * cos_rot)

        return [x + dx, y + dy, w, h, -fix_rotation(r)]

    def fix_rotation(angle: float):
        if angle > 180:
            return angle - 360
        elif angle < -180:
            return angle + 360
        else:
            return angle

    with open(path, 'r') as f:
        data = json.loads(f.read())
    for entry in data:
        name = entry['image'].split('/')[-1]
        labels = [fix_format(label['original_width'] * label['x'] / 100,
                             label['original_height'] * label['y'] / 100,
                             label['original_width'] * label['width'] / 100,
                             label['original_height'] * label['height'] / 100,
                             label['rotation']
                             ) for label in entry['label']]

        labels = [xywhr2xyxyxyxy(np.array(l))[0] for l in labels]

        original_dimentions = [entry['label'][0]['original_width'], entry['label'][0]['original_height']]

        normalized = []
        for label in labels:
            normalized_label = []
            for i, x in enumerate(label):
                normalized_label.append(x / original_dimentions[i % 2])
            normalized.append(normalized_label)

        txt_name = name[:name.rfind('.')] + '.txt'
        with open(os.path.join(save_path, txt_name), 'w') as f:
            for label in normalized:
                f.write('0 ' + ' '.join(str(x) for x in label) + "\n")


def rename_image_labels(path: str):
    """
    Renames image and labels to something more readable
    """
    for index, file in enumerate(os.listdir(os.path.join(path, 'images', 'original'))):
        split = file.rfind('.')
        extension = file[split:].replace('jpeg', 'jpg')
        if extension not in ['.jpeg', '.jpg', '.png']:
            print(file)
            continue
        name = file[:split]
        txt_name = name + '.txt'
        # print(file)
        shutil.copy(os.path.join(path, 'images', 'original', file),
                    os.path.join(path, 'images', 'renamed', str(index) + extension))
        shutil.copy(os.path.join(path, 'labels', 'original', txt_name),
                    os.path.join(path, 'labels', 'renamed', str(index) + '.txt'))


if __name__ == '__main__':
    label_studio_csv2xyxyxyxy('labels.json', 'tmp/labels/original')
    rename_image_labels('tmp/')
