import math
import random
import shutil

import numpy as np
from ultralytics.utils.ops import xywhr2xyxyxyxy
import pandas as pd
import os
import json
from typing import List
import cv2


def get_class(result, names):
    idx = result.probs.numpy().data.argmax()
    return names[int(idx)]


def filter_files(files: List[str]):
    formats = ['png', 'jpg', 'jpeg']
    return filter(lambda x: x.split('.')[-1] in formats, files)


def get_images(in_dir):
    return [os.path.join(in_dir, x) for x in filter_files(os.listdir(in_dir))]


def label_studio_classification_to_directories(json_path: str, images_dir: str, save_path: str):
    with open(json_path, 'r') as f:
        data = json.loads(f.read())

    counts = {}
    for entry in data:
        name = entry['image'].split('-')[-1]
        label = entry['choice']
        if label not in counts:
            counts[label] = 0
        else:
            counts[label] += 1

        path = os.path.join(save_path, label, f'{counts[label]}{name[name.rfind("."):]}')
        target_directory = os.path.dirname(path)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        shutil.copy2(os.path.join(images_dir, name), path)


def label_studio_csv2xyxyxyxy(json_path: str, save_path: str = None):
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

    with open(json_path, 'r') as f:
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
                skip = False
                for x in label:
                    if x < 0 or x > 1:
                        skip = True
                        break
                if skip:
                    continue

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


def train_val_test_split(images_in_dir, labels_in_dir, out_dir, train_p=0.6, val_p=0.2, test_p=0.2):
    images = get_images(images_in_dir)
    pairs = []
    for image in images:
        name = image.split('/')[-1]
        name = name[:name.rfind('.')]

        label = os.path.join(labels_in_dir, name + '.txt')
        pairs.append((image, label))

    random.shuffle(pairs)
    for i, (image, label) in enumerate(pairs):
        split_ratio = float(i) / len(pairs)

        if split_ratio < train_p:
            image_dst = os.path.join(out_dir, 'images', 'train', image.split('/')[-1])
            label_dst = os.path.join(out_dir, 'labels', 'train', label.split('/')[-1])
        elif split_ratio < train_p + val_p:
            image_dst = os.path.join(out_dir, 'images', 'val', image.split('/')[-1])
            label_dst = os.path.join(out_dir, 'labels', 'val', label.split('/')[-1])
        else:
            image_dst = os.path.join(out_dir, 'images', 'test', image.split('/')[-1])
            label_dst = os.path.join(out_dir, 'labels', 'test', label.split('/')[-1])

        shutil.copy(image, image_dst)
        shutil.copy(label, label_dst)


def classification_train_val_test_split(images_in_dir, out_dir, train_p=0.6, val_p=0.2, test_p=0.2):
    for dir in os.listdir(images_in_dir):
        if not os.path.isdir(os.path.join(images_in_dir, dir)):
            continue

        image_paths = get_images(os.path.join(images_in_dir, dir))
        random.shuffle(image_paths)

        for i, image_path in enumerate(image_paths):
            split_ratio = float(i) / len(image_paths)

            if split_ratio < train_p:
                image_dst = os.path.join(out_dir, 'train', dir, image_path.split('/')[-1])
            elif split_ratio < train_p + val_p:
                image_dst = os.path.join(out_dir, 'val', dir, image_path.split('/')[-1])
            else:
                image_dst = os.path.join(out_dir, 'test', dir, image_path.split('/')[-1])

            target_directory = os.path.dirname(image_dst)
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)

            shutil.copy(image_path, image_dst)


def crop_minAreaRect(img, rect):
    # Get parameters of the rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # Adjust angle and size if width > height
    if size[0] > size[1]:
        angle -= 90
        size = tuple(reversed(size))

    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # Perform rotation on src image
    img_rot = cv2.warpAffine(img, M, img.shape[1::-1])

    # Draw the rotated rectangle
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop


if __name__ == '__main__':
    random.seed(413)

    classification_train_val_test_split(
        os.path.join(os.getcwd(), 'data/images/augmented-classification-split'),
        os.path.join(os.getcwd(), 'datasets/classification'),
    )
