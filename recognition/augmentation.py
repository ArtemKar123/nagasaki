import os
import random
from typing import List
import cv2
import numpy as np
import albumentations as A
import shutil
from utils import get_images


def extract_background(images_in_dir, labels_in_dir, images_out_dir, labels_out_dir):
    lower_bgr = np.array([35, 53, 0])
    upper_bgr = np.array([255, 221, 63])

    for file in get_images(images_in_dir):
        name = file.split('/')[-1]
        name = name[:name.rfind('.')]
        image = cv2.imread(file)

        mask = cv2.bitwise_not(cv2.inRange(image, lower_bgr, upper_bgr))
        res = cv2.bitwise_and(image, image.copy(), mask=mask)

        cv2.imwrite(os.path.join(images_out_dir, 'no_background_' + file.split('/')[-1]), res)
        shutil.copy2(os.path.join(labels_in_dir, name + '.txt'),
                     os.path.join(labels_out_dir, 'no_background_' + name + '.txt'))


def random_augmentations(images_in_dir, labels_in_dir, images_out_dir, labels_out_dir, per_image_range=(3, 5)):
    transform = A.Compose([
        A.Flip(p=0.2),
        A.ShiftScaleRotate(p=1, border_mode=1, rotate_limit=90),
        A.ISONoise(intensity=(0.1, 0.3)),
        A.RandomBrightnessContrast(p=0.2),
    ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels', 'order']))

    for file in get_images(images_in_dir):
        name = file.split('/')[-1]
        name = name[:name.rfind('.')]

        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_RGB2BGR)

        with open(os.path.join(labels_in_dir, f'{name}.txt'), 'r') as f:
            labels = [[float(x) for x in line.split(" ")[1:]] for line in f.read().split("\n")]

        keypoints = []
        class_labels = []
        orders = []
        h, w, _ = image.shape
        for label_index, label in enumerate(labels):
            for i in range(int(len(label) / 2)):
                keypoints.append((int(label[i * 2] * w), int(label[i * 2 + 1] * h)))
                class_labels.append(label_index)
                orders.append(i)

        for augmentation_index in range(random.randint(*per_image_range)):
            print(f'augmented_{augmentation_index}_{file.split("/")[-1]}')
            transformed = transform(image=image, keypoints=keypoints, class_labels=class_labels, order=orders)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']
            transformed_class_labels = transformed['class_labels']
            transformed_order = transformed['order']

            h, w, _ = transformed_image.shape
            class_counts = [0 for _ in range(len(labels))]
            for l in transformed_class_labels:
                class_counts[l] += 1

            valid_classes = set(filter(lambda x: class_counts[x] == 4, [i for i in range(len(labels))]))

            result_points = {}
            for i, point in enumerate(transformed_keypoints):
                if transformed_class_labels[i] in valid_classes:
                    if transformed_class_labels[i] not in result_points:
                        result_points[transformed_class_labels[i]] = [() for _ in range(4)]

                    result_points[transformed_class_labels[i]][transformed_order[i]] = (
                        point[0] / w, point[1] / h)

            cv2.imwrite(os.path.join(images_out_dir, f'augmented_{augmentation_index}_{file.split("/")[-1]}'),
                        cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

            with open(os.path.join(labels_out_dir, f'augmented_{augmentation_index}_{name}.txt'), 'w') as f:
                for points in result_points.values():
                    f.write('0')
                    for point in points:
                        f.write(f' {point[0]} {point[1]}')
                    f.write('\n')
        # break


if __name__ == '__main__':
    # extract_background(os.path.join(os.getcwd(), 'tmp/images/renamed'),
    #                    os.path.join(os.getcwd(), 'tmp/labels/renamed'),
    #                    os.path.join(os.getcwd(), 'tmp/images/augmented/background'),
    #                    os.path.join(os.getcwd(), 'tmp/labels/augmented/background')
    #                    )
    random_augmentations(os.path.join(os.getcwd(), 'tmp/images/augmented/in'),
                         os.path.join(os.getcwd(), 'tmp/labels/augmented/in'),
                         os.path.join(os.getcwd(), 'tmp/images/augmented/transforms'),
                         os.path.join(os.getcwd(), 'tmp/labels/augmented/transforms'))
