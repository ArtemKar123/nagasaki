import copy
import os
import random
import math
import cv2
import numpy as np
import albumentations as A
import shutil
from utils import get_images
import scipy


def extract_background_with_labels(images_in_dir: str, labels_in_dir: str, images_out_dir: str, labels_out_dir: str):
    for file in get_images(images_in_dir):
        name = file.split('/')[-1]
        name = name[:name.rfind('.')]
        image = extract_background_with_labels(cv2.imread(file))

        cv2.imwrite(os.path.join(images_out_dir, 'no_background_' + file.split('/')[-1]), image)
        shutil.copy2(os.path.join(labels_in_dir, name + '.txt'),
                     os.path.join(labels_out_dir, 'no_background_' + name + '.txt'))


def draw_random_rectangles(image: np.ndarray):
    """
    Function to draw random rectangles on a given image.
    :param image: OpenCV Image
    :return: modified Image with rectangles
    """

    # Get the image dimensions
    height, width, _ = image.shape

    for _ in range(random.randint(10, 15)):
        # Generate random points for the rectangle
        top_left = (random.randint(0, width - 1), random.randint(0, height - 1))
        bottom_right = (random.randint(top_left[0], width - 1), random.randint(top_left[1], height - 1))

        # Generate a random color (B, G, R)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) if random.random() < 0.5 \
            else (255, 255, 255)

        # Generate a random thickness
        thickness = random.randint(1, 6)

        # Draw the rectangle on the image
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

 
def draw_random_lines(image: np.ndarray):
    """
    Function to draw random lines on a given image
    :param image: OpenCV Image
    :return: Image with lines
    """
    # Get the image dimensions
    height, width, _ = image.shape
    max_length = math.sqrt(width ** 2 + height ** 2) / 2
    for _ in range(random.randint(25, 40)):
        start_point = (random.randint(0, width - 1), random.randint(0, height - 1))
        length = random.uniform(0, max_length)
        angle = random.uniform(0, 2 * math.pi)
        end_point_x = int(start_point[0] + length * math.cos(angle))
        end_point_y = int(start_point[1] + length * math.sin(angle))
        end_point = (max(0, min(width - 1, end_point_x)), max(0, min(height - 1, end_point_y)))

        # Generate a random color (B, G, R)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Generate a random thickness
        thickness = random.randint(1, 6)

        # Draw the line on the image
        cv2.line(image, start_point, end_point, color, thickness)
    return image


def background_augmentation(image, keypoints, change_background=False):
    points = copy.deepcopy(keypoints)
    h, w, _ = image.shape

    # keypoints are expected to be in  [0, 1]
    for i in range(len(points)):
        for j in range(len(points[i])):
            points[i][j] = [int(points[i][j][0] * w), int(points[i][j][1] * h)]

    if change_background:
        background = generate_gradient(w, h)
    else:
        background = copy.deepcopy(image)
    background = draw_random_lines(draw_random_rectangles(background))
    pts = np.array(points, np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.fillPoly(mask, pts, 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    masked_image = cv2.bitwise_and(image, mask)
    mask_inv = cv2.bitwise_not(mask)
    masked_background = cv2.bitwise_and(background, mask_inv)
    combined = cv2.bitwise_or(masked_image, masked_background)
    return combined


def generate_gradient(width=1280, height=1280):
    transform = A.Compose([
        A.RandomGravel(p=1, number_of_patches=random.randint(30, 60), gravel_roi=(0.1, 0.1, 1, 1)),
    ])
    num_x = np.random.randint(2, 10)  # Random keypoints in the x-axis
    num_y = np.random.randint(2, 10)  # Random keypoints in the y-axis

    # Generate random keypoints and sort them
    x_points = np.sort(np.random.choice(width, num_x, replace=False))
    y_points = np.sort(np.random.choice(height, num_y, replace=False))

    # Generate Random colors corresponding to keypoints
    colors = np.random.randint(0, 256, (num_y, num_x, 3), dtype=np.uint8)

    # Make an empty image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Use 2D Interpolation for colors between keypoints
    for i in range(3):  # For each color separately
        interp = scipy.interpolate.interp2d(x_points, y_points, colors[:, :, i])
        image[:, :, i] = interp(np.arange(width), np.arange(height))

    # Convert the image from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    transformed = transform(image=image)
    transformed_image = transformed['image']

    return transformed_image


def random_augmentations(images_in_dir: str, labels_in_dir: str, images_out_dir: str, labels_out_dir: str,
                         per_image_range=(3, 5), change_background_p=0.5):
    transform = A.Compose([
        A.Flip(p=0.2),
        A.ShiftScaleRotate(p=1, border_mode=1, rotate_limit=90),
        A.ISONoise(intensity=(0.1, 0.3)),
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

            # if random.random() < change_background_p:
            transformed_image = background_augmentation(transformed_image, list(result_points.values()),
                                                        random.random() < change_background_p)

            cv2.imwrite(os.path.join(images_out_dir, f'augmented_{augmentation_index}_{file.split("/")[-1]}'),
                        cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

            with open(os.path.join(labels_out_dir, f'augmented_{augmentation_index}_{name}.txt'), 'w') as f:
                for points in result_points.values():
                    f.write('0')
                    for point in points:
                        f.write(f' {point[0]} {point[1]}')
                    f.write('\n')


def classification_augmentation(images_in_dir: str, images_out_dir: str, per_image_range=(5, 8), per_class_limit=30):
    transform = A.Compose([
        A.Rotate(limit=(180, 180)),
        A.Rotate(limit=10),
        A.RGBShift(),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2)),
        A.ISONoise(intensity=(0.1, 0.2)),
    ])

    for dir in os.listdir(images_in_dir):
        if not os.path.isdir(os.path.join(images_in_dir, dir)):
            continue

        target_directory = os.path.join(images_out_dir, dir)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        i = 0
        for image_path in get_images(os.path.join(images_in_dir, dir)):
            image = cv2.imread(os.path.join(images_in_dir, dir, image_path))
            cv2.imwrite(os.path.join(target_directory, f'{i}.png'), image)
            i += 1

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for _ in range(random.randint(*per_image_range)):
                transformed = transform(image=image)
                transformed_image = transformed['image']
                cv2.imwrite(os.path.join(target_directory, f'{i}.png'),
                            cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
                i += 1
                if i >= per_class_limit:
                    break
            if i >= per_class_limit:
                break


if __name__ == '__main__':
    random.seed(413)
    # classification_augmentation(
    #     os.path.join(os.getcwd(), 'data/images/classification-split2'),
    #     os.path.join(os.getcwd(), 'data/images/augmented-classification-split3'),
    #     per_image_range=(20, 25),
    #     per_class_limit=200
    # )

    # extract_background_with_labels(os.path.join(os.getcwd(), 'tmp/images/renamed'),
    #                    os.path.join(os.getcwd(), 'tmp/labels/renamed'),
    #                    os.path.join(os.getcwd(), 'tmp/images/augmented/background'),
    #                    os.path.join(os.getcwd(), 'tmp/labels/augmented/background')
    #                    )
    random_augmentations(os.path.join(os.getcwd(), 'data/images/renamed'),
                         os.path.join(os.getcwd(), 'data/labels/renamed'),
                         os.path.join(os.getcwd(), 'data/images/augmented3'),
                         os.path.join(os.getcwd(), 'data/labels/augmented3'), per_image_range=(25, 40))
    # cv2.imwrite('full2-nobg.png', extract_background(cv2.imread('full2.png')))
    # classification_augmentation(
    #     os.path.join(os.getcwd(), 'data/images/classification-split'),
    #     os.path.join(os.getcwd(), 'data/images/augmented-classification-split2'),
    #     per_image_range=(20, 25),
    #     per_class_limit=80
    # )
