import cv2


def draw(im_path, label_path):
    image = cv2.imread(im_path)
    h, w, _ = image.shape
    print(w, h)
    with open(label_path, 'r') as f:
        labels = [[float(x) for x in line.split(" ")[1:]] for line in f.read().split("\n")]
    for label in labels:
        for i in range(int(len(label) / 2)):
            image = cv2.circle(image, (int(label[2 * i] * w), int(label[2 * i + 1] * h)), 2, (0, 0, 255), 3)

    cv2.imshow('a', image)
    cv2.waitKey()


if __name__ == '__main__':
    for i in range(48):
        try:
            draw(f'tmp/images/renamed/{i}.jpg', f'tmp/labels/renamed/{i}.txt')
        except Exception as e:
            pass
