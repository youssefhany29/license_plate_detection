import os
import cv2
import numpy as np

def load_image(path):
    return cv2.imread(path)

def normalize_image(img):
    return img / 255.0

def load_yolo_label(label_path, img_width, img_height):
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        return None

    # each line is one object: take the first plate annotation only
    data = lines[0].split()

    if len(data) < 5:
        return None

    cls, x_c, y_c, w, h = map(float, data[:5])  # [:5] safely ignores extra values

    x_c *= img_width
    y_c *= img_height
    w *= img_width
    h *= img_height

    x_min = int(x_c - w / 2)
    y_min = int(y_c - h / 2)
    x_max = int(x_c + w / 2)
    y_max = int(y_c + h / 2)

    return cls, x_min, y_min, x_max, y_max

def list_files(folder):
    return sorted(os.listdir(folder))

def join_path(folder, file):
    return os.path.join(folder, file)

def ensure_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(path, img):
    cv2.imwrite(path, img)
