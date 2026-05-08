import cv2
import numpy as np


def dilation(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def erosion(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def opening(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations
    )


def closing(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )


def apply_morphology(
    image,
    method="closing",
    kernel_size=(3, 3),
    iterations=1
):
    if method == "dilation":
        return dilation(image, kernel_size, iterations)

    elif method == "erosion":
        return erosion(image, kernel_size, iterations)

    elif method == "opening":
        return opening(image, kernel_size, iterations)

    elif method == "closing":
        return closing(image, kernel_size, iterations)

    else:
        raise ValueError(
            "Method must be dilation, erosion, opening, or closing"
        )
