import cv2
import os
import numpy as np


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist")

    image = cv2.imread(image_path)
    return image


def resize_image(image, width=600):
    h, w = image.shape[:2]
    scale = width / w
    new_height = int(h * scale)

    resized = cv2.resize(image, (width, new_height))
    return resized


def to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced = clahe.apply(image)
    return enhanced


def apply_frequency_highpass(image, radius=30):
    dft = np.fft.fft2(image.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)
    y_grid, x_grid = np.ogrid[:rows, :cols]
    dist = np.sqrt((y_grid - crow) ** 2 + (x_grid - ccol) ** 2)
    mask[dist <= radius] = 0

    filtered_shift = dft_shift * mask
    idft_shift = np.fft.ifftshift(filtered_shift)
    result = np.fft.ifft2(idft_shift)
    result = np.abs(result)

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def preprocess_image(
    image_path=None,
    external_image=None,
    width=600,
    use_clahe=False,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    blur_kernel_size=(5, 5),
    blur_sigma=0
):
    if external_image is not None:
        image = external_image
    else:
        image = load_image(image_path)

    resized = resize_image(image, width)
    gray = to_gray(resized)

    if use_clahe:
        processed_gray = apply_clahe(
            gray,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size
        )
    else:
        processed_gray = gray

    blur = apply_gaussian_blur(
        processed_gray,
        kernel_size=blur_kernel_size,
        sigma=blur_sigma
    )

    return resized, processed_gray, blur
