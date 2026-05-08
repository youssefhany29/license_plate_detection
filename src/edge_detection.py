import cv2


# canny edge
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


# sobel edge
def sobel_edge_detection(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)

    edges = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    return edges


# pipeline edge
def detect_edges(
    image,
    method="canny",
    low_threshold=50,
    high_threshold=150
):
    if method == "canny":
        return canny_edge_detection(
            image,
            low_threshold,
            high_threshold
        )

    elif method == "sobel":
        return sobel_edge_detection(image)

    else:
        raise ValueError("method must be canny or sobel")
