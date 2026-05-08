import cv2


def find_contours(image):
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def calculate_edge_density_score(processed_image, bbox, ideal_density=0.22):
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return 0.0

    roi = processed_image[y:y + h, x:x + w]

    if roi.size == 0:
        return 0.0

    white_pixels = cv2.countNonZero(roi)
    total_pixels = w * h

    edge_density = white_pixels / float(total_pixels)

    density_score = 1.0 - min(
        abs(edge_density - ideal_density) / ideal_density,
        1.0
    )

    return density_score


def calculate_plate_score(
    bbox,
    contour,
    image_shape,
    processed_image,
    ideal_density=0.22
):
    x, y, w, h = bbox
    img_h, img_w = image_shape[:2]

    if h == 0 or img_w == 0 or img_h == 0:
        return 0.0

    box_area = w * h
    image_area = img_w * img_h
    aspect_ratio = w / float(h)

    ideal_aspect_ratio = 4.0

    aspect_score = 1.0 - min(
        abs(aspect_ratio - ideal_aspect_ratio) / ideal_aspect_ratio,
        1.0
    )

    area_ratio = box_area / float(image_area)

    if area_ratio < 0.005:
        area_score = area_ratio / 0.005
    elif area_ratio > 0.25:
        area_score = max(
            0.0,
            1.0 - ((area_ratio - 0.25) / 0.25)
        )
    else:
        area_score = 1.0

    contour_area = cv2.contourArea(contour)
    extent_score = contour_area / float(box_area) if box_area > 0 else 0.0
    extent_score = min(extent_score, 1.0)

    center_y = y + h / 2
    vertical_position = center_y / float(img_h)

    if 0.30 <= vertical_position <= 0.90:
        position_score = 1.0
    else:
        position_score = 0.6

    margin_x = img_w * 0.02
    margin_y = img_h * 0.02

    touches_border = (
        x <= margin_x or
        y <= margin_y or
        x + w >= img_w - margin_x or
        y + h >= img_h - margin_y
    )

    border_penalty = 0.7 if touches_border else 1.0

    edge_density_score = calculate_edge_density_score(
        processed_image,
        bbox,
        ideal_density=ideal_density
    )

    score = (
        0.30 * aspect_score +
        0.20 * area_score +
        0.15 * extent_score +
        0.15 * position_score +
        0.20 * edge_density_score
    )

    score = score * border_penalty

    return score


def filter_plate_contours(
    contours,
    image_shape,
    processed_image,
    min_area_ratio=0.002,
    max_area_ratio=0.30,
    min_aspect_ratio=1.5,
    max_aspect_ratio=8.0,
    ideal_density=0.22
):
    valid_contours = []

    img_h, img_w = image_shape[:2]
    image_area = img_w * img_h

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h == 0:
            continue

        box_area = w * h
        area_ratio = box_area / float(image_area)
        aspect_ratio = w / float(h)

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        score = calculate_plate_score(
            bbox=(x, y, w, h),
            contour=cnt,
            image_shape=image_shape,
            processed_image=processed_image,
            ideal_density=ideal_density
        )

        valid_contours.append((cnt, (x, y, w, h), score))

    return valid_contours


def select_best_plate(contours):
    if len(contours) == 0:
        return None

    contours = sorted(
        contours,
        key=lambda c: c[2],
        reverse=True
    )

    best_contour, best_bbox, best_score = contours[0]

    return best_contour, best_bbox


def crop_plate(image, bbox):
    x, y, w, h = bbox
    plate = image[y:y + h, x:x + w]

    return plate


def localize_plate(original_image, processed_image, ideal_density=0.22):
    contours = find_contours(processed_image)

    candidates = filter_plate_contours(
        contours=contours,
        image_shape=original_image.shape,
        processed_image=processed_image,
        ideal_density=ideal_density
    )

    best = select_best_plate(candidates)

    if best is None:
        return None, None

    cnt, bbox = best
    plate_region = crop_plate(original_image, bbox)

    return plate_region, bbox
