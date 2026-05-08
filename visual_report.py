import os
import random
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

from evaluation.metrics import compute_iou
from src.edge_detection import detect_edges
from src.morphology import apply_morphology
from src.preprocessing import preprocess_image
from src.utils import list_files, join_path, load_image, load_yolo_label
from src.plate_localization import (
    find_contours,
    filter_plate_contours,
    select_best_plate,
    crop_plate
)


OUTPUT_DIR = os.path.join("Graphs", "presentation_visuals")

IMAGE_DIR = "data/test/images"
LABEL_DIR = "data/test/labels"

RESIZE_WIDTH = 600

BEST_CONFIG = {
    "name": "Canny 73/198 + closing 2x2 + blur 7x7 + density 0.219",
    "edge_method": "canny",
    "low_threshold": 73,
    "high_threshold": 198,
    "morph_method": "closing",
    "kernel_size": (2, 2),
    "iterations": 1,
    "use_clahe": False,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
    "blur_kernel_size": (7, 7),
    "blur_sigma": 0,
    "ideal_density": 0.219
}


def ensure_output_folder():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def read_image_files(image_dir):
    files = []

    for file_name in list_files(image_dir):
        lower_name = file_name.lower()

        if lower_name.endswith(".jpg") or lower_name.endswith(".jpeg") or lower_name.endswith(".png"):
            files.append(file_name)

    return files


def get_label_path(image_name):
    label_name = (
        image_name.replace(".jpg", ".txt")
                  .replace(".jpeg", ".txt")
                  .replace(".png", ".txt")
    )

    return join_path(LABEL_DIR, label_name)


def convert_yolo_box_to_resized_box(label, orig_w, orig_h):
    _, x1, y1, x2, y2 = label

    scale = RESIZE_WIDTH / orig_w

    return (
        int(x1 * scale),
        int(y1 * scale),
        int(x2 * scale),
        int(y2 * scale)
    )


def process_image(image):
    resized, gray, blur = preprocess_image(
        external_image=image,
        use_clahe=BEST_CONFIG["use_clahe"],
        clahe_clip_limit=BEST_CONFIG["clahe_clip_limit"],
        clahe_tile_grid_size=BEST_CONFIG["clahe_tile_grid_size"],
        blur_kernel_size=BEST_CONFIG["blur_kernel_size"],
        blur_sigma=BEST_CONFIG["blur_sigma"]
    )

    edges = detect_edges(
        blur,
        method=BEST_CONFIG["edge_method"],
        low_threshold=BEST_CONFIG["low_threshold"],
        high_threshold=BEST_CONFIG["high_threshold"]
    )

    morph = apply_morphology(
        edges,
        method=BEST_CONFIG["morph_method"],
        kernel_size=BEST_CONFIG["kernel_size"],
        iterations=BEST_CONFIG["iterations"]
    )

    contours = find_contours(morph)

    candidates = filter_plate_contours(
        contours=contours,
        image_shape=resized.shape,
        processed_image=morph,
        ideal_density=BEST_CONFIG["ideal_density"]
    )

    best = select_best_plate(candidates)

    if best is None:
        plate = None
        bbox = None
    else:
        _, bbox = best
        plate = crop_plate(resized, bbox)

    return resized, gray, blur, edges, morph, candidates, plate, bbox


def draw_box(image, box, color, thickness=2):
    output = image.copy()

    if box is None:
        return output

    x1, y1, x2, y2 = box
    cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

    return output


def draw_candidates(image, candidates, max_candidates=10):
    output = image.copy()

    sorted_candidates = sorted(
        candidates,
        key=lambda c: c[2],
        reverse=True
    )

    for index, candidate in enumerate(sorted_candidates[:max_candidates]):
        _, bbox, score = candidate
        x, y, w, h = bbox

        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 160, 0), 2)
        cv2.putText(
            output,
            f"{index + 1}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 160, 0),
            2
        )

    return output


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_pipeline_stages(image_name, image, gt_box):
    resized, gray, blur, edges, morph, candidates, plate, bbox = process_image(image)

    pred_box = None

    if bbox is not None:
        x, y, w, h = bbox
        pred_box = (x, y, x + w, y + h)

    prediction_overlay = resized.copy()

    if gt_box is not None:
        prediction_overlay = draw_box(prediction_overlay, gt_box, (0, 255, 0), 2)

    if pred_box is not None:
        prediction_overlay = draw_box(prediction_overlay, pred_box, (0, 0, 255), 2)

    candidates_overlay = draw_candidates(resized, candidates)

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(bgr_to_rgb(resized))
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("Grayscale Image")

    axes[0, 2].imshow(blur, cmap="gray")
    axes[0, 2].set_title("Gaussian Blur 7x7")

    axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title("Canny Edge Detection 73/198")

    axes[1, 1].imshow(morph, cmap="gray")
    axes[1, 1].set_title("Morphological Closing 2x2")

    axes[1, 2].imshow(bgr_to_rgb(prediction_overlay))
    axes[1, 2].set_title("Prediction vs Ground Truth")

    for ax in axes.flat:
        ax.axis("off")

    figure.suptitle(
        f"Pipeline Stages - {BEST_CONFIG['name']}",
        fontsize=16,
        fontweight="bold"
    )

    figure.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"pipeline_stages_{os.path.splitext(image_name)[0]}.png"
    )

    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    candidate_figure, candidate_axes = plt.subplots(1, 2, figsize=(14, 6))

    candidate_axes[0].imshow(bgr_to_rgb(candidates_overlay))
    candidate_axes[0].set_title("Top Candidate Boxes")
    candidate_axes[0].axis("off")

    candidate_axes[1].imshow(bgr_to_rgb(prediction_overlay))
    candidate_axes[1].set_title("Final Prediction: Red | Ground Truth: Green")
    candidate_axes[1].axis("off")

    candidate_figure.tight_layout()

    candidate_save_path = os.path.join(
        OUTPUT_DIR,
        f"candidate_selection_{os.path.splitext(image_name)[0]}.png"
    )

    candidate_figure.savefig(candidate_save_path, dpi=300, bbox_inches="tight")
    plt.close(candidate_figure)

    return pred_box


def save_multiple_predictions(sample_records):
    columns = 3
    rows = int(np.ceil(len(sample_records) / columns))

    figure, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))

    if rows == 1:
        axes = np.array([axes])

    for index, record in enumerate(sample_records):
        row = index // columns
        col = index % columns

        image = record["image"]
        image_name = record["image_name"]
        gt_box = record["gt_box"]
        pred_box = record["pred_box"]
        iou = record["iou"]

        output = image.copy()

        if gt_box is not None:
            output = draw_box(output, gt_box, (0, 255, 0), 2)

        if pred_box is not None:
            output = draw_box(output, pred_box, (0, 0, 255), 2)

        axes[row, col].imshow(bgr_to_rgb(output))
        axes[row, col].set_title(f"{image_name}\nIoU = {iou:.3f}")
        axes[row, col].axis("off")

    total_plots = rows * columns

    for index in range(len(sample_records), total_plots):
        row = index // columns
        col = index % columns
        axes[row, col].axis("off")

    figure.suptitle(
        "Prediction Examples: Red = Predicted Box | Green = Ground Truth",
        fontsize=16,
        fontweight="bold"
    )

    figure.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "prediction_examples_grid.png")
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_iou_distribution(iou_values):
    plt.figure(figsize=(10, 6))
    plt.hist(iou_values, bins=20)
    plt.title("IoU Distribution on Test Samples")
    plt.xlabel("IoU")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "iou_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_detection_summary(iou_values, threshold=0.5):
    successful = sum(1 for iou in iou_values if iou >= threshold)
    failed = len(iou_values) - successful

    labels = ["Detected Correctly", "Not Detected Correctly"]
    values = [successful, failed]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)
    plt.title(f"Detection Outcome at IoU Threshold {threshold}")
    plt.ylabel("Number of Images")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "detection_outcome_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_results_summary():
    csv_path = os.path.join("Graphs", "results_summary.csv")

    if not os.path.exists(csv_path):
        return []

    rows = []

    with open(csv_path, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] == "test":
                rows.append({
                    "config": row["config"],
                    "mean_iou": float(row["mean_iou"]),
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1_score": float(row["f1_score"]),
                    "mse": float(row["mse"])
                })

    rows = sorted(rows, key=lambda r: r["mean_iou"], reverse=True)
    return rows


def save_metrics_dashboard():
    rows = load_results_summary()

    if len(rows) == 0:
        return

    top_rows = rows[:5]

    configs = [row["config"] for row in top_rows]
    mean_iou = [row["mean_iou"] for row in top_rows]
    accuracy = [row["accuracy"] for row in top_rows]
    precision = [row["precision"] for row in top_rows]
    recall = [row["recall"] for row in top_rows]
    f1_score = [row["f1_score"] for row in top_rows]

    x = np.arange(len(configs))
    width = 0.15

    plt.figure(figsize=(16, 8))
    plt.bar(x - 2 * width, mean_iou, width, label="Mean IoU")
    plt.bar(x - width, accuracy, width, label="Accuracy")
    plt.bar(x, precision, width, label="Precision")
    plt.bar(x + width, recall, width, label="Recall")
    plt.bar(x + 2 * width, f1_score, width, label="F1 Score")

    plt.title("Top Configurations - Metrics Dashboard")
    plt.xlabel("Configuration")
    plt.ylabel("Score")
    plt.xticks(x, configs, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "metrics_dashboard_top_configs.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ensure_output_folder()

    image_files = read_image_files(IMAGE_DIR)

    random.seed(7)
    random.shuffle(image_files)

    all_records = []
    iou_values = []

    for image_name in image_files:
        img_path = join_path(IMAGE_DIR, image_name)
        label_path = get_label_path(image_name)

        image = load_image(img_path)

        if image is None:
            continue

        orig_h, orig_w = image.shape[:2]
        gt_label = load_yolo_label(label_path, orig_w, orig_h)

        if gt_label is None:
            continue

        gt_box = convert_yolo_box_to_resized_box(gt_label, orig_w, orig_h)

        resized, gray, blur, edges, morph, candidates, plate, bbox = process_image(image)

        pred_box = None

        if bbox is not None:
            x, y, w, h = bbox
            pred_box = (x, y, x + w, y + h)

        iou = compute_iou(gt_box, pred_box) if pred_box is not None else 0.0
        iou_values.append(iou)

        all_records.append({
            "image_name": image_name,
            "original_image": image,
            "image": resized,
            "gt_box": gt_box,
            "pred_box": pred_box,
            "iou": iou
        })

    all_records = sorted(
        all_records,
        key=lambda r: r["iou"],
        reverse=True
    )

    selected_records = all_records[:6]

    for record in selected_records:
        save_pipeline_stages(
            record["image_name"],
            record["original_image"],
            record["gt_box"]
        )

    if len(selected_records) > 0:
        save_multiple_predictions(selected_records)

    if len(iou_values) > 0:
        save_iou_distribution(iou_values)
        save_detection_summary(iou_values)

    save_metrics_dashboard()

    print(f"Presentation visuals saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
