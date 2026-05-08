import os
import csv

import matplotlib.pyplot as plt

from configs import CONFIGS

from evaluation.metrics import (
    compute_iou,
    compute_mean_iou,
    compute_accuracy,
    compute_precision_recall_f1,
    compute_mse
)

from src.edge_detection import detect_edges
from src.morphology import apply_morphology
from src.plate_localization import localize_plate
from src.preprocessing import preprocess_image
from src.utils import list_files, join_path, load_image, load_yolo_label


RESIZE_WIDTH = 600
GRAPHS_DIR = "Graphs"

SPLITS = {
    "train": ("data/train/images", "data/train/labels"),
    "valid": ("data/valid/images", "data/valid/labels"),
    "test": ("data/test/images", "data/test/labels"),
}


def ensure_graphs_folder():
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)


def run_evaluation(
    image_dir,
    label_dir,
    edge_method="canny",
    low_threshold=50,
    high_threshold=150,
    morph_method="closing",
    kernel_size=(3, 3),
    iterations=1,
    use_clahe=False,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
    blur_kernel_size=(5, 5),
    blur_sigma=0,
    ideal_density=0.22
):
    iou_scores = []

    if not os.path.exists(image_dir):
        print(f"  Directory not found: {image_dir} — skipping.")
        return iou_scores

    images = list_files(image_dir)

    for img_name in images:
        img_path = join_path(image_dir, img_name)

        label_path = join_path(
            label_dir,
            img_name.replace(".jpg", ".txt")
                    .replace(".png", ".txt")
                    .replace(".jpeg", ".txt")
        )

        image = load_image(img_path)

        if image is None:
            continue

        orig_h, orig_w = image.shape[:2]

        gt = load_yolo_label(label_path, orig_w, orig_h)

        if gt is None:
            continue

        _, gt_x1, gt_y1, gt_x2, gt_y2 = gt

        resized, gray, blur = preprocess_image(
            external_image=image,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma
        )

        edges = detect_edges(
            blur,
            method=edge_method,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )

        morph = apply_morphology(
            edges,
            method=morph_method,
            kernel_size=kernel_size,
            iterations=iterations
        )

        plate, bbox = localize_plate(
            resized,
            morph,
            ideal_density=ideal_density
        )

        scale = RESIZE_WIDTH / orig_w

        gt_box = (
            int(gt_x1 * scale),
            int(gt_y1 * scale),
            int(gt_x2 * scale),
            int(gt_y2 * scale),
        )

        if bbox is None:
            iou_scores.append(0)
            continue

        x, y, bw, bh = bbox

        pred_box = (
            x,
            y,
            x + bw,
            y + bh
        )

        iou = compute_iou(gt_box, pred_box)
        iou_scores.append(iou)

    return iou_scores


def calculate_metrics(iou_scores):
    mean_iou = compute_mean_iou(iou_scores)
    accuracy = compute_accuracy(iou_scores)
    precision, recall, f1 = compute_precision_recall_f1(iou_scores)
    mse = compute_mse(iou_scores)

    return {
        "mean_iou": mean_iou,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse": mse
    }


def print_metric_results(metrics):
    print(f"Mean IoU  : {metrics['mean_iou']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"MSE       : {metrics['mse']:.4f}")


def save_results_to_csv(all_split_results):
    csv_path = os.path.join(GRAPHS_DIR, "results_summary.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow([
            "config",
            "split",
            "mean_iou",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "mse"
        ])

        for result in all_split_results:
            writer.writerow([
                result["config"],
                result["split"],
                f"{result['mean_iou']:.4f}",
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['mse']:.4f}",
            ])

    print(f"\nCSV saved to: {csv_path}")


def save_bar_chart(results, metric_key, metric_label, filename):
    names = [r["name"] for r in results]
    values = [r[metric_key] for r in results]

    plt.figure(figsize=(14, 7))
    plt.bar(names, values)
    plt.title(f"Test {metric_label} Comparison")
    plt.xlabel("Configuration")
    plt.ylabel(metric_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = os.path.join(GRAPHS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Graph saved: {save_path}")


def save_precision_recall_chart(results):
    names = [r["name"] for r in results]
    precision_values = [r["precision"] for r in results]
    recall_values = [r["recall"] for r in results]

    x = list(range(len(names)))
    width = 0.35

    plt.figure(figsize=(14, 7))
    plt.bar([i - width / 2 for i in x], precision_values, width, label="Precision")
    plt.bar([i + width / 2 for i in x], recall_values, width, label="Recall")

    plt.title("Test Precision vs Recall Comparison")
    plt.xlabel("Configuration")
    plt.ylabel("Score")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(GRAPHS_DIR, "test_precision_recall_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Graph saved: {save_path}")


def save_top_configs_all_metrics_chart(results, top_n=5):
    top_results = results[:top_n]

    metric_keys = ["mean_iou", "accuracy", "precision", "recall", "f1"]
    metric_labels = ["Mean IoU", "Accuracy", "Precision", "Recall", "F1 Score"]

    names = [r["name"] for r in top_results]

    x = list(range(len(names)))
    width = 0.15

    plt.figure(figsize=(15, 8))

    for idx, key in enumerate(metric_keys):
        values = [r[key] for r in top_results]
        positions = [i + (idx - 2) * width for i in x]
        plt.bar(positions, values, width, label=metric_labels[idx])

    plt.title(f"Top {top_n} Configurations - Test Metrics Comparison")
    plt.xlabel("Configuration")
    plt.ylabel("Score")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(GRAPHS_DIR, "top_configs_all_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Graph saved: {save_path}")


def save_best_config_splits_chart(all_split_results, best_config_name):
    split_order = ["train", "valid", "test"]

    best_rows = [
        r for r in all_split_results
        if r["config"] == best_config_name
    ]

    best_rows = sorted(
        best_rows,
        key=lambda r: split_order.index(r["split"])
    )

    metric_keys = ["mean_iou", "accuracy", "precision", "recall", "f1"]
    metric_labels = ["Mean IoU", "Accuracy", "Precision", "Recall", "F1 Score"]

    x = list(range(len(split_order)))
    width = 0.15

    plt.figure(figsize=(12, 7))

    for idx, key in enumerate(metric_keys):
        values = [r[key] for r in best_rows]
        positions = [i + (idx - 2) * width for i in x]
        plt.bar(positions, values, width, label=metric_labels[idx])

    plt.title(f"Best Configuration Metrics Across Splits\n{best_config_name}")
    plt.xlabel("Dataset Split")
    plt.ylabel("Score")
    plt.xticks(x, [s.upper() for s in split_order])
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(GRAPHS_DIR, "best_config_splits_metrics.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Graph saved: {save_path}")


def save_all_graphs(all_results, all_split_results):
    ensure_graphs_folder()

    save_results_to_csv(all_split_results)

    save_bar_chart(
        all_results,
        metric_key="mean_iou",
        metric_label="Mean IoU",
        filename="test_mean_iou_ranking.png"
    )

    save_bar_chart(
        all_results,
        metric_key="accuracy",
        metric_label="Accuracy",
        filename="test_accuracy_ranking.png"
    )

    save_bar_chart(
        all_results,
        metric_key="f1",
        metric_label="F1 Score",
        filename="test_f1_ranking.png"
    )

    save_bar_chart(
        all_results,
        metric_key="mse",
        metric_label="MSE",
        filename="test_mse_ranking.png"
    )

    save_precision_recall_chart(all_results)

    save_top_configs_all_metrics_chart(all_results, top_n=5)

    if len(all_results) > 0:
        best_config_name = all_results[0]["name"]
        save_best_config_splits_chart(all_split_results, best_config_name)


def main():
    ensure_graphs_folder()

    separator = "=" * 70
    all_results = []
    all_split_results = []

    for config in CONFIGS:
        print(f"\n{separator}")
        print(f"  Config: {config['name']}")
        print(separator)

        config_results = {}

        for split_name, (img_dir, lbl_dir) in SPLITS.items():
            print(f"\n  Split: {split_name.upper()}")
            print(f"  {'-' * 40}")

            iou_scores = run_evaluation(
                img_dir,
                lbl_dir,
                edge_method=config["edge_method"],
                low_threshold=config["low_threshold"],
                high_threshold=config["high_threshold"],
                morph_method=config["morph_method"],
                kernel_size=config["kernel_size"],
                iterations=config["iterations"],
                use_clahe=config.get("use_clahe", False),
                clahe_clip_limit=config.get("clahe_clip_limit", 2.0),
                clahe_tile_grid_size=config.get("clahe_tile_grid_size", (8, 8)),
                blur_kernel_size=config.get("blur_kernel_size", (5, 5)),
                blur_sigma=config.get("blur_sigma", 0),
                ideal_density=config.get("ideal_density", 0.22)
            )

            if len(iou_scores) == 0:
                print("  No labelled images found.")
                continue

            metrics = calculate_metrics(iou_scores)
            config_results[split_name] = metrics

            all_split_results.append({
                "config": config["name"],
                "split": split_name,
                **metrics
            })

            print_metric_results(metrics)

        if "test" in config_results:
            all_results.append({
                "name": config["name"],
                **config_results["test"]
            })

    all_results = sorted(
        all_results,
        key=lambda x: x["mean_iou"],
        reverse=True
    )

    print(f"\n{separator}")
    print("  BEST CONFIGURATIONS BASED ON TEST Mean IoU")
    print(separator)

    for rank, result in enumerate(all_results, start=1):
        print(f"\nRank {rank}: {result['name']}")
        print(f"Mean IoU  : {result['mean_iou']:.4f}")
        print(f"Accuracy  : {result['accuracy']:.4f}")
        print(f"Precision : {result['precision']:.4f}")
        print(f"Recall    : {result['recall']:.4f}")
        print(f"F1 Score  : {result['f1']:.4f}")
        print(f"MSE       : {result['mse']:.4f}")

    print(f"\n{separator}")
    print("  Saving graphs and CSV results...")
    print(separator)

    save_all_graphs(all_results, all_split_results)

    print(f"\n{separator}")
    print("  Evaluation complete.")
    print(f"  Graphs saved in folder: {GRAPHS_DIR}")
    print(separator)


if __name__ == "__main__":
    main()
