import numpy as np


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    intersection = inter_width * inter_height

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - intersection

    if union <= 0:
        return 0

    return intersection / union

def evaluate_detection(iou, threshold=0.5):
    return 1 if iou >= threshold else 0

def compute_accuracy(iou_list, threshold=0.5):
    if len(iou_list) == 0:
        return 0

    correct = sum(1 for iou in iou_list if iou >= threshold)
    return correct / len(iou_list)


def compute_precision_recall_f1(iou_list, threshold=0.5):
    TP = sum(1 for iou in iou_list if iou >= threshold)
    FP = sum(1 for iou in iou_list if 0 < iou < threshold)
    FN = sum(1 for iou in iou_list if iou == 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return precision, recall, f1

def compute_mean_iou(iou_list):
    if len(iou_list) == 0:
        return 0.0
    return float(np.mean(iou_list))

def compute_mse(iou_list, threshold=0.5):
    if len(iou_list) == 0:
        return 0.0
    gt     = np.ones(len(iou_list))          # every image has a plate
    pred   = np.array([1 if iou >= threshold else 0 for iou in iou_list])
    return float(np.mean((gt - pred) ** 2))

def print_metrics(iou_list, method_name="", threshold=0.5):
    mean_iou           = compute_mean_iou(iou_list)
    accuracy           = compute_accuracy(iou_list, threshold)
    precision, recall, f1 = compute_precision_recall_f1(iou_list, threshold)
    mse                = compute_mse(iou_list, threshold)

    label = f"[{method_name}] " if method_name else ""
    print(f"{label}Mean IoU  : {mean_iou:.4f}")
    print(f"{label}Accuracy  : {accuracy:.4f}")
    print(f"{label}Precision : {precision:.4f}")
    print(f"{label}Recall    : {recall:.4f}")
    print(f"{label}F1 Score  : {f1:.4f}")
    print(f"{label}MSE       : {mse:.4f}")