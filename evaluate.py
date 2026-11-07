import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import skeletonize
from skimage.measure import label
from torch.utils.data import DataLoader

from dataset import Sen1FloodsDataset
from model import UNet
from attention_unet import AttentionUNet


# =========================================================
# SETTINGS
# =========================================================
IMAGE_SIZE = 256
BATCH_SIZE = 1
THRESHOLD = 0.5

# Model options:
# "unet" or "attention_unet"
MODEL_NAME = "unet"

# Experiment options:
# "random" or "cross_event"
EXPERIMENT = "cross_event"

# Held-out event options:
# "Sri-Lanka", "Ghana", or "Bolivia"
# Used only when EXPERIMENT = "cross_event"
# HELD_OUT_EVENT = "Sri-Lanka"
# HELD_OUT_EVENT = "Ghana"
HELD_OUT_EVENT = "Bolivia"

RANDOM_TEST_CSV = r"D:\Paper1_FloodGeneralization\splits\random_split\test.csv"
MULTI_CROSS_SPLIT_DIR = r"D:\Paper1_FloodGeneralization\splits\multi_cross_event_splits"

OUTPUT_DIR = r"D:\Paper1_FloodGeneralization\evaluation_outputs"


# =========================================================
# METRICS
# =========================================================
EPS = 1e-7


def build_model(model_name):
    if model_name == "unet":
        return UNet(in_channels=2, out_channels=1)

    elif model_name == "attention_unet":
        return AttentionUNet(in_channels=2, out_channels=1)

    else:
        raise ValueError("MODEL_NAME must be 'unet' or 'attention_unet'")


def compute_binary_metrics(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()

    iou = tp / (tp + fp + fn + EPS)
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)

    return iou, dice, precision, recall, f1


def get_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    diag_len = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * diag_len)))

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=dilation)
    boundary = mask.astype(np.uint8) - eroded
    boundary = np.clip(boundary, 0, 1)

    return boundary


def compute_boundary_iou(pred, target):
    pred_boundary = get_boundary(pred)
    target_boundary = get_boundary(target)

    intersection = np.logical_and(pred_boundary, target_boundary).sum()
    union = np.logical_or(pred_boundary, target_boundary).sum()

    return intersection / (union + EPS)


def compute_hausdorff_distance(pred, target):
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan

    hd_forward = directed_hausdorff(pred_points, target_points)[0]
    hd_backward = directed_hausdorff(target_points, pred_points)[0]

    return max(hd_forward, hd_backward)


def compute_cldice(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    pred_skeleton = skeletonize(pred)
    target_skeleton = skeletonize(target)

    tprec = np.logical_and(pred_skeleton, target).sum() / (pred_skeleton.sum() + EPS)
    tsens = np.logical_and(target_skeleton, pred).sum() / (target_skeleton.sum() + EPS)

    cldice = (2 * tprec * tsens) / (tprec + tsens + EPS)

    return cldice


def compute_betti_error(pred, target):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    # beta_0: number of connected components
    pred_components = label(pred, connectivity=1).max()
    target_components = label(target, connectivity=1).max()

    # Approximate beta_1: number of holes
    pred_inv = 1 - pred
    target_inv = 1 - target

    pred_holes = max(0, label(pred_inv, connectivity=1).max() - 1)
    target_holes = max(0, label(target_inv, connectivity=1).max() - 1)

    beta0_error = abs(pred_components - target_components)
    beta1_error = abs(pred_holes - target_holes)

    return beta0_error + beta1_error


# =========================================================
# UTILITIES
# =========================================================
def get_paths(experiment, model_name):
    if experiment == "random":
        test_csv = RANDOM_TEST_CSV
        model_path = rf"D:\Paper1_FloodGeneralization\outputs\{model_name}_random_best.pth"
        save_dir = os.path.join(OUTPUT_DIR, f"{model_name}_random_eval")

    elif experiment == "cross_event":
        split_dir = os.path.join(MULTI_CROSS_SPLIT_DIR, f"heldout_{HELD_OUT_EVENT}")
        test_csv = os.path.join(split_dir, "test.csv")

        safe_event_name = HELD_OUT_EVENT.replace("-", "_")

        model_path = rf"D:\Paper1_FloodGeneralization\outputs\{model_name}_cross_event_{safe_event_name}_best.pth"

        save_dir = os.path.join(
            OUTPUT_DIR,
            f"{model_name}_cross_event_{safe_event_name}_eval"
        )

    else:
        raise ValueError("EXPERIMENT must be 'random' or 'cross_event'")

    return test_csv, model_path, save_dir


def save_prediction_figure(image_tensor, mask_tensor, pred_mask, save_path):
    image = image_tensor.cpu().numpy()       # shape: [C, H, W]
    mask = mask_tensor.cpu().numpy()[0]      # shape: [H, W]

    # Display first SAR channel only
    image_show = image[0]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(image_show, cmap="gray")
    axes[0].set_title("Input SAR")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN EVALUATION
# =========================================================
def main():
    test_csv, model_path, save_dir = get_paths(EXPERIMENT, MODEL_NAME)

    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("Model:", MODEL_NAME)
    print("Experiment:", EXPERIMENT)

    if EXPERIMENT == "cross_event":
        print("Held-out event:", HELD_OUT_EVENT)

    print("Threshold:", THRESHOLD)
    print("Test CSV:", test_csv)
    print("Model path:", model_path)
    print("Save directory:", save_dir)

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_dataset = Sen1FloodsDataset(test_csv, image_size=IMAGE_SIZE)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = build_model(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            pred = (probs.cpu().numpy()[0, 0] > THRESHOLD).astype(np.uint8)
            target = masks.cpu().numpy()[0, 0].astype(np.uint8)

            iou, dice, precision, recall, f1 = compute_binary_metrics(pred, target)
            boundary_iou = compute_boundary_iou(pred, target)
            hausdorff = compute_hausdorff_distance(pred, target)
            cldice = compute_cldice(pred, target)
            betti_error = compute_betti_error(pred, target)

            row = {
                "index": idx,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "boundary_iou": boundary_iou,
                "hausdorff_distance": hausdorff,
                "cldice": cldice,
                "betti_error": betti_error
            }

            results.append(row)

            # Save first 10 prediction figures
            if idx < 10:
                fig_path = os.path.join(figures_dir, f"prediction_{idx}.png")
                save_prediction_figure(images[0].cpu(), masks[0].cpu(), pred, fig_path)

    results_df = pd.DataFrame(results)

    metrics_list = [
        "iou",
        "dice",
        "precision",
        "recall",
        "f1",
        "boundary_iou",
        "hausdorff_distance",
        "cldice",
        "betti_error"
    ]

    summary_rows = []

    for metric in metrics_list:
        values = results_df[metric].dropna()

        summary_rows.append({
            "metric": metric,
            "mean": values.mean(),
            "std": values.std()
        })

    summary_df = pd.DataFrame(summary_rows)

    results_csv_path = os.path.join(save_dir, "all_image_results.csv")
    summary_csv_path = os.path.join(save_dir, "summary_results.csv")

    results_df.to_csv(results_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nSaved per-image results to:")
    print(results_csv_path)

    print("\nSaved summary results to:")
    print(summary_csv_path)

    print("\nSummary Results:")
    print(summary_df)


if __name__ == "__main__":
    main()