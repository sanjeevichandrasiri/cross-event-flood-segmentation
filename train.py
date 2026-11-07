import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import Sen1FloodsDataset
from model import UNet
from attention_unet import AttentionUNet


# =========================================================
# HELD-OUT EVENT SETTINGS
# =========================================================
# Use this only when EXPERIMENT = "cross_event"

# HELD_OUT_EVENT = "Sri-Lanka"
#HELD_OUT_EVENT = "Ghana"
HELD_OUT_EVENT = "Bolivia"


# =========================================================
# PATHS
# =========================================================
MULTI_CROSS_SPLIT_DIR = r"D:\Paper1_FloodGeneralization\splits\multi_cross_event_splits"

RANDOM_TRAIN_CSV = r"D:\Paper1_FloodGeneralization\splits\random_split\train.csv"
RANDOM_VAL_CSV = r"D:\Paper1_FloodGeneralization\splits\random_split\val.csv"

OUTPUT_DIR = r"D:\Paper1_FloodGeneralization\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# TRAINING SETTINGS
# =========================================================
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

# Model options:
# "unet" or "attention_unet"
MODEL_NAME = "unet"

# Experiment options:
# "random" or "cross_event"
EXPERIMENT = "cross_event"


# =========================================================
# LOSSES
# =========================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


# =========================================================
# METRICS
# =========================================================
def calculate_dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (
        preds.sum() + targets.sum() + smooth
    )

    return dice.item()


# =========================================================
# MODEL BUILDER
# =========================================================
def build_model(model_name):
    if model_name == "unet":
        return UNet(in_channels=2, out_channels=1)

    elif model_name == "attention_unet":
        return AttentionUNet(in_channels=2, out_channels=1)

    else:
        raise ValueError("MODEL_NAME must be 'unet' or 'attention_unet'")


# =========================================================
# PATH SELECTION
# =========================================================
def get_paths(experiment, model_name):
    if experiment == "random":
        train_csv = RANDOM_TRAIN_CSV
        val_csv = RANDOM_VAL_CSV
        model_path = os.path.join(
            OUTPUT_DIR,
            f"{model_name}_random_best.pth"
        )

    elif experiment == "cross_event":
        split_dir = os.path.join(
            MULTI_CROSS_SPLIT_DIR,
            f"heldout_{HELD_OUT_EVENT}"
        )

        train_csv = os.path.join(split_dir, "train.csv")
        val_csv = os.path.join(split_dir, "val.csv")

        safe_event_name = HELD_OUT_EVENT.replace("-", "_")

        model_path = os.path.join(
            OUTPUT_DIR,
            f"{model_name}_cross_event_{safe_event_name}_best.pth"
        )

    else:
        raise ValueError("EXPERIMENT must be 'random' or 'cross_event'")

    return train_csv, val_csv, model_path


# =========================================================
# TRAINING FUNCTION
# =========================================================
def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss, device):
    model.train()

    running_loss = 0.0
    running_dice = 0.0

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss_bce = bce_loss(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = loss_bce + loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += calculate_dice_score(logits.detach(), masks)

    avg_loss = running_loss / len(loader)
    avg_dice = running_dice / len(loader)

    return avg_loss, avg_dice


# =========================================================
# VALIDATION FUNCTION
# =========================================================
def validate(model, loader, bce_loss, dice_loss, device):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = loss_bce + loss_dice

            running_loss += loss.item()
            running_dice += calculate_dice_score(logits, masks)

    avg_loss = running_loss / len(loader)
    avg_dice = running_dice / len(loader)

    return avg_loss, avg_dice


# =========================================================
# MAIN
# =========================================================
def main():
    train_csv, val_csv, model_path = get_paths(EXPERIMENT, MODEL_NAME)

    print("Model:", MODEL_NAME)
    print("Experiment:", EXPERIMENT)

    if EXPERIMENT == "cross_event":
        print("Held-out event:", HELD_OUT_EVENT)

    print("Train CSV:", train_csv)
    print("Validation CSV:", val_csv)
    print("Model will be saved to:", model_path)

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = Sen1FloodsDataset(train_csv, image_size=IMAGE_SIZE)
    val_dataset = Sen1FloodsDataset(val_csv, image_size=IMAGE_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = build_model(MODEL_NAME).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_dice = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_dice = train_one_epoch(
            model,
            train_loader,
            optimizer,
            bce_loss,
            dice_loss,
            device
        )

        val_loss, val_dice = validate(
            model,
            val_loader,
            bce_loss,
            dice_loss,
            device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved. Val Dice: {best_val_dice:.4f}")

    print("\nTraining complete.")
    print("Best validation Dice:", best_val_dice)
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()