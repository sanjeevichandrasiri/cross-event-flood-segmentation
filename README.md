# Cross-Event Flood Extent Segmentation (Sentinel-1 SAR)

Code for the paper:

> S. Chandrasiri and S. R. Liyanage, "Topology-Aware Evaluation of Cross-Event Generalization in Flood Extent Segmentation Using Sentinel-1 SAR Imagery," in *Proc. 8th International Conference on Advancements in Computing (ICAC)*, 2026.

This repository contains the code used for data loading, data splits, training, and evaluation of U-Net and Attention U-Net on the hand-labeled subset of the Sen1Floods11 dataset.

## Files

| File | Purpose |
| --- | --- |
| `generate_metadata.py` | Builds the list of hand-labeled image-mask pairs (446 pairs, 11 events) |
| `make_splits.py` | Creates the random split (312 train / 67 validation / 67 test) |
| `make_multi_event_splits.py` | Creates the cross-event splits (Sri Lanka, Ghana, Bolivia held out) |
| `dataset.py` | Data loader for the Sen1Floods11 hand-labeled subset |
| `model.py` | Standard U-Net, trained from scratch |
| `attention_unet.py` | Attention U-Net (same architecture as U-Net plus additive attention gates), trained from scratch |
| `train.py` | Model training |
| `evaluate.py` | Pixel-level and structural evaluation metrics |
| `splits/` | Exact training, validation, and test file lists used in the paper |
| `requirements.txt` | Python packages |

## 1. Installation

```bash
git clone https://github.com/sanjeevichandrasiri/cross-event-flood-segmentation.git
cd cross-event-flood-segmentation
pip install -r requirements.txt
```

## 2. Data

1. Download the Sen1Floods11 dataset: https://github.com/cloudtostreet/Sen1Floods11
2. Use only the hand-labeled Sentinel-1 images (`S1Hand`) and labels (`LabelHand`): 446 image-mask pairs from 11 flood events.
3. Set the data and output paths at the top of each script (`DATA_ROOT`, `METADATA_CSV`, `OUTPUT_DIR`, and the split paths) to the folders on your computer.

The dataset is not included in this repository.

## 3. Preprocessing (in `dataset.py`)

- Input: two-channel SAR image (VV and VH).
- NaN and infinite values are replaced with 0.
- Images are resized to 256 x 256 pixels using bilinear interpolation; masks using nearest-neighbor interpolation.
- Each band of each image is min-max normalized to [0, 1]: `x' = (x - x_min) / (x_max - x_min)`, where `x_min` and `x_max` are the minimum and maximum of that band in that image.
- Label values in Sen1Floods11: 1 = flooded, 0 = not flooded, -1 = "not sure".
- Pixels labeled 1 are treated as flooded; all other pixels, including "not sure" pixels, are treated as non-flooded.
- No data augmentation is used.

## 4. Data splits

All splits are created with scikit-learn's `train_test_split` using a fixed random seed (`random_state = 42`), stratified by flood event.

| Setting | Train | Validation | Test |
| --- | --- | --- | --- |
| Random split (70 / 15 / 15) | 312 | 67 | 67 |
| Sri Lanka held out | 323 | 81 | 42 |
| Ghana held out | 314 | 79 | 53 |
| Bolivia held out | 344 | 87 | 15 |

In the cross-event setting, all samples of the held-out event form the test set, and the remaining samples are split 80% training / 20% validation.

To create the splits:

```bash
python generate_metadata.py
python make_splits.py
python make_multi_event_splits.py
```

The exact split files used in the paper are provided in `splits/`. Note: `make_splits.py` also creates `splits/cross_event_split/` from an earlier single-event experiment; the results in the paper use `splits/random_split/` and `splits/multi_cross_event_splits/`.

## 5. Training

Set `MODEL_NAME` (`"unet"` or `"attention_unet"`), `EXPERIMENT` (`"random"` or `"cross_event"`), and `HELD_OUT_EVENT` (`"Sri-Lanka"`, `"Ghana"`, or `"Bolivia"`) at the top of `train.py`, then run:

```bash
python train.py
```

Training settings:

| Setting | Value |
| --- | --- |
| Initialization | Random (trained from scratch, no pretrained weights) |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Epochs | 20 |
| Model selection | Checkpoint with the highest validation Dice score |
| Learning-rate schedule | None |
| Augmentation | None |
| Loss | Binary cross-entropy + Dice (unweighted) |
| Training runs | One run per configuration |
| Hardware | Intel Core i7-5500U CPU (2.40 GHz), 16 GB RAM |
| Software | Python 3.11, PyTorch 2.12.1 (CPU) |

## 6. Evaluation

Use the same `MODEL_NAME`, `EXPERIMENT`, and `HELD_OUT_EVENT` settings at the top of `evaluate.py`, then run:

```bash
python evaluate.py
```

Metrics (computed for each test image and then averaged over the test set):

- **Pixel-level:** IoU, Dice, precision, recall, F1-score (F1 equals Dice for binary masks).
- **Boundary IoU:** boundary width of 2% of the image diagonal (7 pixels for 256 x 256 images).
- **Hausdorff Distance:** maximum distance in pixels between predicted and ground-truth flood regions; images where the prediction or ground truth has no flooded pixels are excluded from its average.
- **clDice:** skeleton-based structural similarity.
- **Betti Error:** difference in the number of connected components plus difference in the number of holes, using 4-connectivity.

A fixed threshold of 0.5 is used to convert probability maps to flood masks. Per-image results and summary results are saved as CSV files.

## Citation

```bibtex
@inproceedings{chandrasiri2026flood,
  title     = {Topology-Aware Evaluation of Cross-Event Generalization in Flood Extent Segmentation Using Sentinel-1 SAR Imagery},
  author    = {Chandrasiri, S. and Liyanage, S. R.},
  booktitle = {Proc. 8th International Conference on Advancements in Computing (ICAC)},
  year      = {2026}
}
```

## Acknowledgment

This work uses the Sen1Floods11 dataset:

> D. Bonafilia, B. Tellman, T. Anderson, and E. Issenberg, "Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1," in *Proc. IEEE/CVF CVPR Workshops*, 2020, pp. 835-845.