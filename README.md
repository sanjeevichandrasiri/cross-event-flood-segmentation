<<<<<<< HEAD
# Cross-Event Flood Extent Segmentation (Sentinel-1 SAR)

Code for the paper:

> S. Chandrasiri and S. R. Liyanage, "Topology-Aware Evaluation of Cross-Event Generalization in Flood Extent Segmentation Using Sentinel-1 SAR Imagery," in *Proc. 8th International Conference on Advancements in Computing (ICAC)*, 2026.

This repository contains the code used for data loading, data splits, training, and evaluation of U-Net and Attention U-Net on the hand-labeled subset of the Sen1Floods11 dataset.

## Files

| File | Purpose |
| --- | --- |
| `dataset.py` | Data loader for the Sen1Floods11 hand-labeled subset (clipping, normalization, resizing, "not sure" mask) |
| `make_splits.py` | Creates the random split (312 train / 66 validation / 68 test) |
| `make_multi_event_splits.py` | Creates the cross-event splits (Sri Lanka, Ghana, Bolivia held out) |
| `generate_metadata.py` | Builds the list of image-mask pairs used by the split scripts |
| `model.py` | U-Net with ResNet-34 encoder (ImageNet pretrained) |
| `attention_unet.py` | Attention U-Net (Oktay et al., 2018), trained from scratch |
| `train.py` | Model training |
| `evaluate.py` | Pixel-level and structural evaluation metrics |
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
3. Place them in a `data/` folder, or change the data path in the scripts.

The dataset is not included in this repository.

## 3. Preprocessing (in `dataset.py`)

- Input: two-channel SAR image (VV and VH), values in dB.
- NaN and infinite values are replaced with -50 dB.
- Values are clipped to [-50, 1] dB.
- Each band is min-max normalized to [0, 1].
- Images and masks are resized to 256 x 256 pixels.
- Label values: 1 = flooded, 0 = not flooded, -1 = "not sure".
- Pixels labeled -1 are excluded from the loss and all metrics using a validity mask.

## 4. Data splits

All splits use a fixed random seed (seed = 0).

| Setting | Train | Validation | Test |
| --- | --- | --- | --- |
| Random split | 312 | 66 | 68 |
| Sri Lanka held out | 324 | 80 | 42 |
| Ghana held out | 315 | 78 | 53 |
| Bolivia held out | 345 | 86 | 15 |

To create the splits:

```bash
python generate_metadata.py
python make_splits.py
python make_multi_event_splits.py
```

## 5. Training

```bash
python train.py
```

Training settings:

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Schedule | Cosine annealing, max 80 epochs |
| Early stopping | 15 epochs without validation Dice improvement |
| Loss | Weighted binary cross-entropy + Dice (valid pixels only) |
| Augmentation | Flips, 90-degree rotations, brightness/contrast jitter, Gaussian noise |
| Mixed precision | Yes |
| Seed | 0 |

## 6. Evaluation

```bash
python evaluate.py
```

Metrics:

- Pixel-level: IoU, Dice, precision, recall, F1-score
- Structural: Boundary IoU, Hausdorff Distance (95th percentile), clDice, Betti Error

A fixed threshold of 0.5 is used to convert probability maps to flood masks.

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
=======
# cross-event-flood-segmentation
Code for the ICAC 2026 paper on cross-event flood segmentation
>>>>>>> 6698090c4236c824928b7c368f1f7d4cad2712d1
