import cv2
import torch
import rasterio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Sen1FloodsDataset(Dataset):
    def __init__(self, csv_path, image_size=256):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def read_sar_image(self, path):
        """
        Reads Sentinel-1 SAR image.
        Sen1Floods11 S1 images usually contain 2 bands: VV and VH.
        Output shape: [2, H, W]
        """
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)

        # Replace invalid values
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Resize each band
        resized_bands = []
        for band in image:
            band = cv2.resize(
                band,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR
            )

            # Normalize each band using min-max normalization
            min_val = np.min(band)
            max_val = np.max(band)

            if max_val - min_val > 1e-6:
                band = (band - min_val) / (max_val - min_val)
            else:
                band = np.zeros_like(band)

            resized_bands.append(band)

        image = np.stack(resized_bands, axis=0)
        return image

    def read_mask(self, path):
        """
        Reads hand-labeled flood mask.
        Converts mask into binary:
        flood = 1
        non-flood/background = 0
        """
        with rasterio.open(path) as src:
            mask = src.read(1)

        mask = np.nan_to_num(mask, nan=0.0)

        mask = cv2.resize(
            mask.astype(np.float32),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert to binary mask
        # Any positive value is treated as flood
        mask = (mask > 0).astype(np.float32)

        # Shape: [1, H, W]
        mask = np.expand_dims(mask, axis=0)

        return mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        mask_path = row["mask_path"]

        image = self.read_sar_image(image_path)
        mask = self.read_mask(mask_path)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask